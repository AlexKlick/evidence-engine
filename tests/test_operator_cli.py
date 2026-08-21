"""Operator-loop CLI: outcomes record/list, calibration, report regen, review."""

from __future__ import annotations

from typer.testing import CliRunner

from conftest import FakeEmbedder
from evidence_engine.cli import app
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo
from evidence_engine.store.models import Experiment, ProblemHypothesis

runner = CliRunner()
VERTICAL = "local-ai-tooling"


def seed(session_factory, settings) -> None:
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)


def test_outcomes_record_list_and_calibration(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_REPORTS_DIR", str(settings.reports_dir))

    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        experiment_id = experiment.id
        idea_id = experiment.idea_id

    result = runner.invoke(
        app,
        [
            "outcomes", "record", "-e", experiment_id, "-k", "deposit_paid",
            "--value", '{"amount": 50}', "--status", "running", "--decision", "advance",
        ],
    )
    assert result.exit_code == 0, result.output

    with session_factory() as session:
        outcomes = session.query(repo.Outcome).all()
        assert len(outcomes) == 1
        assert outcomes[0].kind == "deposit_paid"
        assert outcomes[0].value == {"amount": 50}
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "running"
        assert refreshed.decision == "advance"

    listed = runner.invoke(app, ["outcomes", "list", "-e", experiment_id])
    assert listed.exit_code == 0
    assert "deposit_paid" in listed.output

    calibration = runner.invoke(app, ["calibration"])
    assert calibration.exit_code == 0, calibration.output
    assert "outcomes: 1" in calibration.output
    assert "ready_to_fit: False" in calibration.output
    assert idea_id in calibration.output


def test_report_regenerates_from_store_without_collection(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_REPORTS_DIR", str(settings.reports_dir))

    before = set(settings.reports_dir.glob(f"{VERTICAL}-*.md"))
    result = runner.invoke(app, ["report", "-y", VERTICAL])
    assert result.exit_code == 0, result.output
    after = set(settings.reports_dir.glob(f"{VERTICAL}-*.md"))
    new_files = after - before
    assert len(new_files) == 1
    markdown = new_files.pop().read_text(encoding="utf-8")
    for section in ("# Opportunity report", "## Scored ideas", "## Compliance footer"):
        assert section in markdown
    assert "searxng" in markdown


def test_review_via_cli_updates_hypothesis(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        hypothesis = (
            session.query(ProblemHypothesis).filter_by(vertical=VERTICAL).first()
        )
        assert hypothesis is not None
        hypothesis_id = hypothesis.id

    result = runner.invoke(
        app,
        [
            "review", "-H", hypothesis_id,
            "-b", "solo self-hoster",
            "-c", "search",
            "-t", "preorder page",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "applied:" in result.output
    assert "rescored" in result.output

    with session_factory() as session:
        refreshed = repo.get_hypothesis(session, hypothesis_id)
        assert refreshed is not None
        assert refreshed.buyer == "solo self-hoster"
        for idea in repo.ideas_for_hypothesis(session, hypothesis_id):
            assert not [g for g, ok in (idea.gates or {}).items() if not ok]


def test_ideas_band_filter(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    result = runner.invoke(app, ["ideas", "-y", VERTICAL, "--band", "collect_more"])
    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.startswith("idea_")]
    assert lines, "fake-seeded run should land ideas in collect_more"
    assert all("collect_more" in line for line in lines)
