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

    # legal continuation: running -> stopped via the validated outcomes path
    finished = runner.invoke(
        app,
        [
            "outcomes", "record", "-e", experiment_id, "-k", "payment_received",
            "--status", "stopped", "--decision", "advance",
        ],
    )
    assert finished.exit_code == 0, finished.output
    with session_factory() as session:
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "stopped"
        assert session.query(repo.Outcome).count() == 2


def test_outcomes_record_rejects_illegal_status(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        experiment_id = experiment.id

    result = runner.invoke(
        app,
        [
            "outcomes", "record", "-e", experiment_id, "-k", "deposit_paid",
            "--status", "stopped",
        ],
    )
    assert result.exit_code == 1
    assert "draft" in result.output
    assert "running" in result.output
    with session_factory() as session:  # nothing recorded on rejection
        assert session.query(repo.Outcome).count() == 0
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "draft"


def test_experiments_cli_list_start_stop(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        experiment_id = experiment.id

    listed = runner.invoke(app, ["experiments", "list", "-y", VERTICAL])
    assert listed.exit_code == 0, listed.output
    assert experiment_id in listed.output
    assert "cap" in listed.output  # no price evidence yet -> cap unset, not $504.90

    # heuristic seed carries no price signals -> operator sets the price
    started = runner.invoke(
        app, ["experiments", "start", "-e", experiment_id, "--price", "12"]
    )
    assert started.exit_code == 0, started.output
    assert "spend cap: $61.2" in started.output  # 12 * 0.85 * 6
    assert "reviewed:operator" in started.output
    assert "stop condition" in started.output

    with session_factory() as session:  # stamp persisted, not just echoed
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        guardrail = refreshed.spec["economics_guardrail"]
        assert guardrail["price_monthly"] == 12
        assert guardrail["cac_ceiling"] == 61.2
        assert guardrail["price_provenance"].startswith("reviewed:operator")

    again = runner.invoke(app, ["experiments", "start", "-e", experiment_id])
    assert again.exit_code == 1
    assert "legal transitions" in again.output

    stopped = runner.invoke(
        app, ["experiments", "stop", "-e", experiment_id, "--decision", "kill"]
    )
    assert stopped.exit_code == 0, stopped.output

    filtered = runner.invoke(app, ["experiments", "list", "--status", "stopped"])
    assert filtered.exit_code == 0, filtered.output
    assert experiment_id in filtered.output
    assert "kill" in filtered.output


def test_experiments_start_without_price_is_gated(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    """No derived price + no --price -> refused, with the fix in the message."""
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        experiment_id = experiment.id

    result = runner.invoke(app, ["experiments", "start", "-e", experiment_id])
    assert result.exit_code == 1
    assert "--price" in result.output
    assert "price evidence" in result.output
    with session_factory() as session:
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "draft"


def test_experiments_start_derived_price_needs_no_flag(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    """Claims with price signals -> derived median starts without --price."""
    from conftest import PricedFakeLLM

    Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=PricedFakeLLM(),
    ).run(VERTICAL, limit=3, use_llm=True)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        assert experiment.spec["economics_guardrail"]["price_monthly"] == 9
        experiment_id = experiment.id

    result = runner.invoke(app, ["experiments", "start", "-e", experiment_id])
    assert result.exit_code == 0, result.output
    assert "spend cap: $45.9" in result.output  # 9 * 0.85 * 6
    assert "derived:median(n=" in result.output
    assert "price:" in result.output


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


def _patch_pipeline(monkeypatch, settings, session_factory) -> None:
    import evidence_engine.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "_pipeline",
        lambda: Pipeline(
            settings=settings,
            session_factory=session_factory,
            embedder=FakeEmbedder(),
        ),
    )


def test_pipeline_all_cli(fake_adapters, session_factory, settings, monkeypatch) -> None:
    _patch_pipeline(monkeypatch, settings, session_factory)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_CONFIG_DIR", str(settings.config_dir))
    monkeypatch.setenv("EE_REPORTS_DIR", str(settings.reports_dir))
    monkeypatch.setenv("EE_IDEAS_DIR", str(settings.ideas_dir))

    result = runner.invoke(app, ["pipeline", "--all", "--no-llm", "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "3 verticals ok · 0 failed" in result.output
    assert result.output.count("vertical:") == 3


def test_pipeline_and_collect_reject_bad_vertical_args(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch, settings, session_factory)
    for argv in (
        ["pipeline"],
        ["pipeline", "--all", "-y", "local-ai-tooling"],
        ["collect"],
        ["collect", "--all", "-y", "local-ai-tooling"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 1, argv
        assert "error:" in result.output


def test_pipeline_all_cli_exits_nonzero_when_a_vertical_fails(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    _patch_pipeline(monkeypatch, settings, session_factory)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_CONFIG_DIR", str(settings.config_dir))
    monkeypatch.setenv("EE_REPORTS_DIR", str(settings.reports_dir))
    monkeypatch.setenv("EE_IDEAS_DIR", str(settings.ideas_dir))

    import evidence_engine.cli as cli_module

    real_run_all = cli_module.Pipeline.run_all

    def failing_run_all(self, limit=None, use_llm=None):
        batch = real_run_all(self, limit=limit, use_llm=use_llm)
        batch.failed["solo-dev-saas"] = "boom"
        batch.results = [r for r in batch.results if r.vertical != "solo-dev-saas"]
        return batch

    monkeypatch.setattr(cli_module.Pipeline, "run_all", failing_run_all)
    result = runner.invoke(app, ["pipeline", "--all", "--no-llm", "--limit", "2"])
    assert result.exit_code == 1
    assert "2 verticals ok · 1 failed" in result.output
    assert "failed: solo-dev-saas (boom)" in result.output


def test_review_show_cli(fake_adapters, session_factory, settings, monkeypatch) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    with session_factory() as session:
        hypothesis = (
            session.query(ProblemHypothesis).filter_by(vertical=VERTICAL).first()
        )
        assert hypothesis is not None
        hypothesis_id = hypothesis.id

    result = runner.invoke(app, ["review", "--show", "-H", hypothesis_id])
    assert result.exit_code == 0, result.output
    assert "Evidence pack" in result.output
    assert "buyer_identified" in result.output
    assert "http" in result.output  # excerpts carry URLs
    assert f"ee review -H {hypothesis_id}" in result.output


def test_review_show_arg_errors(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    seed(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)

    no_id = runner.invoke(app, ["review", "--show"])
    assert no_id.exit_code == 1
    assert "--show needs --hypothesis" in no_id.output

    with_update = runner.invoke(
        app, ["review", "--show", "-H", "hyp_anything", "--buyer", "x"]
    )
    assert with_update.exit_code == 1
    assert "read-only" in with_update.output

    unknown = runner.invoke(app, ["review", "--show", "-H", "hyp_missing"])
    assert unknown.exit_code == 1
    assert "not found" in unknown.output
