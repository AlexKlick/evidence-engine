"""Decision support: review queue, competitor map, velocity, landing emitter."""

from __future__ import annotations

from sqlalchemy import select

from conftest import FakeEmbedder, FakeLLM
from evidence_engine.config import load_rubric
from evidence_engine.ideas.competitors import (
    alternatives_overview,
    competitor_map,
    render_competitors_markdown,
)
from evidence_engine.ideas.review import apply_review, review_queue
from evidence_engine.pipeline import Pipeline
from evidence_engine.report import velocity_summary
from evidence_engine.store import repository as repo

VERTICAL = "local-ai-tooling"


def seed(session_factory, settings, use_llm: bool = True):
    return Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=FakeLLM() if use_llm else None,
    ).run(VERTICAL, limit=3, use_llm=use_llm)


def test_review_queue_lists_missing_gates_then_empties(
    fake_adapters, session_factory, settings
) -> None:
    seed(session_factory, settings, use_llm=False)  # heuristic: no personas
    with session_factory() as session:
        queue = review_queue(session, VERTICAL)
    assert queue, "heuristic run must leave buyer gates open"
    assert all("buyer_identified" in entry["missing"] for entry in queue)

    # review the top entry -> its hypothesis leaves the queue
    top = queue[0]
    with session_factory() as session:
        apply_review(
            session,
            top["hypothesis_id"],
            load_rubric(settings),
            {"buyer": "solo operator", "channel": "search", "smallest_paid_test": "p"},
        )
        session.commit()
    with session_factory() as session:
        remaining = [
            entry
            for entry in review_queue(session, VERTICAL)
            if entry["hypothesis_id"] == top["hypothesis_id"]
        ]
    assert remaining == []


def test_competitor_map_from_extracted_incumbents(
    fake_adapters, session_factory, settings
) -> None:
    seed(session_factory, settings, use_llm=True)  # FakeLLM claims cite BigTool
    with session_factory() as session:
        entries = competitor_map(session, VERTICAL)
        assert entries, "incumbents from claims must surface"
        bigtool = [e for e in entries if e.name == "bigtool"]
        assert bigtool and bigtool[0].mention_count >= 1

        overview = alternatives_overview(session, VERTICAL)
        assert overview.manual_workaround_claims >= 1  # FakeLLM sets workarounds
        markdown = "\n".join(render_competitors_markdown(overview))
        assert "bigtool" in markdown
        assert "Alternative classes" in markdown
        assert "rights-gated web_crawler" in markdown  # pricing caveat disclosed


def test_velocity_needs_two_days(fake_adapters, session_factory, settings) -> None:
    from datetime import UTC, datetime, timedelta

    from evidence_engine.store.models import EvidenceEvent

    seed(session_factory, settings, use_llm=False)
    with session_factory() as session:
        assert velocity_summary(session, VERTICAL) is None  # single day

        # backdate half the evidence to yesterday -> two observation days
        rows = list(
            session.execute(
                select(EvidenceEvent).where(EvidenceEvent.vertical == VERTICAL)
            ).scalars()
        )
        yesterday = datetime.now(UTC) - timedelta(days=1)
        for row in rows[: len(rows) // 2]:
            row.fetched_at = yesterday
        session.commit()

        summary = velocity_summary(session, VERTICAL)
    assert summary is not None
    assert summary["days"] == 2
    assert summary["recent"] >= 1
    assert summary["baseline"] >= 1
    assert summary["velocity"] is not None


def test_report_sections_reflect_gate_state(
    fake_adapters, session_factory, settings
) -> None:
    # LLM-mode run: personas fill buyers -> gates pass, but the FakeLLM seed
    # crosses the paid_validation threshold, so its hypotheses surface in the
    # review queue as awaiting operator sign-off (band capped at interview),
    # and incumbents exist -> competitor pressure section present.
    result = seed(session_factory, settings, use_llm=True)
    markdown = result.report_path.read_text(encoding="utf-8")
    assert "## Competitor pressure" in markdown
    assert "bigtool" in markdown
    assert "awaiting review sign-off (paid_validation capped)" in markdown
    assert "velocity: n/a (single day of evidence so far)" in markdown


def test_report_queue_section_when_gates_open(
    fake_adapters, session_factory, settings
) -> None:
    # Heuristic run: no personas -> queue section present, no incumbents.
    result = seed(session_factory, settings, use_llm=False)
    markdown = result.report_path.read_text(encoding="utf-8")
    assert "## Review queue" in markdown
    assert "buyer_identified" in markdown
    assert "## Competitor pressure" not in markdown


def test_landing_cli_emits_drafts(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    from typer.testing import CliRunner

    from evidence_engine.cli import app

    seed(session_factory, settings, use_llm=True)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_IDEAS_DIR", str(settings.ideas_dir))

    with session_factory() as session:
        idea = sorted(
            repo.ideas_for_vertical(session, VERTICAL),
            key=lambda i: i.score_total,
            reverse=True,
        )[0]

    runner = CliRunner()
    result = runner.invoke(app, ["landing", "-i", idea.id, "--html"])
    assert result.exit_code == 0, result.output
    landing = settings.ideas_dir / idea.id / "landing.md"
    plan = settings.ideas_dir / idea.id / "experiment-plan.md"
    assert landing.exists() and plan.exists()
    text = landing.read_text(encoding="utf-8")
    assert "Variant A" in text and "Variant B" in text
    assert "economics_guardrail" in plan.read_text(encoding="utf-8")
    # --html adds the static export set (5 files total in the idea folder)
    html_files = sorted(
        p.name for p in (settings.ideas_dir / idea.id).iterdir()
    )
    assert html_files == [
        "events.json",
        "experiment-plan.md",
        "landing-a.html",
        "landing-b.html",
        "landing.md",
    ]


def _seed_contaminated_idea(session_factory, settings, against=False) -> tuple[str, str]:
    """Seed a top idea whose hypothesis.job echoes a cited evidence snippet."""
    from evidence_engine.store.models import EvidenceEvent, Query, SourceRun

    seed(session_factory, settings, use_llm=True)
    span = (
        "download options chain data for any stock in csv or excel "
        "and export instantly"
    )
    with session_factory() as session:
        idea = sorted(
            repo.ideas_for_vertical(session, VERTICAL),
            key=lambda i: i.score_total,
            reverse=True,
        )[0]
        hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
        query = Query(text="options chain export", vertical=VERTICAL)
        session.add(query)
        session.flush()
        run = SourceRun(query_id=query.id, source="searxng", status="ok")
        session.add(run)
        session.flush()
        evidence = EvidenceEvent(
            source_run_id=run.id,
            query_id=query.id,
            vertical=VERTICAL,
            source="searxng",
            url="https://competitor.example/export",
            canonical_url="https://competitor.example/export",
            title="Competitor options chain exporter",
            snippet=f"{span} every trading day. Free, no sign-up.",
            content_hash="c" * 64,
        )
        session.add(evidence)
        session.flush()
        if against:
            hypothesis.evidence_against = list(hypothesis.evidence_against or []) + [
                evidence.id
            ]
        else:
            hypothesis.evidence_for = list(hypothesis.evidence_for or []) + [evidence.id]
        hypothesis.job = f"get chains out: {span}"  # contaminated field
        session.commit()
        return idea.id, evidence.id


def test_landing_cli_fails_when_copy_redisplays_evidence(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    """The lane-B incident as an end-to-end gate: a hypothesis field that
    echoes a >=7-word span of a cited evidence row must abort the export."""
    from typer.testing import CliRunner

    from evidence_engine.cli import app

    idea_id, evidence_id = _seed_contaminated_idea(session_factory, settings)

    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_IDEAS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["landing", "-i", idea_id, "--html"])
    assert result.exit_code == 1, result.output
    assert "error:" in result.output
    assert evidence_id in result.output  # lineage: which evidence was echoed
    # nothing contaminated was written
    assert not (tmp_path / idea_id / "landing.md").exists()
    assert not (tmp_path / idea_id / "landing-a.html").exists()


def test_landing_cli_guard_fires_in_markdown_path_too(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    from typer.testing import CliRunner

    from evidence_engine.cli import app

    idea_id, _ = _seed_contaminated_idea(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_IDEAS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["landing", "-i", idea_id])  # no --html
    assert result.exit_code == 1, result.output
    assert "error:" in result.output
    assert not (tmp_path / idea_id / "landing.md").exists()


def test_landing_cli_guards_evidence_against_lineage(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    """Codex finding: an echo carried by an evidence_against row (anti-demand
    lineage) must abort the export too, not just evidence_for."""
    from typer.testing import CliRunner

    from evidence_engine.cli import app

    idea_id, evidence_id = _seed_contaminated_idea(
        session_factory, settings, against=True
    )
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_IDEAS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["landing", "-i", idea_id, "--html"])
    assert result.exit_code == 1, result.output
    assert evidence_id in result.output


def test_landing_uses_persisted_experiment_spec(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    """Codex finding: `experiments start --price` commits economics to
    Experiment.spec; the landing must render THAT spec, not re-draft from
    claims (which showed 'price TBD' next to a running $12 experiment)."""
    from typer.testing import CliRunner

    from evidence_engine.cli import app

    seed(session_factory, settings, use_llm=True)  # synthetic job/pain: guard-clean
    with session_factory() as session:
        idea = sorted(
            repo.ideas_for_vertical(session, VERTICAL),
            key=lambda i: i.score_total,
            reverse=True,
        )[0]
        experiment = repo.latest_experiment_for_idea(session, idea.id)
    assert experiment is not None
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setenv("EE_IDEAS_DIR", str(tmp_path))
    runner = CliRunner()
    started = runner.invoke(
        app, ["experiments", "start", "-e", experiment.id, "--price", "12"]
    )
    assert started.exit_code == 0, started.output
    experiment_id = experiment.id

    result = runner.invoke(app, ["landing", "-i", idea.id, "--html"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / idea.id / "landing.md").read_text(encoding="utf-8")
    assert "$12.0/month" in text  # persisted float renders 12.0
    assert "TBD" not in text
    assert experiment_id  # linked experiment existed
