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
    # LLM-mode run: personas fill buyers -> gates pass -> NO review queue,
    # but incumbents exist -> competitor pressure section present.
    result = seed(session_factory, settings, use_llm=True)
    markdown = result.report_path.read_text(encoding="utf-8")
    assert "## Competitor pressure" in markdown
    assert "bigtool" in markdown
    assert "## Review queue" not in markdown
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
