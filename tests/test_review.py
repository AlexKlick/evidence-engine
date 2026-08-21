"""Human review: field updates, gate flips, snapshot appends, evidence packs."""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder
from evidence_engine.config import load_rubric
from evidence_engine.ideas.review import apply_review, evidence_pack, render_evidence_pack
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo
from evidence_engine.store.models import ProblemHypothesis, ScoreSnapshot

VERTICAL = "local-ai-tooling"


def seed(session_factory, settings) -> str:
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        hypothesis = (
            session.query(ProblemHypothesis).filter_by(vertical=VERTICAL).first()
        )
        assert hypothesis is not None
        return hypothesis.id


def test_review_flips_gates_and_appends_snapshots(
    fake_adapters, session_factory, settings
) -> None:
    hypothesis_id = seed(session_factory, settings)
    rubric = load_rubric(settings)

    with session_factory() as session:
        ideas = repo.ideas_for_hypothesis(session, hypothesis_id)
        assert ideas, "pipeline should have produced ideas for the hypothesis"
        totals_before = {idea.id: idea.score_total for idea in ideas}
        snapshots_before = session.query(ScoreSnapshot).count()

    with session_factory() as session:
        outcome = apply_review(
            session,
            hypothesis_id,
            rubric,
            {
                "buyer": "hobbyist local-LLM operator",
                "channel": "search",
                "smallest_paid_test": "landing page + $50 refundable deposit",
            },
        )
        session.commit()

    assert outcome.applied["buyer"] == "hobbyist local-LLM operator"
    assert len(outcome.ideas) == len(ideas)
    for entry in outcome.ideas:
        assert entry["failed_gates"] == [], "all four gates must pass after review"
        assert entry["total"] >= totals_before[entry["id"]]

    with session_factory() as session:
        snapshots_after = session.query(ScoreSnapshot).count()
        assert snapshots_after == snapshots_before + len(ideas)
        hypothesis = repo.get_hypothesis(session, hypothesis_id)
        assert hypothesis is not None
        assert hypothesis.buyer == "hobbyist local-LLM operator"
        # features unchanged (no hindsight leakage): latest snapshots share features
        for idea in repo.ideas_for_hypothesis(session, hypothesis_id):
            snap = repo.latest_snapshot(session, idea.id)
            assert snap is not None and snap.features


def test_review_rejects_bad_input(fake_adapters, session_factory, settings) -> None:
    hypothesis_id = seed(session_factory, settings)
    rubric = load_rubric(settings)

    with session_factory() as session:
        with pytest.raises(KeyError):
            apply_review(session, "hyp_missing", rubric, {"buyer": "x"})
        with pytest.raises(ValueError, match="not reviewable"):
            apply_review(session, hypothesis_id, rubric, {"score_total": "99"})
        with pytest.raises(ValueError, match="compliance_status"):
            apply_review(
                session, hypothesis_id, rubric, {"compliance_status": "probably_fine"}
            )


def test_noop_review_touches_nothing(fake_adapters, session_factory, settings) -> None:
    hypothesis_id = seed(session_factory, settings)
    with session_factory() as session:
        snapshots_before = session.query(ScoreSnapshot).count()
        outcome = apply_review(session, hypothesis_id, load_rubric(settings), {})
        assert outcome.applied == {}
        assert session.query(ScoreSnapshot).count() == snapshots_before


def test_evidence_pack_carries_claims_excerpts_features_gates(
    fake_adapters, session_factory, settings
) -> None:
    hypothesis_id = seed(session_factory, settings)
    with session_factory() as session:
        pack = evidence_pack(session, hypothesis_id)

    assert pack["hypothesis"]["id"] == hypothesis_id
    assert pack["hypothesis"]["vertical"] == VERTICAL
    assert pack["claims"], "heuristic run still extracts claims to cite"
    assert all("evidence_id" in claim for claim in pack["claims"])
    assert pack["evidence"], "pack must show excerpts to read"
    first = pack["evidence"][0]
    assert first["url"] and first["source"] and first["title"] is not None
    assert pack["features"]["unique_evidence_count"] == len(
        pack["hypothesis"]["evidence_for"]
    )
    assert pack["ideas"] and all("band" in idea for idea in pack["ideas"])
    # un-reviewed hypothesis: buyer is the gate the heuristic builder leaves
    # open (channel/compliance derive from evidence); every listed gate is fillable
    assert "buyer_identified" in pack["missing"]
    assert set(pack["missing"]) <= {
        "buyer_identified",
        "channel_identified",
        "payment_test_defined",
        "rights_clear",
    }


def test_evidence_pack_unknown_hypothesis_raises(session_factory) -> None:
    with session_factory() as session, pytest.raises(KeyError):
        evidence_pack(session, "hyp_missing")


def test_render_evidence_pack_lines(fake_adapters, session_factory, settings) -> None:
    hypothesis_id = seed(session_factory, settings)
    with session_factory() as session:
        pack = evidence_pack(session, hypothesis_id)
    lines = render_evidence_pack(pack)
    text = "\n".join(lines)

    assert lines[0].startswith("Evidence pack")
    assert "Missing gates: buyer_identified" in text
    assert "## Pain claims" in text
    assert "## Evidence excerpts" in text
    assert "## Features" in text
    assert f"ee review -H {hypothesis_id}" in text
    assert '--buyer "..."' in text
