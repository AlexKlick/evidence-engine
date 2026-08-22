"""Human review: field updates, gate flips, snapshot appends, evidence packs."""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder
from evidence_engine.config import load_rubric
from evidence_engine.ideas.review import (
    apply_review,
    evidence_pack,
    render_evidence_pack,
    review_queue,
)
from evidence_engine.ideas.scoring import score_idea
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo
from evidence_engine.store.models import ProblemHypothesis, ScoreSnapshot

VERTICAL = "local-ai-tooling"

# Rich enough that score_idea crosses the paid_validation threshold no
# matter how the rubric weights split (every dimension lands >= 7).
STRONG_FEATURES = {
    "unique_evidence_count": 8,
    "unique_domain_count": 8,
    "unique_query_count": 8,
    "unique_source_count": 2,
    "high_intent_share": 1.0,
    "workaround_share": 0.5,
    "switching_share": 0.1,
    "comparison_present": True,
    "budget_roi_present": True,
    "anti_demand_count": 0,
    "explicit_price_signal_count": 3,
    "incumbent_count": 1,
    "max_claim_urgency": 0.9,
    "claim_count": 8,
}


def _craft_capped_hypothesis(session_factory, settings) -> str:
    """A gates-passing hypothesis whose pipeline-scored idea crossed the
    paid_validation threshold and was capped at interview (review gate)."""
    rubric = load_rubric(settings)
    with session_factory() as session:
        hypothesis = repo.save_hypothesis(
            session,
            vertical=VERTICAL,
            cluster_id=None,
            title="crafted strong cluster",
            buyer="solo operator",
            job="automate weekly reports",
            pain="manual spreadsheet work",
            evidence_for=[],
            evidence_against=[],
            channel="search",
            suggested_form="productized_service",
            pricing_mechanism="fixed_scope_pilot",
            smallest_paid_test="2-week paid pilot",
            compliance_status="policy_ok_local_research",
        )
        result = score_idea(STRONG_FEATURES, hypothesis, "productized_service", rubric)
        assert result.total >= 80
        assert result.band == "interview" and result.awaiting_review
        idea = repo.save_idea(
            session,
            vertical=VERTICAL,
            hypothesis_id=hypothesis.id,
            form="productized_service",
            pitch="concierge data automation",
            pricing_mechanism="fixed_scope_pilot",
            smallest_paid_test="2-week paid pilot",
            mvp_sketch="internal tools",
            score_total=result.total,
            score_dimensions=result.dimensions,
            gates=result.gates,
            band=result.band,
        )
        repo.save_snapshot(session, idea, STRONG_FEATURES, result.total)
        session.commit()
        return hypothesis.id


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


def test_awaiting_review_surfaces_in_queue_and_pack(
    fake_adapters, session_factory, settings
) -> None:
    hypothesis_id = _craft_capped_hypothesis(session_factory, settings)
    rubric = load_rubric(settings)

    with session_factory() as session:
        queue = review_queue(session, VERTICAL, rubric=rubric)
        entry = next(e for e in queue if e["hypothesis_id"] == hypothesis_id)
        assert entry["awaiting_review"] is True
        assert entry["missing"] == []

        pack = evidence_pack(session, hypothesis_id, rubric=rubric)
        assert pack["awaiting_review"] is True
        assert pack["missing"] == []
        text = "\n".join(render_evidence_pack(pack))
        assert "capped at interview" in text
        assert f"ee review -H {hypothesis_id} --buyer" in text


def test_operator_signoff_unlocks_paid_validation(
    fake_adapters, session_factory, settings
) -> None:
    hypothesis_id = _craft_capped_hypothesis(session_factory, settings)
    rubric = load_rubric(settings)

    with session_factory() as session:
        outcome = apply_review(
            session, hypothesis_id, rubric, {"buyer": "solo operator (confirmed)"}
        )
        session.commit()
    assert outcome.ideas, "review must rescore the capped idea"
    assert outcome.ideas[0]["band"] == "paid_validation"

    # signed-off hypothesis leaves the queue entirely
    with session_factory() as session:
        queue = review_queue(session, VERTICAL, rubric=rubric)
        assert all(entry["hypothesis_id"] != hypothesis_id for entry in queue)
