"""Scoring: features, weights, bands, hard gates."""

from __future__ import annotations

from types import SimpleNamespace

from evidence_engine.config import load_rubric
from evidence_engine.ideas.forms import generate_forms
from evidence_engine.ideas.hypothesis import build_hypothesis
from evidence_engine.ideas.scoring import aggregate_features, score_idea

DEFAULT_TEXT = "how do i automate spreadsheet manual import"


def make_row(row_id: str, labels: list[str], text: str = DEFAULT_TEXT):
    return SimpleNamespace(
        id=row_id,
        intent_labels=labels,
        domain=f"d{row_id[-1]}.com",
        query_id=f"q{row_id[-1]}",
        source="searxng",
        title=text[:40],
        snippet=text,
        rights_profile={"store_derived": True},
    )


def make_claim(claim_id: str, urgency: float = 0.8, incumbents=None):
    return SimpleNamespace(
        evidence_id=claim_id,
        persona="solo operator",
        job="automate weekly reports",
        obstacle="manual spreadsheet work",
        current_workaround="spreadsheet macros",
        urgency=urgency,
        incumbents=incumbents or ["BigIncumbent"],
        price_signal="$50/mo",
    )


def test_rubric_weights_sum_to_100(settings) -> None:
    rubric = load_rubric(settings)
    total = sum(body["weight"] for body in rubric["dimensions"].values())
    assert total == 100


def test_aggregate_features_counts_independence() -> None:
    rows = [
        make_row("ev1", ["problem_aware", "workaround"]),
        make_row("ev2", ["transactional", "comparison"]),
        make_row("ev3", ["problem_aware"]),
    ]
    claims = [make_claim("ev1"), make_claim("ev2", incumbents=[])]
    features = aggregate_features(rows, claims)
    assert features["unique_evidence_count"] == 3
    assert features["workaround_share"] > 0
    assert features["high_intent_share"] > 0
    assert features["incumbent_count"] == 1
    assert features["max_claim_urgency"] == 0.8


def strong_hypothesis():
    return SimpleNamespace(
        buyer="solo operator",
        job="automate weekly reports",
        pain="manual spreadsheet work",
        current_workaround="spreadsheet macros",
        current_paid_alternative="BigIncumbent",
        incumbent_failures="too expensive",
        evidence_for=["ev1", "ev2"],
        evidence_against=[],
        channel="search",
        suggested_form="productized_service",
        pricing_mechanism="fixed_scope_pilot",
        smallest_paid_test="2-week paid pilot",
        fastest_mvp="concierge",
        compliance_status="policy_ok_local_research",
    )


def test_strong_idea_scores_and_bands(settings) -> None:
    rubric = load_rubric(settings)
    rows = [make_row(f"ev{i}", ["problem_aware", "workaround", "transactional"]) for i in range(6)]
    claims = [make_claim(f"ev{i}") for i in range(6)]
    features = aggregate_features(rows, claims)
    result = score_idea(features, strong_hypothesis(), "productized_service", rubric)
    assert result.total > 0
    assert result.gates == {
        "buyer_identified": True,
        "channel_identified": True,
        "payment_test_defined": True,
        "rights_clear": True,
    }
    assert result.band in {"interview", "paid_validation", "collect_more"}


def test_hard_gates_cap_band(settings) -> None:
    rubric = load_rubric(settings)
    rows = [make_row(f"ev{i}", ["transactional", "comparison"]) for i in range(8)]
    claims = [make_claim(f"ev{i}") for i in range(8)]
    features = aggregate_features(rows, claims)
    weak = strong_hypothesis()
    weak.buyer = None  # no identifiable buyer -> hard gate
    result = score_idea(features, weak, "software", rubric)
    assert "buyer_identified" in result.gate_reasons
    assert result.band != "paid_validation"


def test_rights_not_clear_gates_and_zeroes_compliance(settings) -> None:
    rubric = load_rubric(settings)
    rows = [make_row("ev1", ["problem_aware"])]
    features = aggregate_features(rows, [])
    hypothesis = strong_hypothesis()
    hypothesis.compliance_status = "rights_review_required"
    result = score_idea(features, hypothesis, "software", rubric)
    assert not result.gates["rights_clear"]
    assert result.dimensions["compliance_rights"] == 0.0
    assert result.band != "paid_validation"


def test_build_hypothesis_service_first_when_workarounds() -> None:
    rows = [
        make_row("ev1", ["problem_aware", "workaround"]),
        make_row("ev2", ["workaround", "comparison"]),
    ]
    claims = [make_claim("ev1"), make_claim("ev2")]
    draft = build_hypothesis(
        vertical="v",
        cluster_id=None,
        cluster_label="spreadsheet pain [workaround]",
        evidence_rows=rows,
        claims=claims,
        rights_clear=True,
    )
    assert draft.suggested_form == "productized_service"
    assert draft.channel == "search"
    assert draft.buyer == "solo operator"
    assert draft.evidence_for and not draft.evidence_against


def test_six_forms_generated() -> None:
    forms = generate_forms(strong_hypothesis())
    assert {form["form"] for form in forms} == {
        "software",
        "productized_service",
        "template_info_product",
        "integration",
        "managed_workflow",
        "marketplace",
    }
    assert all(form["smallest_paid_test"] for form in forms)
