"""Problem-hypothesis compiler: cluster evidence + claims -> the doc's template.

A hypothesis is generated only when the evidence supports the fields —
no evidence, no idea.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evidence_engine.nlp.intent import HIGH_INTENT_VALUES


@dataclass
class HypothesisDraft:
    vertical: str
    cluster_id: str | None
    title: str
    buyer: str | None = None
    job: str | None = None
    pain: str | None = None
    current_workaround: str | None = None
    current_paid_alternative: str | None = None
    incumbent_failures: str | None = None
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    channel: str | None = None
    suggested_form: str | None = None
    pricing_mechanism: str | None = None
    smallest_paid_test: str | None = None
    fastest_mvp: str | None = None
    compliance_status: str = "unreviewed"


def _most_common(values: list[str | None]) -> str | None:
    filled = [v for v in values if v]
    if not filled:
        return None
    return Counter(filled).most_common(1)[0][0]


def build_hypothesis(
    vertical: str,
    cluster_id: str | None,
    cluster_label: str,
    evidence_rows: list,
    claims: list,
    rights_clear: bool,
) -> HypothesisDraft:
    """Compile the doc's hypothesis template from cluster members + claims."""
    by_claim_evidence = {claim.evidence_id: claim for claim in claims}

    personas = [c.persona for c in claims]
    jobs = [c.job for c in claims]
    obstacles = [c.obstacle for c in claims]
    workarounds = [c.current_workaround for c in claims]
    incumbents = sorted({i for c in claims for i in (c.incumbents or []) if i})
    paid_incumbent = incumbents[0] if incumbents else None

    evidence_for: list[str] = []
    evidence_against: list[str] = []
    high_intent = False
    comparison = False
    workaround_hit = False
    for row in evidence_rows:
        labels = set(row.intent_labels or [])
        if labels & HIGH_INTENT_VALUES:
            high_intent = True
        if "comparison" in labels:
            comparison = True
        if "workaround" in labels:
            workaround_hit = True
        if "anti_demand" in labels:
            evidence_against.append(row.id)
        elif (labels & {"problem_aware", "feature_request", "workaround", "solution_aware"}) or (
            row.id in by_claim_evidence
        ):
            evidence_for.append(row.id)

    # Service-first when people already spend effort on manual workarounds —
    # fastest path to payment per the founding doc.
    suggested_form = (
        "productized_service" if workaround_hit or workarounds else "software"
    )
    buyer = _most_common(personas)
    job = _most_common(jobs)
    obstacle = _most_common(obstacles)

    return HypothesisDraft(
        vertical=vertical,
        cluster_id=cluster_id,
        title=cluster_label,
        buyer=buyer,
        job=job,
        pain=obstacle,
        current_workaround=_most_common(workarounds),
        current_paid_alternative=paid_incumbent,
        incumbent_failures=(
            f"recurring complaints/switching language around: {', '.join(incumbents)}"
            if incumbents
            else None
        ),
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        channel="search" if (high_intent or comparison) else None,
        suggested_form=suggested_form,
        pricing_mechanism=(
            "fixed_scope_pilot"
            if suggested_form == "productized_service"
            else "hybrid_base_plus_usage"
        ),
        smallest_paid_test=(
            "2-week fixed-scope paid pilot sold on the outcome"
            if suggested_form == "productized_service"
            else "landing page with visible price + refundable deposit"
        ),
        fastest_mvp="concierge fulfillment with internal tools; automate only repeated steps",
        compliance_status="policy_ok_local_research" if rights_clear else "rights_review_required",
    )
