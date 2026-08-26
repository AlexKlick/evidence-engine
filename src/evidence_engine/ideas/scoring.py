"""Weighted rubric scoring + hard gates + feature snapshots.

Scores are PRIORS from documented heuristics — never "validation". Hard
gates cap advancement independently of arithmetic. Feature snapshots are
written at scoring time (no hindsight leakage) for the future calibrated
ranker (ranking/calibration.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_engine.nlp.intent import has_price_signal


@dataclass
class ScoreResult:
    total: float
    dimensions: dict[str, float]
    gates: dict[str, bool]
    band: str
    gate_reasons: list[str] = field(default_factory=list)
    # True when the score crossed the paid_validation threshold but the band
    # was capped at interview because no operator review has signed off yet.
    awaiting_review: bool = False


def aggregate_features(evidence_rows: list, claims: list) -> dict:
    """Independence-aware aggregate features (duplicates already excluded)."""
    total = len(evidence_rows)
    label_rows = [list(r.intent_labels or []) for r in evidence_rows]

    def share(family: str) -> float:
        if not total:
            return 0.0
        return sum(1 for labels in label_rows if family in labels) / total

    high_intent_share = sum(
        1
        for row in evidence_rows
        if any(
            label in {"solution_aware", "comparison", "transactional", "switching"}
            for label in (row.intent_labels or [])
        )
    ) / max(total, 1)

    price_signal_count = sum(
        1 for row in evidence_rows if has_price_signal(f"{row.title} {row.snippet}")
    ) + sum(1 for c in claims if c.price_signal)

    incumbents = {i for c in claims for i in (c.incumbents or []) if i}

    return {
        "unique_evidence_count": total,
        "unique_domain_count": len({r.domain for r in evidence_rows if r.domain}),
        "unique_query_count": len({r.query_id for r in evidence_rows}),
        "unique_source_count": len({r.source for r in evidence_rows}),
        "high_intent_share": round(high_intent_share, 4),
        "workaround_share": round(share("workaround"), 4),
        "switching_share": round(share("switching"), 4),
        "comparison_present": any("comparison" in labels for labels in label_rows),
        "budget_roi_present": any("budget_roi" in labels for labels in label_rows),
        "anti_demand_count": sum(1 for labels in label_rows if "anti_demand" in labels),
        "explicit_price_signal_count": price_signal_count,
        "incumbent_count": len(incumbents),
        "max_claim_urgency": max((c.urgency for c in claims), default=0.0),
        "claim_count": len(claims),
    }


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, value))


def score_idea(
    features: dict,
    hypothesis,
    form: str,
    rubric: dict,
    reviewed: bool = False,
    weights_override: dict | None = None,
) -> ScoreResult:
    """0-10 per dimension, weighted to 100; gates cap the band.

    `paid_validation` is an operator-reviewed state, not an arithmetic one:
    pipeline scoring (reviewed=False) caps at `interview` even when the total
    crosses the threshold — derived gates (LLM personas, form templates) are
    priors. Only a human `ee review` pass (reviewed=True) can unlock
    `paid_validation`.
    """
    urgency = features.get("max_claim_urgency", 0.0)
    workaround_share = features.get("workaround_share", 0.0)
    incumbent_count = features.get("incumbent_count", 0)
    price_signals = features.get("explicit_price_signal_count", 0)
    has_buyer = bool(getattr(hypothesis, "buyer", None))
    has_channel = bool(getattr(hypothesis, "channel", None))
    rights_clear = hypothesis.compliance_status == "policy_ok_local_research"

    dimensions = {
        "pain_severity": _clamp(0.6 * urgency * 10 + 0.4 * workaround_share * 10),
        "commercial_intent": _clamp(features.get("high_intent_share", 0.0) * 10),
        "demand_breadth": _clamp(
            0.5 * features.get("unique_evidence_count", 0)
            + features.get("unique_domain_count", 0)
        ),
        "wtp_proxy": _clamp(
            3 + 2 * min(price_signals, 3) + (2 if incumbent_count else 0)
        ),
        "competitive_gap": _clamp(
            (4 if incumbent_count == 0 else max(8 - incumbent_count, 3))
            + (2 if features.get("switching_share", 0) > 0 else 0)
        ),
        "buyer_reachability": _clamp(
            (6 if has_buyer else 2) + (2 if has_channel else 0)
            + (1 if features.get("comparison_present") else 0)
        ),
        "repeatability": _clamp(
            4
            + (4 if workaround_share > 0.25 else 0)
            + (2 if features.get("budget_roi_present") else 0)
        ),
        "compliance_rights": 10.0 if rights_clear else 0.0,
        "mvp_speed": _clamp(
            8 if form in {"productized_service", "template_info_product", "managed_workflow"} else 5
        ),
        "unit_economics": _clamp(5 + (2 if price_signals else 0)),
    }

    if weights_override:
        # v4: learned weights from data/ranker_weights.json (loaded once at
        # boot). Same 10-dim scoring surface; missing keys default to 0.0
        # so an artifact with a sparse feature set never crashes the score.
        weights = {
            name: float(weights_override.get(name, 0.0))
            for name in (rubric.get("dimensions") or {}).keys()
        }
    else:
        weights = {
            name: float(body.get("weight", 0))
            for name, body in (rubric.get("dimensions") or {}).items()
        }
    total = sum(dimensions.get(name, 0.0) * weight for name, weight in weights.items()) / 10.0

    gates = {
        "buyer_identified": has_buyer,
        "channel_identified": has_channel,
        "payment_test_defined": bool(getattr(hypothesis, "smallest_paid_test", None)),
        "rights_clear": rights_clear,
    }
    gate_reasons = [name for name, passed in gates.items() if not passed]

    bands = rubric.get("bands") or {}
    if total >= bands.get("paid_validation", 80):
        band = "paid_validation"
    elif total >= bands.get("interview", 70):
        band = "interview"
    elif total >= bands.get("collect_more", 55):
        band = "collect_more"
    else:
        band = "archive"

    if gate_reasons and band in {"paid_validation", "interview"}:
        band = "collect_more"  # hard gates cap advancement regardless of score

    awaiting_review = False
    if not reviewed and band == "paid_validation":
        band = "interview"  # paid_validation requires operator sign-off
        awaiting_review = True

    return ScoreResult(
        total=round(total, 1),
        dimensions={k: round(v, 1) for k, v in dimensions.items()},
        gates=gates,
        band=band,
        gate_reasons=gate_reasons,
        awaiting_review=awaiting_review,
    )
