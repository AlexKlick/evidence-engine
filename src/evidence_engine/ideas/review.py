"""Human review: the operator gate from the founding doc.

Reviews update hypothesis fields (buyer, channel, smallest paid test, ...),
then re-score the hypothesis' ideas against their STORED feature snapshots —
features never change retroactively (no hindsight leakage); only the
gates/bands respond to review. Every rescore appends a new snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.ideas.scoring import aggregate_features, score_idea
from evidence_engine.logging_setup import get_logger
from evidence_engine.store import repository as repo
from evidence_engine.store.models import ProblemHypothesis

logger = get_logger("ideas.review")

REVIEWABLE_FIELDS = (
    "title",
    "buyer",
    "job",
    "pain",
    "current_workaround",
    "current_paid_alternative",
    "incumbent_failures",
    "channel",
    "smallest_paid_test",
    "fastest_mvp",
    "compliance_status",
)

PACK_CLAIMS = 10
PACK_EVIDENCE = 8
PACK_SNIPPET_CHARS = 140

# gate name (scoring.py) -> review flag that satisfies it
GATE_TO_FLAG = {
    "buyer_identified": '--buyer "..."',
    "channel_identified": '--channel "..."',
    "payment_test_defined": '--smallest-paid-test "..."',
    "rights_clear": "--compliance policy_ok_local_research",
}

VALID_COMPLIANCE = {"policy_ok_local_research", "rights_review_required", "unreviewed"}


@dataclass
class ReviewOutcome:
    hypothesis_id: str
    applied: dict[str, str]
    ideas: list[dict]  # [{id, form, total, band, failed_gates}]


def review_queue(
    session: Session, vertical: str | None = None, rubric: dict | None = None
) -> list[dict]:
    """The operator's worklist, ordered by best idea score.

    Two kinds of entry:
    - gates failing (`missing` non-empty) — fill --buyer/--channel/...;
    - `awaiting_review` (only when `rubric` is passed) — gates all pass and
      the best idea crossed the paid_validation threshold, but pipeline
      scoring capped it at interview; an explicit `ee review` sign-off
      (re-affirm at least one field) is what unlocks the band.
    """
    paid_band = (rubric or {}).get("bands", {}).get("paid_validation", 80)
    stmt = select(ProblemHypothesis)
    if vertical:
        stmt = stmt.where(ProblemHypothesis.vertical == vertical)
    queue: list[dict] = []
    for hypothesis in session.execute(stmt).scalars():
        ideas = repo.ideas_for_hypothesis(session, hypothesis.id)
        if not ideas:
            continue
        missing: set[str] = set()
        for idea in ideas:
            missing.update(name for name, ok in (idea.gates or {}).items() if not ok)
        best = max(idea.score_total for idea in ideas)
        if missing:
            queue.append(
                {
                    "hypothesis_id": hypothesis.id,
                    "vertical": hypothesis.vertical,
                    "title": hypothesis.title,
                    "missing": sorted(missing),
                    "awaiting_review": False,
                    "buyer_set": bool(hypothesis.buyer),
                    "channel_set": bool(hypothesis.channel),
                    "ideas": len(ideas),
                    "best_score": best,
                }
            )
        elif rubric is not None and best >= paid_band and any(
            idea.band == "interview" for idea in ideas
        ):
            queue.append(
                {
                    "hypothesis_id": hypothesis.id,
                    "vertical": hypothesis.vertical,
                    "title": hypothesis.title,
                    "missing": [],
                    "awaiting_review": True,
                    "buyer_set": bool(hypothesis.buyer),
                    "channel_set": bool(hypothesis.channel),
                    "ideas": len(ideas),
                    "best_score": best,
                }
            )
    queue.sort(key=lambda entry: entry["best_score"], reverse=True)
    return queue


def evidence_pack(
    session: Session, hypothesis_id: str, rubric: dict | None = None
) -> dict:
    """Read-only operator pack for one hypothesis: claims, evidence excerpts,
    aggregate features, idea scores, and the gates still holding them back.

    Everything needed to fill --buyer/--channel/--smallest-paid-test in one
    view. Features are recomputed here read-only — scores only ever move via
    apply_review, against STORED snapshots. When `rubric` is passed, the pack
    also flags ideas capped at interview by the review gate (total crossed
    the paid_validation threshold without an operator sign-off).
    """
    hypothesis = repo.get_hypothesis(session, hypothesis_id)
    if hypothesis is None:
        raise KeyError(f"hypothesis {hypothesis_id!r} not found")

    evidence_rows = repo.evidence_by_ids(
        session, list(hypothesis.evidence_for or [])
    )
    claims = repo.claims_for_evidence(session, [row.id for row in evidence_rows])
    features = aggregate_features(evidence_rows, claims)

    ideas: list[dict] = []
    missing: set[str] = set()
    for idea in sorted(
        repo.ideas_for_hypothesis(session, hypothesis_id),
        key=lambda idea: idea.score_total,
        reverse=True,
    ):
        failed = [name for name, ok in (idea.gates or {}).items() if not ok]
        missing.update(failed)
        ideas.append(
            {
                "id": idea.id,
                "form": idea.form,
                "total": idea.score_total,
                "band": idea.band,
                "failed_gates": failed,
            }
        )

    paid_band = (rubric or {}).get("bands", {}).get("paid_validation", 80)
    awaiting_review = bool(ideas) and not missing and (
        max(idea["total"] for idea in ideas) >= paid_band
        and any(idea["band"] == "interview" for idea in ideas)
    )

    return {
        "hypothesis": {
            "id": hypothesis.id,
            "vertical": hypothesis.vertical,
            "title": hypothesis.title,
            "buyer": hypothesis.buyer,
            "job": hypothesis.job,
            "pain": hypothesis.pain,
            "channel": hypothesis.channel,
            "smallest_paid_test": hypothesis.smallest_paid_test,
            "compliance_status": hypothesis.compliance_status,
            "current_paid_alternative": hypothesis.current_paid_alternative,
            "incumbent_failures": hypothesis.incumbent_failures,
            "evidence_for": list(hypothesis.evidence_for or []),
            "evidence_against": list(hypothesis.evidence_against or []),
        },
        "claims": [
            {
                "persona": claim.persona,
                "job": claim.job,
                "obstacle": claim.obstacle,
                "urgency": claim.urgency,
                "commercial_intent": claim.commercial_intent,
                "price_signal": claim.price_signal,
                "incumbents": claim.incumbents or [],
                "evidence_id": claim.evidence_id,
            }
            for claim in sorted(
                claims, key=lambda claim: claim.urgency, reverse=True
            )[:PACK_CLAIMS]
        ],
        "evidence": [
            {
                "id": row.id,
                "title": row.title,
                "snippet": row.snippet,
                "source": row.source,
                "url": row.url,
                "domain": row.domain,
            }
            for row in evidence_rows[:PACK_EVIDENCE]
        ],
        "features": features,
        "ideas": ideas,
        "missing": sorted(missing),
        "awaiting_review": awaiting_review,
    }


def render_evidence_pack(pack: dict) -> list[str]:
    """Markdown-ish lines for `ee review --show` (read-only operator view)."""
    hyp = pack["hypothesis"]
    lines = [
        f"Evidence pack — {hyp['title']}  ({hyp['vertical']}, {hyp['id']})",
        "",
        f"buyer:          {hyp['buyer'] or '—'}",
        f"job/pain:       {hyp['job'] or '—'} / {hyp['pain'] or '—'}",
        f"channel:        {hyp['channel'] or '—'}",
        f"paid test:      {hyp['smallest_paid_test'] or '—'}",
        f"compliance:     {hyp['compliance_status'] or 'unreviewed'}",
        f"paid incumbent: {hyp['current_paid_alternative'] or '—'}",
        "",
        "## Ideas",
    ]
    for idea in pack["ideas"]:
        failed = ",".join(idea["failed_gates"]) or "pass"
        lines.append(
            f"  {idea['id']}  {idea['total']:5.1f}  {idea['band']:<15} "
            f"{idea['form']:<24} gates:{failed}"
        )
    if pack["missing"]:
        lines += ["", f"Missing gates: {', '.join(pack['missing'])}"]
    if pack["claims"]:
        lines += ["", "## Pain claims (top by urgency)"]
        for claim in pack["claims"]:
            who = claim["persona"] or "unknown persona"
            what = claim["obstacle"] or claim["job"] or ""
            incumbents = ", ".join(claim["incumbents"]) or "—"
            price = f"  price: {claim['price_signal']}" if claim["price_signal"] else ""
            lines.append(
                f"  [{claim['urgency']:.1f}] {who} — {what} "
                f"(incumbents: {incumbents}; intent: {claim['commercial_intent']}"
                f"{price})"
            )
    if pack["evidence"]:
        lines += ["", "## Evidence excerpts"]
        for row in pack["evidence"]:
            snippet = " ".join((row["snippet"] or "").split())[:PACK_SNIPPET_CHARS]
            lines.append(f"  {row['domain'] or row['source']}: {row['title']}")
            lines.append(f"    {row['url']}")
            lines.append(f"    {snippet}")
    features = pack["features"]
    lines += [
        "",
        "## Features (recomputed read-only; stored snapshots never change)",
        f"  evidence={features.get('unique_evidence_count', 0)} "
        f"domains={features.get('unique_domain_count', 0)} "
        f"high_intent={features.get('high_intent_share', 0):.0%} "
        f"workaround={features.get('workaround_share', 0):.0%} "
        f"switching={features.get('switching_share', 0):.0%} "
        f"price_signals={features.get('explicit_price_signal_count', 0)} "
        f"incumbents={features.get('incumbent_count', 0)}",
    ]
    if pack["missing"]:
        flags = [
            GATE_TO_FLAG[gate]
            for gate in pack["missing"]
            if gate in GATE_TO_FLAG
        ]
        lines += ["", "next:", f"  ee review -H {hyp['id']} " + " ".join(flags)]
    elif pack.get("awaiting_review"):
        lines += [
            "",
            "Score crossed the paid_validation threshold but no operator has "
            "signed off —",
            "band capped at interview (derived gates are priors, not review).",
            "next: confirm the fields yourself, e.g.",
            f"  ee review -H {hyp['id']} --buyer \"{hyp['buyer'] or '...'}\"",
        ]
    return lines


def apply_review(
    session: Session,
    hypothesis_id: str,
    rubric: dict,
    updates: dict,
    weights_override: dict | None = None,
) -> ReviewOutcome:
    """Apply operator updates to a hypothesis and rescore its ideas.

    `weights_override` (v4): when present, `score_idea` uses the learned
    ranker coefficients from `data/ranker_weights.json` instead of the
    YAML rubric. The CLI's `ee review` passes `load_learned_weights(settings)`
    at boot; tests pass `None` to keep the YAML baseline.
    """
    hypothesis = repo.get_hypothesis(session, hypothesis_id)
    if hypothesis is None:
        raise KeyError(f"hypothesis {hypothesis_id!r} not found")

    applied: dict[str, str] = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key not in REVIEWABLE_FIELDS:
            raise ValueError(
                f"field {key!r} is not reviewable; valid: {', '.join(REVIEWABLE_FIELDS)}"
            )
        if key == "compliance_status" and value not in VALID_COMPLIANCE:
            raise ValueError(
                f"compliance_status must be one of {sorted(VALID_COMPLIANCE)}"
            )
        setattr(hypothesis, key, value)
        applied[key] = value

    if not applied:
        return ReviewOutcome(hypothesis_id=hypothesis_id, applied={}, ideas=[])

    rescored: list[dict] = []
    for idea in repo.ideas_for_hypothesis(session, hypothesis_id):
        snapshot = repo.latest_snapshot(session, idea.id)
        features = dict(snapshot.features) if snapshot and snapshot.features else {}
        if snapshot is None:
            logger.warning(
                "idea %s has no feature snapshot; scoring against empty features", idea.id
            )
        result = score_idea(
            features,
            hypothesis,
            idea.form,
            rubric,
            reviewed=True,
            weights_override=weights_override,
        )
        idea.score_total = result.total
        idea.score_dimensions = result.dimensions
        idea.gates = result.gates
        idea.band = result.band
        repo.save_snapshot(session, idea, features, result.total)
        rescored.append(
            {
                "id": idea.id,
                "form": idea.form,
                "total": result.total,
                "band": result.band,
                "failed_gates": result.gate_reasons,
            }
        )
    session.flush()
    return ReviewOutcome(hypothesis_id=hypothesis_id, applied=applied, ideas=rescored)
