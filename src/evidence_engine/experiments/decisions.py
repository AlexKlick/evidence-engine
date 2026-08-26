"""Typed experiment decisions (advance|kill|iterate).

Routes through validate_transition (lifecycle.py:30) for the lifecycle side
and synthesizes a learnable Outcome row when a decision lands. This is the
foundation for closing the ranker loop — calibration
(ranking/calibration.py:33) joins snapshots to outcomes, but the existing
'decision' column on Experiment (models.py:235) was a free-form str(64)
that was set but never queried. Typed here, queried in calibration.
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.experiments.lifecycle import validate_transition
from evidence_engine.logging_setup import get_logger
from evidence_engine.store.models import Experiment, Outcome
from evidence_engine.store.repository import record_outcome

Decision = Literal["advance", "kill", "iterate"]
DECISIONS: tuple[Decision, ...] = ("advance", "kill", "iterate")

# Decisions also drive a status transition:
#   advance / kill -> running -> stopped
#   iterate        -> running stays running (more rounds)
DECISION_TARGET_STATUS: dict[str, str] = {
    "advance": "stopped",
    "kill": "stopped",
    "iterate": "running",
}

# Decisions also synthesize an Outcome row (kind + value).
DECISION_OUTCOME_KIND: dict[str, str] = {
    "advance": "advance",
    "kill": "kill",
    "iterate": "iterate",
}

# Mild ordering check: paid milestones should precede an 'advance'. This is a
# warning, not a hard gate — operators override (smoke-tests, concierge
# experiments, partial rollouts).
PAID_OUTCOME_KINDS: frozenset[str] = frozenset({
    "deposit_paid",
    "paid_pilot_started",
    "payment_received",
})

_REASON_CAP = 500

logger = get_logger("experiments.decisions")


def apply_decision(
    session: Session,
    experiment_id: str,
    decision: Decision,
    reason: str | None = None,
    decider: str = "operator",
    override: bool = False,
) -> tuple[Experiment, Outcome | None, list[str]]:
    """Validate, transition, synthesize. Caller owns the commit.

    Returns (experiment, outcome_or_None, warnings). Outcome.value never
    carries evidence row text (CLAUDE.md: 'no raw source content redisplayed'
    in published artifacts) — only operator-authored metadata.
    """
    if decision not in DECISIONS:
        raise ValueError(
            f"unknown decision {decision!r}; valid: {', '.join(DECISIONS)}"
        )
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise KeyError(f"experiment {experiment_id!r} not found")

    target = DECISION_TARGET_STATUS[decision]
    # iterate is a no-op on status (running stays running) — bypass the
    # lifecycle validator only for that case. advance/kill still go through
    # the validated `running -> stopped` transition.
    if experiment.status != target:
        validate_transition(experiment.status, target)

    warnings: list[str] = []
    if decision == "advance":
        seen_kinds = {
            row.kind
            for row in session.execute(
                select(Outcome).where(Outcome.experiment_id == experiment_id)
            ).scalars()
        }
        if not seen_kinds & PAID_OUTCOME_KINDS:
            msg = (
                f"decision=advance on {experiment_id} but no paid outcome "
                f"recorded yet ({sorted(PAID_OUTCOME_KINDS)})"
            )
            if not override:
                warnings.append(msg)
                logger.warning("decision-without-paid-outcome: %s", msg)
            else:
                logger.info("decision-without-paid-outcome override: %s", msg)

    reason_text = (reason or "")[:_REASON_CAP]

    experiment.status = target
    experiment.decision = decision  # fits in String(64)
    experiment.decision_reason = reason_text

    outcome: Outcome | None = None
    # advance / kill always synthesize; iterate only when reason is provided
    # (so 'iterate' on its own is silent — operator can iterate many times).
    if decision != "iterate" or reason_text:
        outcome = record_outcome(
            session,
            experiment,
            DECISION_OUTCOME_KIND[decision],
            {"reason": reason_text, "decider": decider},
        )

    logger.info(
        "experiment %s decision=%s status=%s reason=%r",
        experiment_id,
        decision,
        target,
        reason_text,
    )
    return experiment, outcome, warnings


__all__ = [
    "DECISIONS",
    "DECISION_OUTCOME_KIND",
    "DECISION_TARGET_STATUS",
    "Decision",
    "apply_decision",
]
