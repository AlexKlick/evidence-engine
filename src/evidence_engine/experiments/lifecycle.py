"""Experiment lifecycle: draft -> running -> stopped (validated transitions).

Experiment.status is a plain String column; this module is the one place
that validates status writes — `ee experiments start/stop` and
`ee outcomes record --status` both route through validate_transition, so
free-text statuses are impossible going forward. No schema change: the
legacy 'running' row written before this module is a legal value.
"""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from evidence_engine.experiments.experiment import cac_ceiling
from evidence_engine.logging_setup import get_logger
from evidence_engine.store.models import Experiment

logger = get_logger("experiments.lifecycle")

VALID_STATUSES: tuple[str, ...] = ("draft", "running", "stopped")
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"running"}),
    "running": frozenset({"stopped"}),
    "stopped": frozenset(),  # terminal
}


def validate_transition(current: str, target: str) -> None:
    """Raise ValueError unless current -> target is a legal lifecycle step."""
    if target not in VALID_STATUSES:
        raise ValueError(
            f"unknown status {target!r}; valid: {', '.join(VALID_STATUSES)}"
        )
    if current not in VALID_STATUSES:
        raise ValueError(
            f"experiment is in unknown status {current!r}; "
            f"valid: {', '.join(VALID_STATUSES)}"
        )
    if target not in ALLOWED_TRANSITIONS[current]:
        legal = ", ".join(sorted(ALLOWED_TRANSITIONS[current])) or "nothing"
        raise ValueError(
            f"experiment is {current!r}; legal transitions: {legal} "
            f"(got {target!r})"
        )


def _validate_economics(experiment_id: str, spec: dict) -> None:
    """The spend cap must be real money: a finite positive price, and a cap
    that still equals price x margin x payback.

    Presence alone is not enough. A NaN or negative price passes a `is None`
    check and then sets a nonsense cap, and a hand-edited or drifting spec can
    carry a cap unrelated to its own economics.
    """
    guardrail = (spec or {}).get("economics_guardrail") or {}
    price = guardrail.get("price_monthly")
    if price is None:
        raise ValueError(
            f"cannot start experiment {experiment_id!r}: no price on the "
            "economics_guardrail — pass --price or collect price evidence"
        )
    try:
        price_value = float(price)
    except (TypeError, ValueError):
        price_value = float("nan")
    if not math.isfinite(price_value) or price_value <= 0:
        raise ValueError(
            f"cannot start experiment {experiment_id!r}: price_monthly "
            f"{price!r} is not a finite positive amount"
        )
    cap = guardrail.get("cac_ceiling")
    if cap is None:
        return
    margin = float(guardrail.get("gross_margin") or 0.0)
    months = int(guardrail.get("cac_payback_months") or 0)
    expected = cac_ceiling(price_value, margin, months)
    for label, value in (("cac_ceiling", cap), ("maximum_spend", spec.get("maximum_spend"))):
        if value is None:
            continue
        if not math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=0.01):
            raise ValueError(
                f"cannot start experiment {experiment_id!r}: spend cap "
                f"{label}={value} does not match its own economics "
                f"(price {price_value:g} x margin {margin:g} x "
                f"{months} months = {expected}) — the spec drifted"
            )


def start_experiment(session: Session, experiment_id: str) -> Experiment:
    """draft -> running; caller owns the commit.

    Hard gate: an experiment whose economics_guardrail has no usable price
    cannot start — that is the anti-default guarantee (no invented $99). The
    fix is evidence (price signals on claims) or the operator override
    (`ee experiments start --price`).
    """
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise KeyError(f"experiment {experiment_id!r} not found")
    _validate_economics(experiment_id, experiment.spec or {})
    validate_transition(experiment.status, "running")
    experiment.status = "running"
    session.flush()
    logger.info("experiment %s started", experiment_id)
    return experiment


def stop_experiment(
    session: Session, experiment_id: str, decision: str | None = None
) -> Experiment:
    """running -> stopped with an optional decision; caller owns the commit."""
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise KeyError(f"experiment {experiment_id!r} not found")
    validate_transition(experiment.status, "stopped")
    experiment.status = "stopped"
    if decision:
        experiment.decision = decision[:64]
    session.flush()
    logger.info(
        "experiment %s stopped (decision=%s)", experiment_id, experiment.decision
    )
    return experiment


__all__ = [
    "ALLOWED_TRANSITIONS",
    "VALID_STATUSES",
    "start_experiment",
    "stop_experiment",
    "validate_transition",
]
