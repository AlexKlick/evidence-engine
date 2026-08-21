"""Experiment lifecycle: draft -> running -> stopped (validated transitions).

Experiment.status is a plain String column; this module is the one place
that validates status writes — `ee experiments start/stop` and
`ee outcomes record --status` both route through validate_transition, so
free-text statuses are impossible going forward. No schema change: the
legacy 'running' row written before this module is a legal value.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

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


def start_experiment(session: Session, experiment_id: str) -> Experiment:
    """draft -> running; caller owns the commit."""
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise KeyError(f"experiment {experiment_id!r} not found")
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
