"""Closed-loop calibration — honest stub.

Feature snapshots are already written at scoring time (store.ScoreSnapshot),
so there is no hindsight leakage. Once >= ~30 experiments have first-party
outcomes, train an interpretable model (logistic regression / GBDT) on
features -> paid_within_30_days / CAC / retained. Until then this returns the
joined dataset and refuses to fit — a hand-fit ranker on tiny data would be
ceremony, not calibration.

NEVER train on platform-restricted source content where the source contract
forbids it (rights.model_training). Aggregate features are ours; raw text is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.store.models import Experiment, Outcome, ScoreSnapshot

MIN_OUTCOMES_FOR_FIT = 30


@dataclass
class CalibrationDataset:
    rows: list[dict]
    outcome_count: int
    ready_to_fit: bool


def assemble_calibration_dataset(session: Session) -> CalibrationDataset:
    """Join score snapshots with their experiments' outcomes."""
    rows: list[dict] = []
    outcome_count = 0
    snapshots = list(session.execute(select(ScoreSnapshot)).scalars())
    for snapshot in snapshots:
        experiments = list(
            session.execute(
                select(Experiment).where(Experiment.idea_id == snapshot.idea_id)
            ).scalars()
        )
        outcomes = []
        for experiment in experiments:
            found = list(
                session.execute(
                    select(Outcome).where(Outcome.experiment_id == experiment.id)
                ).scalars()
            )
            outcomes.extend(found)
        outcome_count += len(outcomes)
        rows.append(
            {
                "snapshot_id": snapshot.id,
                "idea_id": snapshot.idea_id,
                "features": snapshot.features,
                "total": snapshot.total,
                "outcomes": [
                    {"kind": o.kind, "value": o.value} for o in outcomes
                ],
            }
        )
    return CalibrationDataset(
        rows=rows,
        outcome_count=outcome_count,
        ready_to_fit=outcome_count >= MIN_OUTCOMES_FOR_FIT,
    )
