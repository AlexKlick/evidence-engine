"""Closed-loop calibration — honest stub (now fit-capable).

Feature snapshots are already written at scoring time (store.ScoreSnapshot),
so there is no hindsight leakage. Once >= ~30 experiments have first-party
outcomes, train an interpretable model on features -> decision. The fit
itself lives in `ranking.fit`; this module joins snapshots with outcomes
and provides a text-leakage gate before the ranker consumes them.

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


def assemble_calibration_dataset(
    session: Session, check_no_text_leakage: bool = False
) -> CalibrationDataset:
    """Join score snapshots with their experiments' outcomes.

    When `check_no_text_leakage=True`, raise `RankerFitLeakage` on the first
    Outcome whose value carries evidence text. The default stays zero-cost
    so the live `ee calibration` read path is unchanged.
    """
    # Lazy import to keep the read path free of fit-time types.
    from evidence_engine.ranking.fit import (
        RankerFitLeakage,
        outcome_value_carries_text,
    )

    rows: list[dict] = []
    distinct_outcomes: set[str] = set()
    snapshots = list(session.execute(select(ScoreSnapshot)).scalars())
    for snapshot in snapshots:
        experiments = list(
            session.execute(
                select(Experiment).where(Experiment.idea_id == snapshot.idea_id)
            ).scalars()
        )
        outcomes = []
        # v2: any of this idea's experiments carrying a typed decision labels
        # the row. First non-null wins; experiments are an unbounded small
        # set per idea, so we don't need to sort by created_at here.
        latest_decision: str | None = next(
            (e.decision for e in experiments if e.decision), None
        )
        for experiment in experiments:
            found = list(
                session.execute(
                    select(Outcome).where(Outcome.experiment_id == experiment.id)
                ).scalars()
            )
            if check_no_text_leakage:
                for outcome in found:
                    snippet = outcome_value_carries_text(outcome)
                    if snippet is not None:
                        raise RankerFitLeakage(outcome.id, snippet)
            outcomes.extend(found)
        distinct_outcomes.update(outcome.id for outcome in outcomes)
        rows.append(
            {
                "snapshot_id": snapshot.id,
                "idea_id": snapshot.idea_id,
                "features": snapshot.features,
                "total": snapshot.total,
                "decision": latest_decision,
                "outcomes": [
                    {"kind": outcome.kind, "value": outcome.value} for outcome in outcomes
                ],
            }
        )
    return CalibrationDataset(
        rows=rows,
        outcome_count=len(distinct_outcomes),
        ready_to_fit=len(distinct_outcomes) >= MIN_OUTCOMES_FOR_FIT,
    )
