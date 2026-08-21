"""First-party outcome recording (the loop-closing half of the system)."""

from __future__ import annotations

import csv
import io

OUTCOME_KINDS = [
    "email_signup",
    "interview_completed",
    "demo_request",
    "price_shown_interest",
    "deposit_paid",
    "paid_pilot_started",
    "payment_received",
    "activation",
    "retention_60d",
    "cac_observed",
]


def export_outcomes_csv(rows: list) -> str:
    """rows: Outcome ORM rows -> CSV string (id, experiment, kind, observed)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["outcome_id", "experiment_id", "kind", "value", "observed_at"])
    for row in rows:
        writer.writerow(
            [row.id, row.experiment_id, row.kind, row.value, row.observed_at]
        )
    return buffer.getvalue()
