"""Pure-stdlib logistic fit on score-snapshot features -> typed decision.

The 14 features below are the keys written by `aggregate_features()`
(`src/evidence_engine/ideas/scoring.py:53-67`). The 10-dim scoring surface
reads a subset of them; the 4 write-only keys
(`unique_query_count`, `unique_source_count`, `anti_demand_count`,
`claim_count`) still ship in the artifact so v5 can audit them.

Stdlib only by design — no numpy / scikit-learn. The default
`uv sync` (`pyproject.toml:7-13`) carries neither; the `cluster`
optional extra is opt-in. v4 keeps the fit inside the default install.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from re import compile

from evidence_engine.store.models import Outcome

# Evidence row ids are emitted as `ev_<hex>` by the seed code. Any such
# substring inside an Outcome.value means the row leaked scraped text; refuse.
_EVIDENCE_ID_RE = compile(r"\bev_[0-9a-f]{6,}\b")


def outcome_value_carries_text(outcome: Outcome) -> str | None:
    """Return a snippet if the outcome's value looks like leaked source text.

    Returns None when safe. Public helper so `ee outcomes import --csv`
    can apply the same gate the fit does, without flushing a row first.
    """
    value = outcome.value
    if not isinstance(value, dict):
        return None
    for key, raw in value.items():
        if not isinstance(raw, str):
            continue
        if _EVIDENCE_ID_RE.search(raw):
            return f"{key}={raw[:80]!r}"
        if len(raw) >= 200:
            return f"{key}=<len {len(raw)}>"
    return None

FEATURE_NAMES: list[str] = [
    "unique_evidence_count",
    "unique_domain_count",
    "unique_query_count",
    "unique_source_count",
    "high_intent_share",
    "workaround_share",
    "switching_share",
    "comparison_present",
    "budget_roi_present",
    "anti_demand_count",
    "explicit_price_signal_count",
    "incumbent_count",
    "max_claim_urgency",
    "claim_count",
]


class InsufficientLabelDiversity(Exception):
    """Raised when the label set collapses to a single value."""


class RankerFitRefused(Exception):
    """Raised when the fit produced NaN/inf coefficients or refused on math."""


class RankerFitLeakage(Exception):
    """Raised when an Outcome.value carries evidence row text — a hard
    rights rule (CLAUDE.md: no raw source content redisplayed in
    published artifacts)."""

    def __init__(self, outcome_id: str, snippet: str) -> None:
        super().__init__(
            f"outcome {outcome_id} carries evidence text in value: {snippet[:120]!r}"
        )
        self.outcome_id = outcome_id
        self.snippet = snippet


@dataclass(frozen=True)
class WeightsFit:
    features: list[str]
    means: list[float]
    stds: list[float]
    weights: list[float]                # length 14
    intercept: float
    loss: float
    regularization: str
    row_count: int
    decision_counts: dict[str, int]
    label_encoding: str
    dropped_constant_features: list[str]


# ---- pure math helpers -------------------------------------------------------


def _sigmoid(z: float) -> float:
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _parse_l2(regularization: str) -> float:
    """Accepts "none" / "ridge(l2=0.01)". Unknown strings fall back to no L2."""
    if regularization == "none":
        return 0.0
    if regularization.startswith("ridge(l2=") and regularization.endswith(")"):
        try:
            return max(float(regularization[len("ridge(l2=") : -1]), 0.0)
        except ValueError:
            return 0.0
    return 0.0


# ---- core fit ----------------------------------------------------------------


def fit_logistic(
    rows: list[dict],
    regularization: str = "ridge(l2=0.01)",
) -> WeightsFit:
    """Fit a logistic regression on stdlib only.

    `rows`: list of {"features": dict, "label": 0|1, "decision": str}.
    Caller drops `decision is None` rows before calling.
    """
    n_rows = len(rows)
    if n_rows < 2:
        raise InsufficientLabelDiversity(
            f"need at least 2 rows; got {n_rows}"
        )

    # Decision counts (raw, pre-collapse) for the artifact's audit trail.
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision_counts[row["decision"]] = decision_counts.get(row["decision"], 0) + 1

    labels = [int(row["label"]) for row in rows]
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise InsufficientLabelDiversity(
            f"label set is {unique_labels}; need both 0 and 1"
        )

    # Means / stds per feature, with constant-column detection.
    means: list[float] = [0.0] * len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        col = [float(row["features"].get(name, 0.0) or 0.0) for row in rows]
        means[FEATURE_NAMES.index(name)] = sum(col) / n_rows

    stds: list[float] = []
    dropped: list[str] = []
    kept_indices: list[int] = []
    for i, name in enumerate(FEATURE_NAMES):
        col = [float(row["features"].get(name, 0.0) or 0.0) for row in rows]
        variance = sum((x - means[i]) ** 2 for x in col) / n_rows
        std = math.sqrt(variance)
        if std < 1e-9:
            stds.append(0.0)
            dropped.append(name)
        else:
            stds.append(std)
            kept_indices.append(i)

    # Build the standardized design matrix (only kept columns).
    n_features = len(kept_indices)
    X: list[list[float]] = []
    for row in rows:
        features_row = [
            (float(row["features"].get(FEATURE_NAMES[i], 0.0) or 0.0) - means[i]) / stds[i]
            for i in kept_indices
        ]
        X.append(features_row)

    # Gradient descent (deterministic — no RNG).
    weights = [0.0] * n_features
    intercept = 0.0
    l2 = _parse_l2(regularization)
    lr = 0.5 / n_rows
    n_iter = 4000
    tolerance = 1e-7

    prev_loss = math.inf
    for _ in range(n_iter):
        # Forward pass.
        preds = [_sigmoid(intercept + sum(w * x for w, x in zip(weights, xi))) for xi in X]
        # Loss.
        eps = 1e-12
        loss = (
            sum(
                -y * math.log(max(p, eps)) - (1.0 - y) * math.log(max(1.0 - p, eps))
                for y, p in zip(labels, preds)
            )
            / n_rows
        ) + l2 * sum(w * w for w in weights)
        if abs(prev_loss - loss) < tolerance:
            break
        prev_loss = loss

        # Gradients.
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for j in range(n_features):
            grad_w[j] = sum((p - y) * X[i][j] for i, (p, y) in enumerate(zip(preds, labels))) / n_rows
            grad_w[j] += 2.0 * l2 * weights[j]
        grad_b = sum(p - y for p, y in zip(preds, labels)) / n_rows

        for j in range(n_features):
            weights[j] -= lr * grad_w[j]
        intercept -= lr * grad_b

    # Validate finiteness.
    if any(math.isnan(w) or math.isinf(w) for w in weights):
        raise RankerFitRefused("gradient descent produced NaN/inf weights")
    if math.isnan(intercept) or math.isinf(intercept):
        raise RankerFitRefused("gradient descent produced NaN/inf intercept")

    # Expand weights back to 14-length with zeros for dropped columns.
    all_weights = [0.0] * len(FEATURE_NAMES)
    for k, i in enumerate(kept_indices):
        all_weights[i] = weights[k]

    return WeightsFit(
        features=list(FEATURE_NAMES),
        means=means,
        stds=stds,
        weights=all_weights,
        intercept=intercept,
        loss=loss,
        regularization=regularization,
        row_count=n_rows,
        decision_counts=decision_counts,
        label_encoding="advance_vs_collapse",
        dropped_constant_features=dropped,
    )


# ---- artifact I/O ------------------------------------------------------------


def _fit_to_payload(fit: WeightsFit) -> dict:
    return {
        "fitted_at": datetime.now(UTC).isoformat(),
        "row_count": fit.row_count,
        "decision_counts": dict(fit.decision_counts),
        "label_encoding": fit.label_encoding,
        "features": list(fit.features),
        "weights": {name: weight for name, weight in zip(fit.features, fit.weights)},
        "intercept": fit.intercept,
        "regularization": fit.regularization,
        "loss": fit.loss,
        "means": list(fit.means),
        "stds": list(fit.stds),
        "dropped_constant_features": list(fit.dropped_constant_features),
        "schema_version": 1,
    }


def write_weights_artifact(fit: WeightsFit, primary: Path, audit_dir: Path) -> Path:
    """Write `fit` to `primary` (atomic) and a timestamped copy under `audit_dir`.

    Returns the audit-copy path. `primary`'s parent and `audit_dir` are
    created via `mkdir(parents=True, exist_ok=True)`.
    """
    payload = _fit_to_payload(fit)

    primary.parent.mkdir(parents=True, exist_ok=True)
    tmp = primary.with_suffix(primary.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, primary)

    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    audit_path = audit_dir / f"ranker-weights-{stamp}.json"
    audit_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return audit_path


__all__ = [
    "FEATURE_NAMES",
    "InsufficientLabelDiversity",
    "RankerFitLeakage",
    "RankerFitRefused",
    "WeightsFit",
    "fit_logistic",
    "outcome_value_carries_text",
    "write_weights_artifact",
]
