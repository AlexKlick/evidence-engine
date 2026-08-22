"""Experiment spec — the doc's anti-retrospective-metric-selection object."""

from __future__ import annotations

from evidence_engine.nlp.pricing import PriceBand


def cac_ceiling(price_monthly: float, gross_margin: float, payback_months: int) -> float:
    """Gross-profit CAC ceiling: price * margin * payback."""
    return round(price_monthly * gross_margin * payback_months, 2)


def _price_provenance(band: PriceBand) -> str:
    """Where the spec's price came from, e.g. derived:median(n=7,p25=10,p75=99)."""
    return (
        f"derived:median(n={band.n},"
        f"p25={round(band.p25, 2):g},p75={round(band.p75, 2):g})"
    )


def draft_experiment_spec(
    idea,
    hypothesis,
    experiment_defaults: dict,
    price_band: PriceBand | None = None,
) -> dict:
    """Prefill the experiment object from rubric defaults + the idea.

    The price is evidence-derived (median of the claims' price signals) or
    absent — there is deliberately NO default price. A spec without a price
    cannot start (`experiments.lifecycle.start_experiment` hard-gates on it;
    `ee experiments start --price` is the operator override).
    """
    margin = float(experiment_defaults.get("gross_margin", 0.85))
    months = int(experiment_defaults.get("cac_payback_months", 6))
    # half-up (round() is banker's rounding: 10.5 -> 10)
    price = int(price_band.median_monthly + 0.5) if price_band is not None else None
    cap = cac_ceiling(price, margin, months) if price is not None else None
    return {
        "hypothesis": f"{getattr(hypothesis, 'title', '')} :: {getattr(idea, 'form', '')}",
        "primary_metric": "qualified_checkout_starts_per_session",
        "secondary_metrics": ["email_signup_rate", "demo_requests", "deposit_rate"],
        "economics_guardrail": {
            "price_monthly": price,
            "price_provenance": (
                _price_provenance(price_band) if price_band is not None else None
            ),
            "gross_margin": margin,
            "cac_payback_months": months,
            "cac_ceiling": cap,
        },
        "baseline_assumption": None,
        "minimum_detectable_effect": 0.3,
        "allocation": "50/50 randomized",
        "planned_sample_size": None,
        "maximum_spend": cap,
        "start_condition": "gates passed + landing page live",
        "stop_condition": (
            "spend cap reached or 2 sequential weeks without a qualified "
            "checkout start"
        ),
        "result": None,
        "decision": None,
        "status": "draft",
    }
