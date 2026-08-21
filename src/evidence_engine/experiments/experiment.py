"""Experiment spec — the doc's anti-retrospective-metric-selection object."""

from __future__ import annotations


def cac_ceiling(price_monthly: float, gross_margin: float, payback_months: int) -> float:
    """Gross-profit CAC ceiling: price * margin * payback."""
    return round(price_monthly * gross_margin * payback_months, 2)


def draft_experiment_spec(
    idea,
    hypothesis,
    experiment_defaults: dict,
) -> dict:
    """Prefill the experiment object from rubric defaults + the idea."""
    price = float(experiment_defaults.get("price_monthly", 99))
    margin = float(experiment_defaults.get("gross_margin", 0.85))
    months = int(experiment_defaults.get("cac_payback_months", 6))
    return {
        "hypothesis": f"{getattr(hypothesis, 'title', '')} :: {getattr(idea, 'form', '')}",
        "primary_metric": "qualified_checkout_starts_per_session",
        "secondary_metrics": ["email_signup_rate", "demo_requests", "deposit_rate"],
        "economics_guardrail": {
            "price_monthly": price,
            "gross_margin": margin,
            "cac_payback_months": months,
            "cac_ceiling": cac_ceiling(price, margin, months),
        },
        "baseline_assumption": None,
        "minimum_detectable_effect": 0.3,
        "allocation": "50/50 randomized",
        "planned_sample_size": None,
        "maximum_spend": cac_ceiling(price, margin, months),
        "start_condition": "gates passed + landing page live",
        "stop_condition": (
            "spend cap reached or 2 sequential weeks without a qualified "
            "checkout start"
        ),
        "result": None,
        "decision": None,
        "status": "draft",
    }
