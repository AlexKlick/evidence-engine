"""Evidence-grounded pricing: parse price signals, derive a band, spec stamping.

The old $99/month default was lifted from an ILLUSTRATIVE example in the
founding doc and silently set real spend caps — these tests pin the
replacement contract: prices come from evidence (or an explicit operator
override at start), never from a default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.nlp.pricing import (
    PriceBand,
    band_from_claims,
    parse_prices,
    price_band,
)


def amounts(text: str) -> list[float]:
    return [point.amount_monthly for point in parse_prices(text)]


# -- parsing ------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # real-world strings lifted from collected price signals
        ("$9.99 per month or $99 per year for Microsoft 365", [9.99, 8.25]),
        ("free forever plan $0", [0.0]),
        ("free", [0.0]),
        ("$250/mo", [250.0]),
        # monthly variants pass the amount through as-is
        ("$19 monthly", [19.0]),
        ("$14.99/month", [14.99]),
        ("$1,250 per month", [1250.0]),
        # yearly variants normalize to monthly
        ("$99 per year", [8.25]),
        ("$99/yr", [8.25]),
        ("$120 annually", [10.0]),
        # false positives seen in real signals are rejected
        ("10% stock price decline", []),
        ("GPU prices", []),
        ("breaking the bank", []),
        ("stock price decline of 10%", []),
        # a currency amount with no period marker is ambiguous -> rejected
        ("$99", []),
        ("a one-time $499 license", []),
    ],
)
def test_parse_prices_table(text: str, expected: list[float]) -> None:
    assert amounts(text) == expected


def test_price_point_carries_raw_match() -> None:
    (point,) = parse_prices("$250/mo")
    assert point.amount_monthly == 250.0
    assert point.raw == "$250"


def test_free_word_counts_when_no_zero_amount() -> None:
    assert amounts("free tier or $9.99 per month") == [9.99, 0.0]


# -- band math ----------------------------------------------------------------

def test_price_band_statistics() -> None:
    band = price_band([9.99, 8.25, 250.0])
    assert band is not None
    assert band.n == 3
    assert band.free_tier_count == 0
    assert band.median_monthly == pytest.approx(9.99)
    assert band.p25 == pytest.approx(9.12)
    assert band.p75 == pytest.approx(129.995)


def test_price_band_excludes_free_tiers_from_stats() -> None:
    band = price_band([0.0, 10.0, 20.0, 30.0])
    assert band is not None
    assert band.n == 3
    assert band.free_tier_count == 1
    assert band.median_monthly == pytest.approx(20.0)
    assert band.p25 == pytest.approx(15.0)
    assert band.p75 == pytest.approx(25.0)


def test_price_band_none_when_no_paid_amounts() -> None:
    assert price_band([]) is None
    assert price_band([0.0, 0.0]) is None


def test_price_band_single_point() -> None:
    band = price_band([15.0])
    assert band == PriceBand(
        median_monthly=15.0, p25=15.0, p75=15.0, n=1, free_tier_count=0
    )


def test_band_from_claims_uses_price_signals() -> None:
    claims = [
        SimpleNamespace(price_signal="$9.99 per month"),
        SimpleNamespace(price_signal=None),  # most claims carry no signal
        SimpleNamespace(price_signal="10% stock price decline"),  # false positive
    ]
    band = band_from_claims(claims)
    assert band is not None
    assert band.n == 1
    assert band.median_monthly == pytest.approx(9.99)


def test_band_from_claims_empty() -> None:
    assert band_from_claims([]) is None


# -- spec derivation ----------------------------------------------------------

RUBRIC_DEFAULTS = {"gross_margin": 0.85, "cac_payback_months": 6}


def draft(price_band_value, defaults=None):
    return draft_experiment_spec(
        SimpleNamespace(form="monitoring agent"),
        SimpleNamespace(title="keep models running"),
        defaults if defaults is not None else RUBRIC_DEFAULTS,
        price_band=price_band_value,
    )


def test_draft_spec_derives_price_and_provenance_from_band() -> None:
    spec = draft(price_band([9.99, 8.25]))
    guardrail = spec["economics_guardrail"]
    assert guardrail["price_monthly"] == 9  # median 9.12 -> int
    assert guardrail["cac_ceiling"] == 45.9  # 9 * 0.85 * 6
    assert spec["maximum_spend"] == 45.9
    provenance = guardrail["price_provenance"]
    assert provenance.startswith("derived:median(n=2,")
    assert "p25=" in provenance and "p75=" in provenance


def test_draft_spec_without_band_has_no_price_and_no_cap() -> None:
    spec = draft(None)
    guardrail = spec["economics_guardrail"]
    assert guardrail["price_monthly"] is None
    assert guardrail["cac_ceiling"] is None
    assert guardrail["price_provenance"] is None
    assert spec["maximum_spend"] is None
    # anchors that don't depend on price survive
    assert guardrail["gross_margin"] == 0.85
    assert guardrail["cac_payback_months"] == 6


def test_draft_spec_ignores_stale_price_default() -> None:
    """Even a stale price_monthly in config can't inject a silent default."""
    spec = draft(None, defaults={**RUBRIC_DEFAULTS, "price_monthly": 99})
    assert spec["economics_guardrail"]["price_monthly"] is None
    assert spec["maximum_spend"] is None


# -- adversarial-review findings 2026-08-22 -----------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # period window must not bleed into the NEXT price's marker (P0):
        # 99 was misparsed as monthly -> 12x cap inflation
        ("$99/yr, $15/mo", [8.25, 15.0]),
        ("Annual: $99/year. Monthly: $15/month.", [8.25, 15.0]),
        # monthly-first ordering must keep working both ways
        ("$15/mo or $99/yr", [15.0, 8.25]),
    ],
)
def test_yearly_first_strings_do_not_bleed_into_monthly(text: str, expected: list[float]) -> None:
    assert amounts(text) == expected


def test_price_band_accepts_generator_input() -> None:
    band = price_band(amount for amount in [10.0, 0.0, 20.0])
    assert band is not None
    assert band.n == 2
    assert band.free_tier_count == 1


def test_derived_price_rounds_half_up() -> None:
    # round() is banker's rounding: 10.5 -> 10; spend math rounds half up
    from evidence_engine.nlp.pricing import PriceBand

    idea = SimpleNamespace(form="productized_service", pitch="p")
    hypothesis = SimpleNamespace(title="t", buyer="b", job="j", pain="p")
    spec = draft_experiment_spec(
        idea,
        hypothesis,
        {"gross_margin": 0.85, "cac_payback_months": 6},
        price_band=PriceBand(median_monthly=10.5, p25=9.0, p75=12.0, n=3),
    )
    assert spec["economics_guardrail"]["price_monthly"] == 11


# -- codex adversarial review (gpt-5.6-sol) 2026-08-22 ------------------------
# executable repros from the review, pinned as regressions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # prefix-labelled plans: the marker before the amount must associate
        # with THAT amount, not bleed forward or get dropped
        ("Annual $99, monthly $15", [8.25, 15.0]),
        ("Yearly: $99; Monthly: $15", [8.25, 15.0]),
        ("monthly plan: $19", [19.0]),
    ],
)
def test_prefix_period_labels_associate_with_their_amount(
    text: str, expected: list[float]
) -> None:
    assert amounts(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # budgets, losses, and savings are NOT offer prices (the claim field is
        # price_or_budget_signal — these used to become prices and spend caps)
        "$50,000 per year budget",
        "manual errors cost us $10,000 per month",
        "save $500 a year",
        "we waste $400 per month on this",
        "worth $250 per month to us",
    ],
)
def test_budget_loss_savings_language_is_not_an_offer_price(text: str) -> None:
    assert amounts(text) == []


def test_implausible_monthly_amounts_are_rejected() -> None:
    # an enterprise quote / fine / anything > $2.5k normalized-monthly is not
    # a plausible self-serve landing price — reject instead of capping spend at it
    assert amounts("$30,000 per month") == []
