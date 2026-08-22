"""Evidence-grounded pricing: parse price-signal strings into a monthly band.

Replaces the old hardcoded $99/month default (lifted from an ILLUSTRATIVE
example in docs/design/evidence-to-revenue.md) that silently set real spend
caps. Prices now come from collected evidence or an explicit operator
override (`ee experiments start --price`) — never from a default.

Parsing is deliberately conservative: a nonzero amount needs a currency
symbol AND an explicit period (month/year); percentages, bare numbers, and
qualitative "prices" language are rejected.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

# "10%", "9.9 %" — stripped before scanning so percentages never look like prices
PERCENT_TOKEN = re.compile(r"\d+(?:\.\d+)?\s?%")
MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
MONTH_PERIOD = re.compile(r"\bmonth(?:ly)?\b|/mo\b", re.IGNORECASE)
YEAR_PERIOD = re.compile(
    r"\byear(?:ly)?\b|/yr\b|/year\b|\bannually\b|\bannual\b|\bper annum\b",
    re.IGNORECASE,
)
FREE_WORD = re.compile(r"\bfree\b", re.IGNORECASE)
PERIOD_WINDOW = 20  # chars after the amount that may carry the period marker


@dataclass(frozen=True)
class PricePoint:
    """One parsed product price, normalized to a monthly amount."""

    amount_monthly: float
    raw: str


@dataclass(frozen=True)
class PriceBand:
    """Distribution of paid monthly prices; free tiers counted, not averaged."""

    median_monthly: float
    p25: float
    p75: float
    n: int
    free_tier_count: int = 0


def parse_prices(text: str) -> list[PricePoint]:
    """Parse product prices from one price-signal string.

    Rules (see module docstring for the rationale):
    * "$9.99 per month" / "$250/mo" / "$19 monthly" -> amount as-is
    * "$99 per year" / "$99/yr" / annual variants -> amount / 12
    * "free" / "$0" -> 0.0 (a free tier, excluded from band statistics)
    * percentages, currency-less numbers, period-less amounts -> rejected
    * one string can yield several points (monthly OR yearly plan options)
    """
    if not text:
        return []
    cleaned = PERCENT_TOKEN.sub(" ", text)
    points: list[PricePoint] = []
    zero_seen = False
    for match in MONEY.finditer(cleaned):
        amount = float(match.group(1).replace(",", ""))
        if amount == 0:
            zero_seen = True
            points.append(PricePoint(0.0, match.group(0).strip()))
            continue
        # the window ends at the NEXT currency amount so one price's period
        # marker can never bleed into another's ("$99/yr, $15/mo" — review
        # finding 2026-08-22: the unbounded window read "/mo" for $99)
        next_amount = cleaned.find("$", match.end())
        window_end = match.end() + PERIOD_WINDOW
        if next_amount != -1:
            window_end = min(window_end, next_amount)
        window = cleaned[match.end() : window_end]
        month_at = MONTH_PERIOD.search(window)
        year_at = YEAR_PERIOD.search(window)
        # nearest marker wins; year wins ties (annual-first phrasing is common)
        if year_at and (month_at is None or year_at.start() <= month_at.start()):
            points.append(PricePoint(round(amount / 12, 2), match.group(0).strip()))
        elif month_at:
            points.append(PricePoint(amount, match.group(0).strip()))
    if not zero_seen and FREE_WORD.search(cleaned):
        points.append(PricePoint(0.0, "free"))
    return points


def price_band(amounts: Iterable[float]) -> PriceBand | None:
    """Band over the paid (nonzero) monthly amounts; None when there are none.

    Free tiers (0.0) are excluded from the statistics but counted in
    `free_tier_count` so "everyone ships a free plan" stays visible.
    """
    values_in = list(amounts)  # generators would exhaust between the two passes
    values = sorted(amount for amount in values_in if amount > 0)
    if not values:
        return None
    if len(values) == 1:
        q25 = median = q75 = values[0]
    else:
        q25, median, q75 = statistics.quantiles(values, n=4, method="inclusive")
    return PriceBand(
        median_monthly=median,
        p25=q25,
        p75=q75,
        n=len(values),
        free_tier_count=sum(1 for amount in values_in if amount == 0),
    )


def band_from_claims(claims: Iterable[object]) -> PriceBand | None:
    """Derive the band from claim rows' `price_signal` strings (duck-typed)."""
    signals = [
        signal
        for claim in claims
        if (signal := getattr(claim, "price_signal", None))
    ]
    amounts = [
        point.amount_monthly for signal in signals for point in parse_prices(signal)
    ]
    return price_band(amounts)
