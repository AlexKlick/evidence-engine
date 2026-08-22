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
PERIOD_WINDOW = 20  # chars either side of the amount that may carry a marker

# The claim field is `price_or_budget_signal`, so budget/loss/savings strings
# arrive here alongside real offers. "we waste $400 per month" is what the
# PROBLEM costs, not what a buyer would pay — reading it as an offer price
# inflated both the band and the spend cap derived from it.
NON_OFFER = re.compile(
    r"\bbudget(?:s|ed|ing)?\b"
    r"|\bcosts?\s+(?:us|me|them|our|my)\b"
    r"|\bwast(?:e|es|ed|ing)\b"
    r"|\bsav(?:e|es|ed|ing|ings)\b"
    r"|\bworth\b"
    r"|\blos(?:e|es|ing|t)\b",
    re.IGNORECASE,
)

# Above this, a "self-serve landing price" is an enterprise quote, a fine, or
# a misparse. Reject it rather than let it set a real spend cap.
MAX_PLAUSIBLE_MONTHLY = 2500.0


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


def _period_markers(text: str) -> list[tuple[int, int, str]]:
    """Every period marker in the text as (start, end, "month" | "year")."""
    markers = [(m.start(), m.end(), "month") for m in MONTH_PERIOD.finditer(text)]
    markers += [(m.start(), m.end(), "year") for m in YEAR_PERIOD.finditer(text)]
    markers.sort()
    return markers


def parse_prices(text: str) -> list[PricePoint]:
    """Parse product prices from one price-signal string.

    Rules (see module docstring for the rationale):
    * "$9.99 per month" / "$250/mo" / "$19 monthly" -> amount as-is
    * "$99 per year" / "$99/yr" / annual variants -> amount / 12
    * "Annual $99, monthly $15" -> a label BEFORE the amount counts too
    * "free" / "$0" -> 0.0 (a free tier, excluded from band statistics)
    * percentages, currency-less numbers, period-less amounts -> rejected
    * budget/loss/savings phrasing -> rejected (not an offer price)
    * anything over MAX_PLAUSIBLE_MONTHLY -> rejected
    * one string can yield several points (monthly OR yearly plan options)

    Each marker is claimed by exactly ONE amount: nearest wins, year wins
    ties, and a claimed marker is off the table for every other amount. That
    is what keeps "$9.99 per month or $99 per year" from reading "month"
    twice, and "Annual $99, monthly $15" from reading "monthly" for $99.
    """
    if not text:
        return []
    cleaned = PERCENT_TOKEN.sub(" ", text)
    if NON_OFFER.search(cleaned):
        return []
    matches = list(MONEY.finditer(cleaned))
    markers = _period_markers(cleaned)
    claimed: set[int] = set()
    points: list[PricePoint] = []
    zero_seen = False
    for index, match in enumerate(matches):
        amount = float(match.group(1).replace(",", ""))
        if amount == 0:
            zero_seen = True
            points.append(PricePoint(0.0, match.group(0).strip()))
            continue
        # a marker may sit before or after the amount, but never across a
        # neighbouring amount — that bounded search is what stops one price's
        # marker bleeding into another's (review finding 2026-08-22)
        prev_end = matches[index - 1].end() if index else 0
        next_start = (
            matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        )
        best: tuple[tuple[int, bool], int, str] | None = None
        for marker_index, (start, end, kind) in enumerate(markers):
            if marker_index in claimed:
                continue
            if end <= match.start():  # label before the amount
                if start < prev_end or match.start() - end > PERIOD_WINDOW:
                    continue
                distance = match.start() - end
            elif start >= match.end():  # marker after the amount
                if end > next_start or start - match.end() > PERIOD_WINDOW:
                    continue
                distance = start - match.end()
            else:
                continue  # overlaps the amount itself
            # nearest wins; year wins ties (annual-first phrasing is common)
            key = (distance, kind != "year")
            if best is None or key < best[0]:
                best = (key, marker_index, kind)
        if best is None:
            continue
        claimed.add(best[1])
        monthly = round(amount / 12, 2) if best[2] == "year" else amount
        if monthly > MAX_PLAUSIBLE_MONTHLY:
            continue
        points.append(PricePoint(monthly, match.group(0).strip()))
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
