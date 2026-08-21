"""Multi-label intent taxonomy — deterministic weak labels first.

Generic topical similarity misses buying-stage; these families are the
economically meaningful axes from the founding doc. Precision on
high-intent/WTP classes matters more than recall: falsely labeling general
discussion as transactional corrupts the opportunity ranker.
"""

from __future__ import annotations

import re
from enum import StrEnum


class IntentFamily(StrEnum):
    PROBLEM_AWARE = "problem_aware"
    WORKAROUND = "workaround"
    SOLUTION_AWARE = "solution_aware"
    COMPARISON = "comparison"
    TRANSACTIONAL = "transactional"
    SWITCHING = "switching"
    BUDGET_ROI = "budget_roi"
    FEATURE_REQUEST = "feature_request"
    ANTI_DEMAND = "anti_demand"


INTENT_PATTERNS: dict[IntentFamily, re.Pattern[str]] = {
    IntentFamily.PROBLEM_AWARE: re.compile(
        r"\bhow do i\b|\bhow to\b|\bwhy does\b|\bwhy is\b|\btakes? forever\b|"
        r"\bmanual(ly)?\b|\bstruggl\w*\b|\bpainful\b|\bannoying\b|\bfrustrat\w*\b",
        re.IGNORECASE,
    ),
    IntentFamily.WORKAROUND: re.compile(
        r"\bspreadsheet\b|\bexcel\b|\bzapier\b|\bmacro\b|\bscript i wrote\b|"
        r"\bcopy.?paste\b|\bworkaround\b|\bduct.?taped?\b|\bby hand\b|\bmanual(ly)?\b",
        re.IGNORECASE,
    ),
    IntentFamily.SOLUTION_AWARE: re.compile(
        r"\btool for\b|\bsoftware that\b|\bapp that\b|\bautomat\w+\b|\bcli for\b|"
        r"\bplugin for\b|\bservice that\b",
        re.IGNORECASE,
    ),
    IntentFamily.COMPARISON: re.compile(
        r"\bvs\.?\b|\bversus\b|\balternative to\b|\bbest \w+ for\b|\bcompared? to\b|"
        r"\binstead of\b|\breplacement for\b",
        re.IGNORECASE,
    ),
    IntentFamily.TRANSACTIONAL: re.compile(
        r"\bpric\w+\b|\bcost\b|\bhow much\b|\bbuy\b|\btrial\b|\bsubscription\b|"
        r"\bper month\b|\$/mo\b|\blicen[cs]e\b|\bquote\b",
        re.IGNORECASE,
    ),
    IntentFamily.SWITCHING: re.compile(
        r"\bmigrat\w+\b|\bmove (off|from|away)\b|\bcancel\w*\b|\bswitch (from|to)\b|"
        r"\btoo expensive\b|\bdropped\b|\breplac\w+\b",
        re.IGNORECASE,
    ),
    IntentFamily.BUDGET_ROI: re.compile(
        r"\$\s?\d[\d,.]*\b|\bhours (a|per) week\b|\bheadcount\b|\broi\b|\bcost us\b|"
        r"\bsaves? \d+\b|\bworth paying\b",
        re.IGNORECASE,
    ),
    IntentFamily.FEATURE_REQUEST: re.compile(
        r"\bwish (it|there)\b|\bdoes anyone know\b|\bany way to\b|"
        r"\blooking for (a )?(way|tool)\b|\bneed (a )?tool\b|\bwhy (isn'?t|doesn'?t)\b",
        re.IGNORECASE,
    ),
    IntentFamily.ANTI_DEMAND: re.compile(
        r"\bsolved\b|\bnot worth\b|\bfree tool\b|\boverkill\b|\bjust use\b|"
        r"\bgood enough\b|\bworks fine\b",
        re.IGNORECASE,
    ),
}

HIGH_INTENT = {
    IntentFamily.SOLUTION_AWARE,
    IntentFamily.COMPARISON,
    IntentFamily.TRANSACTIONAL,
    IntentFamily.SWITCHING,
}

HIGH_INTENT_VALUES = {family.value for family in HIGH_INTENT}

PRICE_SIGNAL = re.compile(
    r"\$\s?\d[\d,.]*|\bper month\b|\b/mo\b|\bpricing\b|\bsubscription\b|\b_paid\b",
    re.IGNORECASE,
)


def classify(text: str) -> list[str]:
    """Weak multi-label classification of one text."""
    return [
        family.value
        for family, pattern in INTENT_PATTERNS.items()
        if pattern.search(text or "")
    ]


def is_high_intent(labels: list[str]) -> bool:
    return any(label in HIGH_INTENT_VALUES for label in labels)


def has_price_signal(text: str) -> bool:
    return bool(PRICE_SIGNAL.search(text or ""))
