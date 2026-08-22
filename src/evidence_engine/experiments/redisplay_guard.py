"""Verbatim-redisplay guard: scraped source text must never reach artifacts.

Enforces the ADR-0005 claim that published landing copy is *synthesized* from
store_derived evidence. Any artifact string sharing a word n-gram of length
>= ``VERBATIM_NGRAM`` with a cited evidence row's title or snippet is a
publication block, not a warning — shorter spans (product names like
"options chain data") are legitimate vocabulary and pass.
"""

from __future__ import annotations

from collections.abc import Iterable

from evidence_engine.logging_setup import get_logger

logger = get_logger("experiments.redisplay_guard")

# 7 shared words = a sentence fragment, not a product name.
VERBATIM_NGRAM = 7


class VerbatimRedisplayError(Exception):
    """An artifact redisplayed a long verbatim span of collected evidence."""


def _words(text: str) -> list[str]:
    return text.casefold().split()


def _grams(words: list[str], size: int) -> Iterable[str]:
    return (" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def longest_common_word_ngram(a: str, b: str) -> int:
    """Length of the longest word n-gram shared by a and b (casefolded).

    Binary search on the (monotone) predicate "a shared k-gram exists" —
    cheap for the intended scale (~10 evidence rows x a few artifacts).
    """
    words_a, words_b = _words(a), _words(b)
    if not words_a or not words_b:
        return 0

    def has_common(size: int) -> bool:
        grams_b = set(_grams(words_b, size))
        return any(gram in grams_b for gram in _grams(words_a, size))

    lo, hi = 0, min(len(words_a), len(words_b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has_common(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _first_common_span(a: str, b: str, size: int) -> str:
    words_a, words_b = _words(a), _words(b)
    grams_b = set(_grams(words_b, size))
    for gram in _grams(words_a, size):
        if gram in grams_b:
            return gram
    return ""  # pragma: no cover - size came from longest_common_word_ngram


def assert_no_verbatim_redisplay(texts: dict[str, str], evidence_rows: list) -> None:
    """Raise VerbatimRedisplayError if any artifact text echoes evidence.

    The error message carries the artifact name, the evidence id (lineage),
    and the matching span so the operator can fix the source of the copy.
    """
    for artifact_name, text in texts.items():
        for row in evidence_rows:
            row_id = getattr(row, "id", "?")
            for field in ("title", "snippet"):
                source = getattr(row, field, None) or ""
                shared = longest_common_word_ngram(text, source)
                if shared >= VERBATIM_NGRAM:
                    span = _first_common_span(text, source, shared)
                    logger.error(
                        "verbatim redisplay blocked: %s echoes %s.%s (%d-word span)",
                        artifact_name,
                        row_id,
                        field,
                        shared,
                    )
                    raise VerbatimRedisplayError(
                        f"{artifact_name} would redisplay a {shared}-word verbatim "
                        f"span of evidence {row_id} ({field}): {span!r} — "
                        "paraphrase the copy before publishing"
                    )
