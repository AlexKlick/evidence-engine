"""Verbatim-redisplay guard: scraped source text must never reach artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evidence_engine.experiments.redisplay_guard import (
    VerbatimRedisplayError,
    assert_no_verbatim_redisplay,
    longest_common_word_ngram,
)

# The actual contaminated headline the live public landing served (lane B
# incident): competitor marketing copy pasted next to our own price.
BAD_HEADLINE = (
    "Done-for-you service that handles: the recurring job — Download options "
    "chain data for any stock in CSV or Excel. Choose columns — price, Greeks, "
    "volume, OI — and export instantly. Free, no sign-up."
)

BAD_SNIPPET = (
    "Options chain exporter — Download options chain data for any stock in CSV "
    "or Excel. Choose columns — price, Greeks, volume, OI — and export "
    "instantly. Free, no sign-up."
)

CLEAN_HEADLINE = (
    "Weekly options-chain exports, formatted your way — no manual copy-paste"
)


def evidence_row(row_id: str = "ev_bad", snippet: str = BAD_SNIPPET) -> SimpleNamespace:
    return SimpleNamespace(id=row_id, title="Options chain exporter", snippet=snippet)


# -- longest_common_word_ngram -------------------------------------------------


def test_bad_headline_shares_a_long_ngram_with_the_snippet() -> None:
    # the live incident: >= 7 shared words (guard threshold)
    assert longest_common_word_ngram(BAD_HEADLINE, BAD_SNIPPET) >= 7


def test_clean_headline_shares_no_long_ngram() -> None:
    assert longest_common_word_ngram(CLEAN_HEADLINE, BAD_SNIPPET) < 7


def test_short_product_name_is_not_a_long_ngram() -> None:
    # legitimate product vocabulary: 3-4 words, must never trip the guard
    assert longest_common_word_ngram(
        "options chain data made simple", BAD_SNIPPET
    ) < 7


def test_ngram_is_casefolded() -> None:
    a = "download OPTIONS chain data for any stock in csv today"
    b = "Download options chain data for any stock in CSV or Excel"
    assert longest_common_word_ngram(a, b) >= 7


def test_empty_inputs_give_zero() -> None:
    assert longest_common_word_ngram("", BAD_SNIPPET) == 0
    assert longest_common_word_ngram(BAD_SNIPPET, "") == 0


# -- assert_no_verbatim_redisplay ---------------------------------------------


def test_live_bad_headline_is_rejected_with_evidence_id_and_span() -> None:
    with pytest.raises(VerbatimRedisplayError) as excinfo:
        assert_no_verbatim_redisplay(
            {"headline_a": BAD_HEADLINE}, [evidence_row("ev_incident")]
        )
    message = str(excinfo.value)
    assert "headline_a" in message  # artifact named
    assert "ev_incident" in message  # evidence id for lineage
    # the matching span itself is reported (spot-check distinctive words)
    assert "download options chain data" in message.casefold()


def test_clean_synthesized_copy_passes() -> None:
    assert_no_verbatim_redisplay(
        {"headline_a": CLEAN_HEADLINE, "html_b": f"<h1>{CLEAN_HEADLINE}</h1>"},
        [evidence_row()],
    )


def test_short_product_name_does_not_false_positive() -> None:
    assert_no_verbatim_redisplay(
        {"headline_a": "Options chain data for every ticker, weekly"},
        [evidence_row()],
    )


def test_six_word_overlap_passes_seven_fails() -> None:
    snippet = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    six = "one two three alpha beta gamma delta epsilon zeta four five"
    seven = "two three alpha beta gamma delta epsilon zeta eta four five"
    assert longest_common_word_ngram(six, snippet) == 6
    assert longest_common_word_ngram(seven, snippet) == 7
    assert_no_verbatim_redisplay({"page": six}, [evidence_row("ev6", snippet)])
    with pytest.raises(VerbatimRedisplayError):
        assert_no_verbatim_redisplay({"page": seven}, [evidence_row("ev7", snippet)])


def test_title_is_also_checked() -> None:
    long_title_span = " ".join(["titleword"] * 8)
    row = SimpleNamespace(id="ev_title", title=long_title_span, snippet="unrelated")
    with pytest.raises(VerbatimRedisplayError):
        assert_no_verbatim_redisplay(
            {"markdown": f"# {long_title_span} product"}, [row]
        )


def test_no_evidence_rows_passes() -> None:
    assert_no_verbatim_redisplay({"markdown": BAD_HEADLINE}, [])
