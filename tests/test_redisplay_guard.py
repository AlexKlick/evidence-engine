"""Verbatim-redisplay guard: scraped source text must never reach artifacts.

The live-incident headline/snippet from the lane B contamination lives in
an operator-local fixture (see `test_live_incident_competitor_fixture`
below). The public tree exercises a 100-char capped synthetic span built
with the same `compress_source_text` cap that the heuristic extractor
already enforces — same guard behavior, no verbatim third-party copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidence_engine.experiments.redisplay_guard import (
    VerbatimRedisplayError,
    assert_no_verbatim_redisplay,
    longest_common_word_ngram,
)
from evidence_engine.nlp.pain_claims import compress_source_text

# 100-char capped synthetic headline built with the same cap the heuristic
# extractor enforces (`compress_source_text`, nlp/pain_claims.py:33): first
# sentence, word-boundary cut, single ellipsis. The synthetic snippet in
# `SYNTHETIC_SNIPPET` below is built from the same sentence so the public
# test exercises the same 7-word n-gram overlap that caught the lane B
# incident — without committing a verbatim competitor quote.
SYNTHETIC_FULL = (
    "done-for-you service that handles the recurring job every week and "
    "ships a clean csv export so you never copy-paste by hand again"
)
SYNTHETIC_HEADLINE = compress_source_text(SYNTHETIC_FULL)
assert len(SYNTHETIC_HEADLINE) <= 100
assert SYNTHETIC_HEADLINE.count("…") <= 1

# Evidence row carrying a snippet that shares 7+ words with SYNTHETIC_HEADLINE
# — so the guard fires on the same overlap threshold the lane B guard was
# designed to enforce, without copying any third-party text.
SYNTHETIC_SNIPPET = (
    "Handles the recurring job every week and ships a clean csv export so "
    "you never copy-paste by hand again, plus daily digests on top."
)
assert longest_common_word_ngram(SYNTHETIC_HEADLINE, SYNTHETIC_SNIPPET) >= 7

CLEAN_HEADLINE = "Weekly options-chain exports, formatted your way — no manual copy-paste"


def evidence_row(
    row_id: str = "ev_synth", snippet: str = SYNTHETIC_SNIPPET
) -> SimpleNamespace:
    return SimpleNamespace(id=row_id, title="Options chain exporter", snippet=snippet)


# -- longest_common_word_ngram -------------------------------------------------


def test_synth_headline_shares_a_long_ngram_with_the_snippet() -> None:
    # the 7-word threshold that caught the live incident is still enforced
    assert longest_common_word_ngram(SYNTHETIC_HEADLINE, SYNTHETIC_SNIPPET) >= 7


def test_clean_headline_shares_no_long_ngram() -> None:
    assert longest_common_word_ngram(CLEAN_HEADLINE, SYNTHETIC_SNIPPET) < 7


def test_short_product_name_is_not_a_long_ngram() -> None:
    # legitimate product vocabulary: 3-4 words, must never trip the guard
    assert longest_common_word_ngram(
        "options chain data made simple", SYNTHETIC_SNIPPET
    ) < 7


def test_ngram_is_casefolded() -> None:
    a = "download OPTIONS chain data for any stock in csv today"
    b = "Download options chain data for any stock in CSV or Excel"
    assert longest_common_word_ngram(a, b) >= 7


def test_empty_inputs_give_zero() -> None:
    assert longest_common_word_ngram("", SYNTHETIC_SNIPPET) == 0
    assert longest_common_word_ngram(SYNTHETIC_SNIPPET, "") == 0


# -- assert_no_verbatim_redisplay ---------------------------------------------


def test_synth_headline_is_rejected_with_evidence_id_and_span() -> None:
    with pytest.raises(VerbatimRedisplayError) as excinfo:
        assert_no_verbatim_redisplay(
            {"headline_a": SYNTHETIC_HEADLINE}, [evidence_row("ev_incident")]
        )
    message = str(excinfo.value)
    assert "headline_a" in message  # artifact named
    assert "ev_incident" in message  # evidence id for lineage
    # the matching span itself is reported (spot-check distinctive words)
    assert "handles the recurring job every week" in message.casefold()


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
    assert_no_verbatim_redisplay({"markdown": SYNTHETIC_HEADLINE}, [])


# -- codex adversarial review (gpt-5.6-sol): fail closed on incomplete lineage


def test_missing_evidence_rows_fail_closed() -> None:
    """A dangling evidence id (deletion leaves them) must abort the export —
    an unguarded artifact could carry text from a row we can no longer check."""
    row = evidence_row("ev1", SYNTHETIC_SNIPPET)
    with pytest.raises(VerbatimRedisplayError, match="missing evidence rows"):
        assert_no_verbatim_redisplay(
            {"page": CLEAN_HEADLINE}, [row], expected_ids=["ev1", "ev_gone"]
        )


def test_expected_ids_satisfied_passes() -> None:
    row = evidence_row("ev1", SYNTHETIC_SNIPPET)
    assert_no_verbatim_redisplay(
        {"page": CLEAN_HEADLINE}, [row], expected_ids=["ev1"]
    )


# -- operator-only live-incident test -----------------------------------------
# The verbatim competitor SEO copy that contaminated the lane B landing is
# preserved as an operator-local fixture so the public tree carries no
# third-party marketing copy. Public CI runs the synthetic-span test above;
# the full-strength test below gates on operator-local file presence.

OPERATOR_FIXTURE = (
    Path.home()
    / ".claude/operators/evidence-engine/test_fixtures/verbatim_chain_competitor.json"
)


@pytest.mark.skipif(
    not OPERATOR_FIXTURE.exists(),
    reason=(
        "operator-local fixture missing; this is the full-strength "
        "verbatim-redisplay test that exercises the live-incident "
        "competitor copy. Public CI runs the synthetic-span test above. "
        "Place the operator-local fixture at "
        "~/.claude/operators/evidence-engine/test_fixtures/"
        "verbatim_chain_competitor.json to re-enable this case."
    ),
)
def test_live_incident_competitor_fixture_rejected() -> None:
    payload = json.loads(OPERATOR_FIXTURE.read_text())
    with pytest.raises(VerbatimRedisplayError) as excinfo:
        assert_no_verbatim_redisplay(
            {"headline_a": payload["headline"]},
            [
                SimpleNamespace(
                    id=payload["evidence_id"],
                    title=payload["row_title"],
                    snippet=payload["snippet"],
                )
            ],
        )
    assert payload["evidence_id"] in str(excinfo.value)
