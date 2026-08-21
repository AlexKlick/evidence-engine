"""NLP units: normalize, dedupe, intent, clustering."""

from __future__ import annotations

from types import SimpleNamespace

from evidence_engine.nlp.clustering import greedy_cluster, make_label, top_tokens
from evidence_engine.nlp.dedupe import find_duplicates
from evidence_engine.nlp.intent import classify, has_price_signal, is_high_intent
from evidence_engine.nlp.normalize import canonical_url, content_hash

# -- normalize ---------------------------------------------------------------

def test_canonical_url_strips_tracking_and_www() -> None:
    assert canonical_url(
        "https://WWW.Example.com/page/?utm_source=x&id=2&fbclid=abc#frag"
    ) == "https://example.com/page?id=2"


def test_content_hash_stable_across_whitespace() -> None:
    a = content_hash("https://a.com/x", "T", "some  text")
    b = content_hash("https://a.com/x", " T ", "some text")
    assert a == b


# -- dedupe ------------------------------------------------------------------

def row(row_id: str, content_hash_v: str, near: str, vector=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id, content_hash=content_hash_v, near_dup_key=near, embedding=vector
    )


def test_dedupe_exact_and_near() -> None:
    events = [
        row("ev1", "h1", "alpha beta"),
        row("ev2", "h1", "alpha beta"),  # exact dup of ev1
        row("ev3", "h2", "alpha beta"),  # near dup of ev1
        row("ev4", "h3", "other thing"),
    ]
    result = find_duplicates(events)
    assert ("ev2", "ev1") in result.exact_pairs
    assert ("ev3", "ev1") in result.near_pairs
    assert result.duplicate_ids == {"ev2", "ev3"}


def test_dedupe_semantic() -> None:
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.99, 0.02, 0.0]  # ~cos 0.999 -> semantic dup
    events = [
        row("ev1", "h1", "x one", vec_a),
        row("ev2", "h2", "x two", vec_b),
    ]
    result = find_duplicates(events, semantic_threshold=0.97)
    assert ("ev2", "ev1") in result.semantic_pairs


# -- intent ------------------------------------------------------------------

def test_intent_multilabel() -> None:
    labels = classify(
        "How do I stop doing manual spreadsheet imports? Best alternative to Zapier?"
    )
    assert "problem_aware" in labels
    assert "workaround" in labels
    assert "comparison" in labels


def test_intent_price_signal_and_high_intent() -> None:
    labels = classify("pricing per month for a subscription tool")
    assert "transactional" in labels
    assert is_high_intent(labels)
    assert has_price_signal("$99 per month")


def test_anti_demand_detected() -> None:
    assert "anti_demand" in classify("the free tool is good enough, solved it")


# -- clustering ---------------------------------------------------------------

def test_greedy_cluster_groups_similar() -> None:
    groups = greedy_cluster(
        [
            ("a", [1.0, 0.0]),
            ("b", [0.98, 0.05]),
            ("c", [0.0, 1.0]),
        ],
        threshold=0.9,
    )
    sizes = sorted(len(group) for group in groups)
    assert sizes == [1, 2]


def test_make_label_tokens_and_intent() -> None:
    label = make_label(
        ["llama.cpp slow inference fix", "llama.cpp out of memory fix"],
        [["problem_aware"], ["problem_aware"]],
    )
    assert "llama" in label
    assert "[problem_aware]" in label


def test_top_tokens_filters_stopwords() -> None:
    tokens = top_tokens(["the best way to fix slow inference"])
    assert "slow" in tokens or "inference" in tokens
    assert "the" not in tokens
