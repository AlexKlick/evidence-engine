"""Query expansion: sanitize/dedupe, sidecar round-trip, collect merge."""

from __future__ import annotations

import yaml

from conftest import FakeEmbedder
from evidence_engine.config import load_verticals
from evidence_engine.nlp.query_expansion import (
    effective_queries,
    expand_queries,
    expanded_path,
    load_expanded,
    save_expanded,
)
from evidence_engine.pipeline import Pipeline


class FakeExpandLLM:
    model_name = "fake-expand"

    def health(self, timeout: float = 5.0) -> bool:
        return True

    def chat_json(self, system: str, user: str, **kwargs):
        return {
            "queries": [
                {"text": "how do i stop manual llm babysitting", "intent": "problem_aware"},
                # same query with case/whitespace noise -> must dedupe away:
                {"text": "  How do I stop manual LLM babysitting  ", "intent": "problem_aware"},
                {"text": "", "intent": "problem_aware"},  # dropped
                {"text": "x" * 200, "intent": "problem_aware"},  # too long
                {"text": "best alternative to ollama pro", "intent": "comparison"},
                {"text": "llama.cpp slow inference fix", "intent": "problem_aware"},  # dup of seed
            ]
        }


def test_expand_sanitizes_and_dedupes(settings) -> None:
    seeds = ["llama.cpp slow inference fix"]
    result = expand_queries(FakeExpandLLM(), "local-ai-tooling", seeds, [], limit=10)
    texts = [q.text for q in result]
    assert texts == [
        "how do i stop manual llm babysitting",
        "best alternative to ollama pro",
    ]
    assert all(q.origin == "fake-expand" for q in result)


def test_sidecar_roundtrip_and_effective_merge(settings) -> None:
    from evidence_engine.nlp.query_expansion import ExpandedQuery

    vertical = load_verticals(settings)["local-ai-tooling"]
    entries = [
        ExpandedQuery(text="local llm monitoring dashboard", intent="solution_aware"),
        ExpandedQuery(text="LLAMA.CPP slow inference fix", intent="problem_aware"),  # dup of seed
    ]
    path = save_expanded(settings, "local-ai-tooling", entries, "test")
    assert path.exists()

    loaded = load_expanded(settings, "local-ai-tooling")
    assert len(loaded) == 2  # sidecar keeps it; dedupe happens at merge time

    merged = effective_queries(vertical, settings, "local-ai-tooling")
    assert merged[0] == vertical["queries"][0]  # seeds first
    assert "local llm monitoring dashboard" in merged
    assert merged.count("llama.cpp slow inference fix") == 1  # normalized dup removed


def test_collect_uses_expanded_queries(
    fake_adapters, session_factory, settings
) -> None:
    from evidence_engine.nlp.query_expansion import ExpandedQuery

    save_expanded(
        settings,
        "local-ai-tooling",
        [ExpandedQuery(text="extra expanded query one", intent="comparison")],
        "test",
    )
    result = Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
    ).run("local-ai-tooling", limit=2, use_llm=False)
    # 5 seeds + 1 expanded, 3 records per query via the fake adapter
    assert result.collect.records == 18
    assert result.collect.runs == 6


def test_yaml_sidecar_shape(settings) -> None:
    from evidence_engine.nlp.query_expansion import ExpandedQuery

    save_expanded(
        settings,
        "solo-dev-saas",
        [ExpandedQuery(text="automate client reporting", intent="solution_aware")],
        "test-model",
    )
    data = yaml.safe_load(expanded_path(settings, "solo-dev-saas").read_text())
    assert data["origin"].startswith("llm-expansion (test-model)")
    assert data["queries"][0]["text"] == "automate client reporting"
