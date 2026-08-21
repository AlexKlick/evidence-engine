"""Adapters: gate enforcement in collect(), live SearXNG parsing via respx."""

from __future__ import annotations

import httpx
import respx

from evidence_engine.config import Settings
from evidence_engine.policy import PolicyRegistry
from evidence_engine.sources.base import SourceAdapter
from evidence_engine.sources.searxng import SearXngAdapter
from evidence_engine.sources.stubs import STUB_ADAPTERS

SEARXNG_BODY = {
    "query": "test",
    "results": [
        {
            "url": "https://Example.com/page/?utm_source=x",
            "title": "How do I fix slow inference",
            "content": "manual workaround: spreadsheet &amp; scripts take forever",
            "engines": ["google cse"],
            "score": 1.2,
        },
        {"url": "https://other.com/a", "title": "B", "content": "pricing per month"},
    ],
}


def test_searxng_collect_parses_and_normalizes(
    policy: PolicyRegistry, settings: Settings
) -> None:
    with respx.mock:
        respx.get("http://127.0.0.1:8018/search").mock(
            return_value=httpx.Response(200, json=SEARXNG_BODY)
        )
        batch = SearXngAdapter(policy, settings).collect("test", limit=5)
    assert batch.status == "ok"
    assert len(batch.records) == 2
    record = batch.records[0]
    assert record.canonical == "https://example.com/page"
    assert record.title == "How do I fix slow inference"
    assert "spreadsheet" in record.snippet  # html entity unescaped


def test_searxng_error_becomes_batch_status(
    policy: PolicyRegistry, settings: Settings
) -> None:
    with respx.mock:
        respx.get("http://127.0.0.1:8018/search").mock(
            return_value=httpx.Response(500, text="boom")
        )
        batch = SearXngAdapter(policy, settings).collect("test", limit=5)
    assert batch.status == "error"
    assert batch.reason


def test_disabled_stub_never_collects(policy: PolicyRegistry, settings: Settings) -> None:
    for name, adapter_cls in STUB_ADAPTERS.items():
        batch = adapter_cls(policy).collect("anything")
        assert batch.status == "disabled", name
        assert batch.reason, f"{name} must carry its contractual reason"


def test_unknown_adapter_name_yields_nothing(policy: PolicyRegistry, settings: Settings) -> None:
    from evidence_engine.sources.registry import build_adapters

    adapters = build_adapters(policy, settings, names=["nonexistent"])
    assert adapters == {}


def test_collect_gate_hits_before_network(
    policy: PolicyRegistry, settings: Settings
) -> None:
    class ExplodingAdapter(SourceAdapter):
        name = "reddit"

        def _collect(self, query: str, limit: int):  # pragma: no cover
            raise AssertionError("gate must prevent _collect on disabled sources")

    batch = ExplodingAdapter(policy).collect("q")
    assert batch.status == "disabled"
