"""Live SearXNG adapter (self-hosted, loopback :8018, keyless JSON API).

Entitlement: personal research use (see config/source_policies.yaml).
Politeness: the pipeline sleeps `delay_seconds` between queries; this adapter
makes exactly one GET per collect() call.
"""

from __future__ import annotations

import httpx

from evidence_engine.config import Settings
from evidence_engine.logging_setup import get_logger
from evidence_engine.nlp.normalize import norm_text
from evidence_engine.policy import PolicyRegistry
from evidence_engine.sources.base import SourceAdapter, SourceBatch, SourceRecord

logger = get_logger("sources.searxng")


class SearXngAdapter(SourceAdapter):
    name = "searxng"
    version = "0.1.0"

    def __init__(self, policy: PolicyRegistry, settings: Settings) -> None:
        super().__init__(policy)
        self._settings = settings

    def _collect(self, query: str, limit: int) -> SourceBatch:
        cfg = self._settings
        response = httpx.get(
            f"{cfg.searxng_base_url.rstrip('/')}/search",
            params={"q": query, "format": "json", "language": cfg.searxng_language},
            timeout=cfg.searxng_timeout,
            headers={"User-Agent": "evidence-engine/0.1 (personal research)"},
        )
        response.raise_for_status()
        payload = response.json()
        records: list[SourceRecord] = []
        for item in (payload.get("results") or [])[:limit]:
            url = item.get("url") or ""
            if not url:
                continue
            records.append(
                SourceRecord(
                    url=url,
                    title=norm_text(item.get("title") or ""),
                    snippet=norm_text(item.get("content") or ""),
                    engine=", ".join(item.get("engines") or [])
                    if isinstance(item.get("engines"), list)
                    else str(item.get("engines") or ""),
                    score=item.get("score"),
                    published_at=item.get("publishedDate") or None,
                    language=cfg.searxng_language,
                )
            )
        return SourceBatch(
            source=self.name, adapter_version=self.version, status="ok", records=records
        )
