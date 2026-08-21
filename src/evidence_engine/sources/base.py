"""Adapter contract.

`collect()` is a template method: it consults the policy gate first
(default-deny), so no adapter can fetch anything without an enabling
entitlement row. Errors become batch status, never exceptions past the gate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from evidence_engine.logging_setup import get_logger
from evidence_engine.nlp.normalize import canonical_url
from evidence_engine.policy import PolicyRegistry, Purpose

logger = get_logger("sources")

BatchStatus = Literal["ok", "disabled", "error", "empty"]


@dataclass
class SourceRecord:
    """One normalized result item (derived data — raw payloads are separate)."""

    url: str
    title: str
    snippet: str
    engine: str = ""
    score: float | None = None
    published_at: str | None = None
    language: str = "en"

    @property
    def canonical(self) -> str:
        return canonical_url(self.url)


@dataclass
class SourceBatch:
    """Result of one adapter run for one query."""

    source: str
    adapter_version: str
    status: BatchStatus
    records: list[SourceRecord] = field(default_factory=list)
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class SourceAdapter(ABC):
    """Base class: name matches a source_policies.yaml key."""

    name: str = "abstract"
    version: str = "0.1.0"

    def __init__(self, policy: PolicyRegistry) -> None:
        self.policy = policy

    def collect(self, query: str, limit: int = 10) -> SourceBatch:
        """Gate -> collect. Disabled adapters return status='disabled'."""
        decision = self.policy.decision(self.name, Purpose.COLLECT)
        if not decision.allowed:
            logger.info("gate denied collect on %s: %s", self.name, decision.reason)
            return SourceBatch(
                source=self.name,
                adapter_version=self.version,
                status="disabled",
                reason=decision.reason,
            )
        try:
            batch = self._collect(query, limit)
        except Exception as exc:  # noqa: BLE001 — adapter errors become status
            logger.warning("adapter %s failed: %s", self.name, exc)
            return SourceBatch(
                source=self.name,
                adapter_version=self.version,
                status="error",
                reason=str(exc),
            )
        if not batch.records:
            batch.status = "empty"
        return batch

    @abstractmethod
    def _collect(self, query: str, limit: int) -> SourceBatch: ...


class GatedStubAdapter(SourceAdapter):
    """Adapter whose entitlement does not exist yet.

    The gate rejects COLLECT before `_collect` is ever reached; implementing
    this method is the future work item, next to the entitlement row.
    """

    name = "gated_stub"

    def _collect(self, query: str, limit: int) -> SourceBatch:  # pragma: no cover
        raise NotImplementedError(
            f"{self.name}: live collection not implemented — see the "
            "disabled_reason in config/source_policies.yaml"
        )
