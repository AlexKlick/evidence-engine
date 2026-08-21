"""Source adapters. Only adapters with an enabling entitlement row collect."""

from evidence_engine.sources.base import (
    GatedStubAdapter,
    SourceAdapter,
    SourceBatch,
    SourceRecord,
)
from evidence_engine.sources.registry import adapter_registry, build_adapters
from evidence_engine.sources.searxng import SearXngAdapter

__all__ = [
    "GatedStubAdapter",
    "SourceAdapter",
    "SourceBatch",
    "SourceRecord",
    "SearXngAdapter",
    "adapter_registry",
    "build_adapters",
]
