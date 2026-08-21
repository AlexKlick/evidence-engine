"""Adapter registry: policy keys -> adapter instances."""

from __future__ import annotations

from collections.abc import Callable

from evidence_engine.config import Settings
from evidence_engine.policy import PolicyRegistry
from evidence_engine.sources.base import SourceAdapter
from evidence_engine.sources.searxng import SearXngAdapter
from evidence_engine.sources.stubs import STUB_ADAPTERS

AdapterFactory = Callable[[PolicyRegistry, Settings], SourceAdapter]


def _searxng_factory(policy: PolicyRegistry, settings: Settings) -> SourceAdapter:
    return SearXngAdapter(policy, settings)


def _stub_factory(cls: type[SourceAdapter]):
    def factory(policy: PolicyRegistry, _settings: Settings) -> SourceAdapter:
        return cls(policy)

    return factory


adapter_registry: dict[str, AdapterFactory] = {
    "searxng": _searxng_factory,
    **{name: _stub_factory(cls) for name, cls in STUB_ADAPTERS.items()},
}


def build_adapters(
    policy: PolicyRegistry, settings: Settings, names: list[str] | None = None
) -> dict[str, SourceAdapter]:
    """Build adapters for `names` (default: every registry source)."""
    names = names if names is not None else sorted(policy.all_sources())
    adapters: dict[str, SourceAdapter] = {}
    for name in names:
        source = policy.get(name)
        if source is None:
            continue  # unknown source: the gate would deny anyway; skip
        factory = adapter_registry.get(source.adapter)
        if factory is None:
            continue
        adapters[name] = factory(policy, settings)
    return adapters
