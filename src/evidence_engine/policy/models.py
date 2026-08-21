"""Entitlement models + registry loaded from config/source_policies.yaml.

Rights booleans are populated from the actual source contract, never inferred
from public visibility. Default-deny everywhere: unknown source -> deny,
unknown purpose -> deny, ambiguity -> deny.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from evidence_engine.logging_setup import get_logger

logger = get_logger("policy")


class Purpose(StrEnum):
    """What the system wants to do with a source's data."""

    COLLECT = "collect"
    STORE_DERIVED = "store_derived"
    STORE_RAW = "store_raw"
    AGGREGATE = "aggregate"
    LOCAL_INFERENCE = "local_inference"
    EXTERNAL_INFERENCE = "external_inference"
    TRAINING = "training"
    REDISPLAY = "redisplay"


class RightsProfile(BaseModel):
    """Per-source rights snapshot (stored on every evidence record)."""

    collect_enabled: bool = False
    store_derived: bool = False
    store_raw: bool = False
    aggregate: bool = False
    local_inference: bool = False
    external_inference: bool = False
    model_training: bool = False
    redisplay_raw_content: bool = False
    commercial_use: bool = False
    deletion_propagates: bool = False


class SourcePolicy(BaseModel):
    """One registry row: adapter wiring + rights + human-readable reason."""

    name: str
    adapter: str
    enabled: bool = False
    disabled_reason: str | None = None
    rights: RightsProfile = Field(default_factory=RightsProfile)
    retention_days: int | None = None
    notes: str = ""

    @property
    def purpose_rights(self) -> dict[Purpose, bool]:
        r = self.rights
        return {
            Purpose.COLLECT: self.enabled and r.collect_enabled,
            Purpose.STORE_DERIVED: r.store_derived,
            Purpose.STORE_RAW: r.store_raw,
            Purpose.AGGREGATE: r.aggregate,
            Purpose.LOCAL_INFERENCE: r.local_inference,
            Purpose.EXTERNAL_INFERENCE: r.external_inference,
            Purpose.TRAINING: r.model_training,
            Purpose.REDISPLAY: r.redisplay_raw_content,
        }


class PolicyDecision(BaseModel):
    """Result of asking the gate about (source, purpose)."""

    source: str
    purpose: Purpose
    allowed: bool
    reason: str
    policy_version: int = 0


class PolicyRegistry:
    """The first-class entitlement registry (default-deny)."""

    def __init__(
        self,
        sources: dict[str, SourcePolicy] | None = None,
        policy_version: int = 0,
    ) -> None:
        self._sources = sources or {}
        self._version = policy_version

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> PolicyRegistry:
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        version = int(raw.get("policy_version", 0))
        sources: dict[str, SourcePolicy] = {}
        for name, body in (raw.get("sources") or {}).items():
            body = body or {}
            sources[name] = SourcePolicy(
                name=name,
                adapter=body.get("adapter", name),
                enabled=bool(body.get("enabled", False)),
                disabled_reason=body.get("disabled_reason"),
                rights=RightsProfile(**(body.get("rights") or {})),
                retention_days=body.get("retention_days"),
                notes=body.get("notes", ""),
            )
        return cls(sources, version)

    # -- queries -----------------------------------------------------------
    @property
    def policy_version(self) -> int:
        return self._version

    def get(self, source: str) -> SourcePolicy | None:
        return self._sources.get(source)

    def require(self, source: str) -> SourcePolicy:
        policy = self._sources.get(source)
        if policy is None:
            raise KeyError(f"source {source!r} not in entitlement registry (default-deny)")
        return policy

    def all_sources(self) -> dict[str, SourcePolicy]:
        return dict(self._sources)

    def enabled_sources(self) -> list[str]:
        return sorted(name for name, p in self._sources.items() if p.enabled)

    def decision(self, source: str, purpose: Purpose) -> PolicyDecision:
        policy = self._sources.get(source)
        if policy is None:
            return PolicyDecision(
                source=source,
                purpose=purpose,
                allowed=False,
                reason=f"source {source!r} absent from entitlement registry (default-deny)",
                policy_version=self._version,
            )
        if not policy.enabled and purpose is Purpose.COLLECT:
            return PolicyDecision(
                source=source,
                purpose=purpose,
                allowed=False,
                reason=policy.disabled_reason or "source disabled",
                policy_version=self._version,
            )
        allowed = policy.purpose_rights.get(purpose, False)
        reason = (
            "allowed"
            if allowed
            else f"rights.{purpose.value} is false for source {source!r}"
        )
        return PolicyDecision(
            source=source, purpose=purpose, allowed=allowed, reason=reason,
            policy_version=self._version,
        )

    def check(self, source: str, purpose: Purpose) -> None:
        """Raise PermissionError unless (source, purpose) is permitted."""
        decision = self.decision(source, purpose)
        if not decision.allowed:
            raise PermissionError(
                f"policy gate denied {purpose.value} on {source!r}: {decision.reason}"
            )

    def rights_snapshot(self, source: str) -> dict[str, bool]:
        """Row-level rights snapshot for persistence at fetch time."""
        policy = self._sources.get(source)
        if policy is None:
            return RightsProfile().model_dump()
        return policy.rights.model_dump()

    # -- hygiene -----------------------------------------------------------
    def lint(self) -> list[str]:
        """Problems worth surfacing (used by `ee policy lint`)."""
        problems: list[str] = []
        if not self._sources:
            problems.append("registry is empty — nothing can ever be collected")
        for name, policy in self._sources.items():
            if policy.enabled and not policy.rights.collect_enabled:
                problems.append(f"{name}: enabled but rights.collect_enabled is false")
            if policy.rights.store_raw and policy.rights.retention_days is None:
                problems.append(f"{name}: store_raw without retention_days")
            if policy.rights.external_inference and not policy.rights.commercial_use:
                problems.append(
                    f"{name}: external_inference true while commercial_use false "
                    "(sending collected data to third-party models needs review)"
                )
            if policy.enabled and not policy.disabled_reason and policy.disabled_reason == "":
                pass  # enabled sources need no disabled_reason
        if not any(p.enabled for p in self._sources.values()):
            problems.append("no enabled source: pipeline will collect nothing")
        return problems
