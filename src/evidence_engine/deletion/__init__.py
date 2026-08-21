"""Deletion package: retention TTLs + lineage-aware propagation."""

from evidence_engine.deletion.service import propagate_upstream_deletions, purge_expired

__all__ = ["propagate_upstream_deletions", "purge_expired"]
