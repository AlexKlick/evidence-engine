"""Retention + deletion service.

Lineage walk: evidence -> segments/claims/cluster memberships -> gone.
Hypotheses keep their evidence id lists; dangling ids are the visible signal
that evidence was deleted (audit trail lives in deletion_event).

Sources with rights.deletion_propagates (none live in v1) get upstream
deletion sync through propagate_upstream_deletions once their adapters exist.
Even de-identified retention of deleted content violates e.g. Reddit policy —
when that entitlement lands, this is the machinery that keeps it honest.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.logging_setup import get_logger
from evidence_engine.policy import PolicyRegistry
from evidence_engine.store.models import DeletionEvent, EvidenceEvent
from evidence_engine.store.repository import delete_evidence_lineage

logger = get_logger("deletion")


def purge_expired(
    session: Session, policy: PolicyRegistry, now: datetime | None = None
) -> list[DeletionEvent]:
    """Delete evidence past its source's retention TTL; walk lineage."""
    now = now or datetime.now(UTC)
    events: list[DeletionEvent] = []
    for name, source_policy in policy.all_sources().items():
        retention = source_policy.retention_days
        if retention is None:
            continue
        deadline = now - timedelta(days=retention)
        stale = list(
            session.execute(
                select(EvidenceEvent).where(
                    EvidenceEvent.source == name,
                    EvidenceEvent.fetched_at < deadline,
                )
            ).scalars()
        )
        if not stale:
            continue
        event = delete_evidence_lineage(
            session,
            [row.id for row in stale],
            source=name,
            reason=f"retention TTL expired ({retention}d)",
        )
        events.append(event)
        logger.info(
            "purged %d expired evidence records from %s", len(stale), name
        )
    return events


def propagate_upstream_deletions(
    session: Session, upstream: list[dict]
) -> list[DeletionEvent]:
    """Apply upstream deletion notices: [{source, source_object_id}].

    Maps source objects to our evidence via canonical_url/source ids, then
    walks lineage. Used by adapters whose contracts require deletion sync.
    """
    events: list[DeletionEvent] = []
    by_source: dict[str, list[str]] = {}
    for notice in upstream:
        by_source.setdefault(notice["source"], []).append(notice["source_object_id"])
    for source, object_ids in by_source.items():
        rows = list(
            session.execute(
                select(EvidenceEvent).where(EvidenceEvent.source == source)
            ).scalars()
        )
        targets = [r for r in rows if r.canonical_url in set(object_ids)]
        if targets:
            events.append(
                delete_evidence_lineage(
                    session,
                    [r.id for r in targets],
                    source=source,
                    reason="upstream deletion notice",
                )
            )
    return events
