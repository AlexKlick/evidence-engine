"""Store: persistence + lineage-aware deletion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from evidence_engine.policy import PolicyRegistry
from evidence_engine.sources.base import SourceBatch, SourceRecord
from evidence_engine.store import repository as repo
from evidence_engine.store.models import DeletionEvent, EvidenceEvent


def make_batch(records: int, source: str = "searxng") -> SourceBatch:
    return SourceBatch(
        source=source,
        adapter_version="test",
        status="ok",
        records=[
            SourceRecord(
                url=f"https://example.com/{index}?utm_source=junk",
                title=f"How do I fix thing {index}",
                snippet=f"manual spreadsheet workaround {index} takes forever",
                engine="fake",
            )
            for index in range(records)
        ],
    )


def persist(session, policy: PolicyRegistry, count: int) -> list[EvidenceEvent]:
    query = repo.upsert_query(session, "test query", "test-vertical")
    run = repo.record_run(
        session, query, "searxng", "test", policy.policy_version, "ok", None
    )
    return repo.persist_batch(session, run, query, make_batch(count), policy)


def test_persist_batch_snapshots_rights_and_hashes(session, policy) -> None:
    events = persist(session, policy, 3)
    session.commit()
    assert len(events) == 3
    for event in events:
        assert event.rights_profile is not None
        assert event.rights_profile["store_derived"] is True
        assert event.policy_version == policy.policy_version
        assert event.canonical_url.startswith("https://example.com/")
        assert len(event.content_hash) == 64
        assert event.vertical == "test-vertical"


def test_lineage_deletion_removes_children(session, policy) -> None:
    events = persist(session, policy, 1)
    event = events[0]
    repo.add_segments(session, event, ["seg one text here", "seg two text here"])
    repo.add_claim(
        session,
        event,
        {
            "claim_type": "pain",
            "obstacle": "thing",
            "urgency": 0.5,
            "evidence_ids": [event.id],
            "model_version": "test",
            "confidence": 0.5,
        },
    )
    session.commit()

    counts_before = repo.lineage_counts(session, event.id)
    assert counts_before == {"segments": 2, "claims": 1, "memberships": 0}

    repo.delete_evidence_lineage(session, [event.id], "searxng", "unit test")
    session.commit()

    assert repo.lineage_counts(session, event.id) == {
        "segments": 0,
        "claims": 0,
        "memberships": 0,
    }
    deletion = session.query(DeletionEvent).one()
    assert deletion.affected_claims == 1
    assert deletion.affected_segments == 2
    assert session.get(EvidenceEvent, event.id) is None


def test_store_raw_payload_requires_gate(session, policy) -> None:
    """searxng has store_raw=false: the gate refuses STORE_RAW."""
    from evidence_engine.policy import Purpose

    try:
        policy.check("searxng", Purpose.STORE_RAW)
    except PermissionError:
        pass
    else:
        raise AssertionError("searxng must not allow STORE_RAW")


def test_retention_purge_deletes_expired(session, policy) -> None:
    from evidence_engine.deletion.service import purge_expired

    events = persist(session, policy, 2)
    stale = datetime.now(UTC) - timedelta(days=400)
    for event in events:
        event.fetched_at = stale
    session.commit()

    deletions = purge_expired(session, policy)  # searxng retention = 180d
    session.commit()
    assert len(deletions) == 1
    assert deletions[0].reason.startswith("retention TTL expired")
    remaining = (
        session.query(EvidenceEvent)
        .filter(EvidenceEvent.vertical == "test-vertical")
        .count()
    )
    assert remaining == 0
