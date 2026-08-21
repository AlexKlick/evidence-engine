"""Repository: evidence CRUD + the lineage queries deletion depends on.

Every function takes an explicit Session — the pipeline owns the transaction.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.nlp.normalize import content_hash, domain_of, near_dup_key
from evidence_engine.policy import PolicyRegistry
from evidence_engine.sources.base import SourceBatch
from evidence_engine.store.models import (
    Cluster,
    ClusterMember,
    DeletionEvent,
    EvidenceEvent,
    Experiment,
    Outcome,
    PainClaim,
    ProblemHypothesis,
    ProductIdea,
    Query,
    RawPayload,
    ScoreSnapshot,
    SourceRun,
    TextSegment,
    new_id,
)


def _now() -> datetime:
    return datetime.now(UTC)


# NOTE ON FLUSH ORDER: this schema deliberately uses bare ForeignKey columns
# and no relationship() declarations (the repository passes explicit ids). In
# SQLAlchemy 2.0 the unit-of-work then does NOT foreign-key-order INSERTs
# across mappers (verified empirically: `evidence_event` flushes before
# `query`). Every parent-creating function therefore flushes before
# returning, so any child autoflush always sees its parents inserted.


# -- collection ---------------------------------------------------------------

def upsert_query(session: Session, text: str, vertical: str) -> Query:
    stmt = select(Query).where(Query.text == text, Query.vertical == vertical)
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return existing
    query = Query(id=new_id("q"), text=text, vertical=vertical)
    session.add(query)
    session.flush()
    return query


def record_run(
    session: Session,
    query: Query,
    source: str,
    adapter_version: str,
    policy_version: int,
    status: str,
    reason: str | None,
) -> SourceRun:
    run = SourceRun(
        id=new_id("run"),
        query_id=query.id,
        source=source,
        adapter_version=adapter_version,
        policy_version=policy_version,
        status=status,
        reason=reason,
        record_count=0,
        started_at=_now(),
    )
    session.add(run)
    session.flush()
    return run


def persist_batch(
    session: Session,
    run: SourceRun,
    query: Query,
    batch: SourceBatch,
    policy: PolicyRegistry,
) -> list[EvidenceEvent]:
    """Store one adapter batch (gate re-checked for STORE_DERIVED)."""
    rights = policy.rights_snapshot(batch.source)
    events: list[EvidenceEvent] = []
    if batch.status == "ok":
        from evidence_engine.policy import Purpose  # local import: avoid cycle in tests

        policy.check(batch.source, Purpose.STORE_DERIVED)
    for record in batch.records if batch.status == "ok" else []:
        event = EvidenceEvent(
            id=new_id("ev"),
            source_run_id=run.id,
            query_id=query.id,
            vertical=query.vertical,
            source=batch.source,
            url=record.url,
            canonical_url=record.canonical,
            domain=domain_of(record.url),
            title=record.title,
            snippet=record.snippet,
            engine=record.engine,
            score=record.score,
            published_at=record.published_at,
            fetched_at=_now(),
            language=record.language,
            content_hash=content_hash(record.url, record.title, record.snippet),
            near_dup_key=near_dup_key(f"{record.title} {record.snippet}"),
            rights_profile=rights,
            policy_version=policy.policy_version,
        )
        session.add(event)
        events.append(event)
    run.record_count = len(events)
    run.status = batch.status
    run.reason = batch.reason
    run.finished_at = _now()
    session.flush()  # evidence rows are parents of segments/claims/memberships
    return events


# -- evidence reads -----------------------------------------------------------

def evidence_for_vertical(
    session: Session, vertical: str, originals_only: bool = True
) -> list[EvidenceEvent]:
    stmt = select(EvidenceEvent).where(EvidenceEvent.vertical == vertical)
    if originals_only:
        stmt = stmt.where(EvidenceEvent.is_duplicate_of.is_(None))
    return list(session.execute(stmt).scalars())


def claims_for_evidence(session: Session, evidence_ids: list[str]) -> list[PainClaim]:
    if not evidence_ids:
        return []
    stmt = select(PainClaim).where(PainClaim.evidence_id.in_(evidence_ids))
    return list(session.execute(stmt).scalars())


# -- dedupe / enrich ----------------------------------------------------------

def mark_duplicate(
    session: Session, duplicate: EvidenceEvent, original: EvidenceEvent, stage: str
) -> None:
    duplicate.is_duplicate_of = original.id
    duplicate.dedupe_stage = stage


def set_embedding(
    session: Session, evidence: EvidenceEvent, vector: list[float], model: str
) -> None:
    evidence.embedding = vector
    evidence.embedding_model = model


def set_intent(session: Session, evidence: EvidenceEvent, labels: list[str]) -> None:
    evidence.intent_labels = labels


def add_segments(session: Session, evidence: EvidenceEvent, texts: list[str]) -> None:
    for ordinal, text in enumerate(texts):
        session.add(
            TextSegment(id=new_id("sg"), evidence_id=evidence.id, ordinal=ordinal, text=text)
        )


def add_claim(session: Session, evidence: EvidenceEvent, claim: dict) -> PainClaim:
    row = PainClaim(
        id=new_id("clm"),
        evidence_id=evidence.id,
        claim_type=str(claim.get("claim_type", "pain")),
        persona=claim.get("persona"),
        job=claim.get("job"),
        obstacle=claim.get("obstacle"),
        consequence=claim.get("consequence"),
        current_workaround=claim.get("current_workaround"),
        urgency=float(claim.get("urgency") or 0.0),
        commercial_intent=str(claim.get("commercial_intent", "unaware")),
        price_signal=claim.get("price_or_budget_signal"),
        incumbents=claim.get("incumbents") or [],
        evidence_ids=claim.get("evidence_ids") or [evidence.id],
        model_version=str(claim.get("model_version", "")),
        confidence=float(claim.get("confidence") or 0.0),
    )
    session.add(row)
    return row


# -- clustering / ideas -------------------------------------------------------

def create_cluster(
    session: Session,
    vertical: str,
    label: str,
    member_ids: list[str],
    policy_version: int,
    embedding_model: str | None,
    algorithm: str = "greedy_cosine",
) -> Cluster:
    cluster = Cluster(
        id=new_id("cl"),
        vertical=vertical,
        algorithm=algorithm,
        label=label,
        member_count=len(member_ids),
        policy_version=policy_version,
        embedding_model=embedding_model,
    )
    session.add(cluster)
    session.flush()  # members reference cluster.id
    for evidence_id in member_ids:
        session.add(
            ClusterMember(cluster_id=cluster.id, evidence_id=evidence_id)
        )
    return cluster


def save_hypothesis(session: Session, **fields: object) -> ProblemHypothesis:
    hypothesis = ProblemHypothesis(id=new_id("hyp"), **fields)  # type: ignore[arg-type]
    session.add(hypothesis)
    session.flush()  # ideas reference hypothesis.id
    return hypothesis


def save_idea(session: Session, **fields: object) -> ProductIdea:
    idea = ProductIdea(id=new_id("idea"), **fields)  # type: ignore[arg-type]
    session.add(idea)
    session.flush()  # snapshots/experiments reference idea.id
    return idea


def save_snapshot(
    session: Session, idea: ProductIdea, features: dict, total: float
) -> ScoreSnapshot:
    snapshot = ScoreSnapshot(
        id=new_id("snap"), idea_id=idea.id, features=features, total=total
    )
    session.add(snapshot)
    return snapshot


def save_experiment(
    session: Session, idea: ProductIdea, spec: dict, status: str = "draft"
) -> Experiment:
    experiment = Experiment(id=new_id("exp"), idea_id=idea.id, spec=spec, status=status)
    session.add(experiment)
    session.flush()  # outcomes reference experiment.id
    return experiment


def record_outcome(
    session: Session, experiment: Experiment, kind: str, value: dict
) -> Outcome:
    outcome = Outcome(
        id=new_id("out"), experiment_id=experiment.id, kind=kind, value=value
    )
    session.add(outcome)
    return outcome


def hypotheses_for_vertical(session: Session, vertical: str) -> list[ProblemHypothesis]:
    stmt = select(ProblemHypothesis).where(ProblemHypothesis.vertical == vertical)
    return list(session.execute(stmt).scalars())


def ideas_for_vertical(session: Session, vertical: str) -> list[ProductIdea]:
    stmt = select(ProductIdea).where(ProductIdea.vertical == vertical)
    return list(session.execute(stmt).scalars())


def ideas_for_hypothesis(session: Session, hypothesis_id: str) -> list[ProductIdea]:
    stmt = select(ProductIdea).where(ProductIdea.hypothesis_id == hypothesis_id)
    return list(session.execute(stmt).scalars())


def latest_snapshot(session: Session, idea_id: str) -> ScoreSnapshot | None:
    stmt = (
        select(ScoreSnapshot)
        .where(ScoreSnapshot.idea_id == idea_id)
        .order_by(ScoreSnapshot.created_at.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def get_idea(session: Session, idea_id: str) -> ProductIdea | None:
    return session.get(ProductIdea, idea_id)


def get_hypothesis(session: Session, hypothesis_id: str) -> ProblemHypothesis | None:
    return session.get(ProblemHypothesis, hypothesis_id)


def evidence_by_ids(session: Session, ids: list[str]) -> list[EvidenceEvent]:
    if not ids:
        return []
    stmt = select(EvidenceEvent).where(EvidenceEvent.id.in_(ids))
    return list(session.execute(stmt).scalars())


# -- deletion lineage ---------------------------------------------------------

def lineage_counts(session: Session, evidence_id: str) -> dict[str, int]:
    segments = len(
        list(
            session.execute(
                select(TextSegment.id).where(TextSegment.evidence_id == evidence_id)
            ).scalars()
        )
    )
    claims = len(
        list(
            session.execute(
                select(PainClaim.id).where(PainClaim.evidence_id == evidence_id)
            ).scalars()
        )
    )
    memberships = len(
        list(
            session.execute(
                select(ClusterMember.cluster_id).where(
                    ClusterMember.evidence_id == evidence_id
                )
            ).scalars()
        )
    )
    return {
        "segments": segments,
        "claims": claims,
        "memberships": memberships,
    }


def delete_evidence_lineage(
    session: Session, evidence_ids: list[str], source: str, reason: str
) -> DeletionEvent:
    """Walk evidence -> segments/claims/memberships/raw and remove them.

    Hypotheses keep their evidence id lists; dangling ids signal deleted
    evidence (documented in deletion/service.py).
    """
    segments = claims = memberships = 0
    for evidence_id in evidence_ids:
        counts = lineage_counts(session, evidence_id)
        segments += counts["segments"]
        claims += counts["claims"]
        memberships += counts["memberships"]
        for stmt in (
            select(TextSegment).where(TextSegment.evidence_id == evidence_id),
            select(PainClaim).where(PainClaim.evidence_id == evidence_id),
            select(ClusterMember).where(ClusterMember.evidence_id == evidence_id),
            select(RawPayload).where(RawPayload.evidence_id == evidence_id),
        ):
            for row in session.execute(stmt).scalars():
                session.delete(row)
        event = session.get(EvidenceEvent, evidence_id)
        if event is not None:
            session.delete(event)
    deletion = DeletionEvent(
        id=new_id("del"),
        source=source,
        reason=reason,
        evidence_ids=list(evidence_ids),
        affected_claims=claims,
        affected_segments=segments,
        affected_memberships=memberships,
    )
    session.add(deletion)
    return deletion


# -- raw payloads (store_raw only) --------------------------------------------

def store_raw_payload(
    session: Session, evidence_id: str, body: dict, retention_deadline: datetime
) -> RawPayload:
    payload = RawPayload(
        evidence_id=evidence_id, body=body, retention_deadline=retention_deadline
    )
    session.add(payload)
    return payload
