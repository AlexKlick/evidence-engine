"""SQLAlchemy 2.0 evidence graph.

Lineage (the deletion graph from the founding doc):

    query -> source_run -> evidence_event -> text_segment
                                          -> pain_claim
                                          -> embedding (column)
                              cluster_member -> cluster -> problem_hypothesis
                                                                 -> product_idea
                                                                     -> score_snapshot
                                                                     -> experiment -> outcome

Rights snapshots live on source_run and evidence_event so policy changes never
retroactively reinterpret old data.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class Query(Base):
    __tablename__ = "query"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("q"))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    vertical: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SourceRun(Base):
    __tablename__ = "source_run"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("run"))
    query_id: Mapped[str] = mapped_column(ForeignKey("query.id"), index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    adapter_version: Mapped[str] = mapped_column(String(16), default="")
    policy_version: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvidenceEvent(Base):
    __tablename__ = "evidence_event"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("ev"))
    source_run_id: Mapped[str] = mapped_column(ForeignKey("source_run.id"), index=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query.id"), index=True)
    vertical: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    domain: Mapped[str] = mapped_column(String(255), default="", index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    snippet: Mapped[str] = mapped_column(Text, default="")
    engine: Mapped[str] = mapped_column(String(64), default="")
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    published_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    language: Mapped[str] = mapped_column(String(8), default="en")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    near_dup_key: Mapped[str] = mapped_column(Text, default="")
    intent_labels: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_duplicate_of: Mapped[str | None] = mapped_column(
        ForeignKey("evidence_event.id"), nullable=True
    )
    dedupe_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rights_profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    policy_version: Mapped[int] = mapped_column(Integer, default=0)


class RawPayload(Base):
    """Raw provider payloads — only when rights.store_raw (none in v1)."""

    __tablename__ = "raw_payload"

    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_event.id"), primary_key=True
    )
    body: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retention_deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TextSegment(Base):
    __tablename__ = "text_segment"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("sg"))
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_event.id"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text, default="")


class PainClaim(Base):
    """Structured extraction artifact — must cite evidence_ids."""

    __tablename__ = "pain_claim"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("clm"))
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_event.id"), index=True
    )
    claim_type: Mapped[str] = mapped_column(String(16), default="pain")
    persona: Mapped[str | None] = mapped_column(Text, nullable=True)
    job: Mapped[str | None] = mapped_column(Text, nullable=True)
    obstacle: Mapped[str | None] = mapped_column(Text, nullable=True)
    consequence: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_workaround: Mapped[str | None] = mapped_column(Text, nullable=True)
    urgency: Mapped[float] = mapped_column(Float, default=0.0)
    commercial_intent: Mapped[str] = mapped_column(String(24), default="unaware")
    price_signal: Mapped[str | None] = mapped_column(Text, nullable=True)
    incumbents: Mapped[list | None] = mapped_column(JSON, nullable=True)
    evidence_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Cluster(Base):
    __tablename__ = "cluster"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("cl"))
    vertical: Mapped[str] = mapped_column(String(64), index=True)
    algorithm: Mapped[str] = mapped_column(String(32), default="greedy_cosine")
    label: Mapped[str] = mapped_column(Text, default="")
    member_count: Mapped[int] = mapped_column(Integer, default=0)
    policy_version: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ClusterMember(Base):
    __tablename__ = "cluster_member"

    cluster_id: Mapped[str] = mapped_column(
        ForeignKey("cluster.id"), primary_key=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_event.id"), primary_key=True
    )


class ProblemHypothesis(Base):
    """Populated only when the system can fill the doc's template fields."""

    __tablename__ = "problem_hypothesis"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("hyp"))
    cluster_id: Mapped[str | None] = mapped_column(
        ForeignKey("cluster.id"), nullable=True, index=True
    )
    vertical: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    buyer: Mapped[str | None] = mapped_column(Text, nullable=True)
    job: Mapped[str | None] = mapped_column(Text, nullable=True)
    pain: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_workaround: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_paid_alternative: Mapped[str | None] = mapped_column(Text, nullable=True)
    incumbent_failures: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_for: Mapped[list | None] = mapped_column(JSON, nullable=True)
    evidence_against: Mapped[list | None] = mapped_column(JSON, nullable=True)
    channel: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_form: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pricing_mechanism: Mapped[str | None] = mapped_column(String(32), nullable=True)
    smallest_paid_test: Mapped[str | None] = mapped_column(Text, nullable=True)
    fastest_mvp: Mapped[str | None] = mapped_column(Text, nullable=True)
    compliance_status: Mapped[str] = mapped_column(String(32), default="unreviewed")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ProductIdea(Base):
    __tablename__ = "product_idea"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("idea"))
    hypothesis_id: Mapped[str] = mapped_column(
        ForeignKey("problem_hypothesis.id"), index=True
    )
    vertical: Mapped[str] = mapped_column(String(64), index=True)
    form: Mapped[str] = mapped_column(String(32), nullable=False)
    pitch: Mapped[str] = mapped_column(Text, default="")
    pricing_mechanism: Mapped[str | None] = mapped_column(String(32), nullable=True)
    smallest_paid_test: Mapped[str | None] = mapped_column(Text, nullable=True)
    mvp_sketch: Mapped[str | None] = mapped_column(Text, nullable=True)
    score_total: Mapped[float] = mapped_column(Float, default=0.0)
    score_dimensions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    gates: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    band: Mapped[str] = mapped_column(String(24), default="archive")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ScoreSnapshot(Base):
    """Feature snapshot at scoring time — the no-hindsight-leakage record."""

    __tablename__ = "score_snapshot"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("snap"))
    idea_id: Mapped[str] = mapped_column(ForeignKey("product_idea.id"), index=True)
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experiment(Base):
    __tablename__ = "experiment"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("exp"))
    idea_id: Mapped[str] = mapped_column(ForeignKey("product_idea.id"), index=True)
    spec: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="draft")
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Outcome(Base):
    __tablename__ = "outcome"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("out"))
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiment.id"), index=True)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DeletionEvent(Base):
    __tablename__ = "deletion_event"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("del"))
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    evidence_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    affected_claims: Mapped[int] = mapped_column(Integer, default=0)
    affected_segments: Mapped[int] = mapped_column(Integer, default=0)
    affected_memberships: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
