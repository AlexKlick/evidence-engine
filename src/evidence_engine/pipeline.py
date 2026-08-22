"""Pipeline orchestration.

collect -> normalize/store -> embed -> dedupe -> intent/segments -> claims ->
cluster -> hypotheses -> forms -> scoring+snapshots -> experiments -> report.

Every consumption stage re-checks the policy gate for its purpose
(STORE_DERIVED, AGGREGATE, LOCAL_INFERENCE) — permitted-to-retrieve does not
imply permitted-to-use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from evidence_engine.config import Settings, load_rubric, load_verticals
from evidence_engine.deletion.service import purge_expired
from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.ideas.forms import generate_forms
from evidence_engine.ideas.hypothesis import build_hypothesis
from evidence_engine.ideas.scoring import aggregate_features, score_idea
from evidence_engine.logging_setup import get_logger
from evidence_engine.nlp.clustering import cluster_evidence
from evidence_engine.nlp.dedupe import find_duplicates
from evidence_engine.nlp.embeddings import EmbeddingClient, EmbeddingUnavailableError
from evidence_engine.nlp.intent import classify
from evidence_engine.nlp.llm import LLMClient
from evidence_engine.nlp.pain_claims import extract as extract_claims
from evidence_engine.nlp.pricing import band_from_claims
from evidence_engine.nlp.query_expansion import effective_queries
from evidence_engine.nlp.segment import split_sentences
from evidence_engine.policy import PolicyRegistry, Purpose
from evidence_engine.report import CollectSummary, render_vertical_report, write_report
from evidence_engine.sources.registry import build_adapters
from evidence_engine.store import init_db, make_engine, make_session_factory
from evidence_engine.store import repository as repo

logger = get_logger("pipeline")

EMBED_BATCH = 16


@dataclass
class PipelineResult:
    vertical: str
    report_path: Path
    collect: CollectSummary
    originals: int = 0
    duplicates: int = 0
    claims: int = 0
    clusters: int = 0
    hypotheses: int = 0
    ideas: int = 0
    extraction_mode: str = "heuristic"


@dataclass
class PipelineBatch:
    """run_all() outcome: one result per vertical that succeeded."""

    results: list[PipelineResult] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # slug -> error


@dataclass
class CollectBatch:
    """collect_all() outcome: one summary per vertical that succeeded."""

    summaries: list[CollectSummary] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # slug -> error


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        policy: PolicyRegistry | None = None,
        session_factory=None,
        embedder: EmbeddingClient | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.policy = policy or PolicyRegistry.load(self.settings.policies_path)
        self.rubric = load_rubric(self.settings)
        if session_factory is None:
            engine = make_engine(self.settings.db_url)
            init_db(engine)
            session_factory = make_session_factory(engine)
        self.session_factory = session_factory
        self.embedder = embedder or EmbeddingClient(self.settings)
        self.llm = llm or LLMClient(self.settings)

    # -- helpers ------------------------------------------------------------
    def vertical(self, slug: str) -> dict:
        verticals = load_verticals(self.settings)
        if slug not in verticals:
            known = ", ".join(sorted(verticals)) or "(none)"
            raise KeyError(f"unknown vertical {slug!r}; known: {known}")
        return verticals[slug]

    # -- collection ---------------------------------------------------------
    def collect(
        self, slug: str, limit: int | None = None, sources: list[str] | None = None
    ) -> CollectSummary:
        vertical = self.vertical(slug)
        limit = limit or self.settings.default_max_results
        names = sources or vertical.get("sources") or self.settings.default_sources
        adapters = build_adapters(self.policy, self.settings, names)
        summary = CollectSummary(vertical=slug)

        queries = effective_queries(vertical, self.settings, slug)
        with self.session_factory() as session:
            for position, query_text in enumerate(queries):
                if position and any(
                    summary.by_source.get(name, {}).get("live") for name in names
                ):
                    time.sleep(min(self.settings.searxng_delay, 2.0))
                query = repo.upsert_query(session, query_text, slug)
                for name in names:
                    adapter = adapters.get(name)
                    if adapter is None:
                        run = repo.record_run(
                            session, query, name, "-", 0, "error",
                            "no adapter registered for source",
                        )
                        summary.by_source.setdefault(name, {"runs": 0, "records": 0})
                        summary.by_source[name]["runs"] += 1
                        summary.runs += 1
                        continue
                    batch = adapter.collect(query_text, limit)
                    run = repo.record_run(
                        session,
                        query,
                        name,
                        batch.adapter_version,
                        self.policy.policy_version,
                        batch.status,
                        batch.reason,
                    )
                    events = repo.persist_batch(
                        session, run, query, batch, self.policy
                    )
                    entry = summary.by_source.setdefault(
                        name, {"runs": 0, "records": 0, "live": False}
                    )
                    entry["runs"] += 1
                    entry["records"] += len(events)
                    if batch.status == "ok":
                        entry["live"] = True
                        entry["status"] = "ok"
                    elif batch.status == "disabled":
                        entry["status"] = "disabled"
                        entry["reason"] = batch.reason
                    else:
                        entry["status"] = batch.status
                        entry["reason"] = batch.reason
                    summary.runs += 1
                    summary.records += len(events)
            session.commit()
        return summary

    # -- batch (every configured vertical) ------------------------------------
    def collect_all(
        self, limit: int | None = None, sources: list[str] | None = None
    ) -> CollectBatch:
        """Collect every configured vertical; one failure never stops the rest."""
        batch = CollectBatch()
        for slug in load_verticals(self.settings):
            try:
                batch.summaries.append(
                    self.collect(slug, limit=limit, sources=sources)
                )
            except Exception as exc:  # noqa: BLE001 — isolate the batch
                logger.exception("collect failed for vertical %s", slug)
                batch.failed[slug] = str(exc)[:200]
        return batch

    def run_all(
        self, limit: int | None = None, use_llm: bool | None = None
    ) -> PipelineBatch:
        """Full loop over every configured vertical; one failure stops nothing."""
        batch = PipelineBatch()
        for slug in load_verticals(self.settings):
            try:
                batch.results.append(self.run(slug, limit=limit, use_llm=use_llm))
            except Exception as exc:  # noqa: BLE001 — isolate the batch
                logger.exception("pipeline failed for vertical %s", slug)
                batch.failed[slug] = str(exc)[:200]
        return batch

    # -- full run -------------------------------------------------------------
    def run(
        self, slug: str, limit: int | None = None, use_llm: bool | None = None
    ) -> PipelineResult:
        collect_summary = self.collect(slug, limit=limit)
        use_llm = self.settings.use_llm_extraction if use_llm is None else use_llm

        with self.session_factory() as session:
            all_rows = repo.evidence_for_vertical(session, slug, originals_only=False)
            embed_mode = self._embed_missing(session, all_rows)

            dedupe = find_duplicates(
                all_rows, self.settings.semantic_duplicate_threshold
            )
            by_id = {row.id: row for row in all_rows}
            for stage, dup_id, original_id in dedupe.all_pairs:
                repo.mark_duplicate(
                    session, by_id[dup_id], by_id[original_id], stage
                )
            originals = [row for row in all_rows if not row.is_duplicate_of]

            for row in originals:
                repo.set_intent(
                    session, row, classify(f"{row.title} {row.snippet}")
                )
                repo.add_segments(
                    session, row, split_sentences(f"{row.title}. {row.snippet}")
                )

            claims_rows = self._extract(session, originals, use_llm)

            # clustering is AGGREGATE — gate per involved source
            aggregated = self._gate_all(
                {row.source for row in originals}, Purpose.AGGREGATE
            )
            groups = (
                cluster_evidence(all_rows, self.settings.cluster_similarity_threshold)
                if aggregated and len(originals) >= 2
                else []
            )

            cluster_rows = []
            capped = groups[:10]  # compile only the top clusters per run
            if len(groups) > len(capped):
                logger.info(
                    "clustering produced %d groups; compiling top %d", len(groups), len(capped)
                )
            for group in capped:
                cluster_rows.append(
                    repo.create_cluster(
                        session,
                        vertical=slug,
                        label=group.label,
                        member_ids=group.member_ids,
                        policy_version=self.policy.policy_version,
                        embedding_model=self.embedder.model_name,
                    )
                )

            hypotheses = self._compile_hypotheses(
                session, slug, capped, cluster_rows, originals, by_id, claims_rows
            )
            ideas_count = hypotheses["idea_count"]

            session.commit()

        with self.session_factory() as session:
            markdown = render_vertical_report(
                session,
                slug,
                settings=self.settings,
                policy=self.policy,
                rubric=self.rubric,
                embedding_model=self.embedder.model_name,
                collect_summary=collect_summary,
                embed_mode=embed_mode,
            )
        report_path = write_report(markdown, self.settings, slug)
        return PipelineResult(
            vertical=slug,
            report_path=report_path,
            collect=collect_summary,
            originals=len(originals),
            duplicates=len(dedupe.duplicate_ids),
            claims=len(claims_rows),
            clusters=len(cluster_rows),
            hypotheses=hypotheses["hypothesis_count"],
            ideas=ideas_count,
            extraction_mode=hypotheses["extraction_mode"],
        )

    # -- stages ---------------------------------------------------------------
    def _embed_missing(self, session, rows: list) -> str:
        """Embed rows lacking vectors; returns 'local' | 'unavailable'."""
        missing = [row for row in rows if not row.embedding and not row.is_duplicate_of]
        if not missing:
            return "cached"
        try:
            for start in range(0, len(missing), EMBED_BATCH):
                batch = missing[start : start + EMBED_BATCH]
                vectors = self.embedder.embed(
                    [f"{row.title}\n{row.snippet}" for row in batch]
                )
                for row, vector in zip(batch, vectors, strict=True):
                    repo.set_embedding(
                        session, row, vector, self.embedder.model_name
                    )
            return "local"
        except EmbeddingUnavailableError as exc:
            logger.warning("embeddings unavailable (%s); semantic stages degraded", exc)
            return "unavailable"

    def _gate_all(self, sources: set[str], purpose: Purpose) -> bool:
        for source in sources:
            if not self.policy.decision(source, purpose).allowed:
                logger.warning(
                    "stage %s denied for source %s — skipping downstream stage",
                    purpose.value,
                    source,
                )
                return False
        return True

    def _extract(self, session, originals: list, use_llm: bool):
        llm = None
        if use_llm and self._gate_all(
            {row.source for row in originals}, self.llm.inference_purpose
        ):
            llm = self.llm
        result = extract_claims(
            originals,
            llm=llm,
            batch_size=self.settings.llm_max_batch,
            limit=self.settings.llm_max_evidences,
        )
        claims_rows = []
        for claim in result.claims:
            evidence = session.get(
                repo.EvidenceEvent, claim.get("evidence_id")
            )
            if evidence is not None:
                claims_rows.append(repo.add_claim(session, evidence, claim))
        return claims_rows

    def _compile_hypotheses(
        self, session, slug, groups, cluster_rows, originals, by_id, claims_rows
    ) -> dict:
        claims_by_evidence = {claim.evidence_id: claim for claim in claims_rows}
        hypothesis_count = 0
        idea_count = 0
        extraction_mode = "heuristic"
        # any heuristic-* version is heuristic (stored heuristic-0.1 rows must
        # not mislabel as llm now that the constant has moved on)
        if claims_rows and not claims_rows[0].model_version.startswith("heuristic"):
            extraction_mode = "llm"

        def compile_one(cluster_row, member_rows, label):
            nonlocal hypothesis_count, idea_count
            member_claims = [
                claims_by_evidence[row.id]
                for row in member_rows
                if row.id in claims_by_evidence
            ]
            rights_clear = all(
                (policy := self.policy.get(row.source)) is not None
                and policy.enabled
                and bool((row.rights_profile or {}).get("store_derived"))
                for row in member_rows
            )
            draft = build_hypothesis(
                vertical=slug,
                cluster_id=cluster_row.id if cluster_row else None,
                cluster_label=label,
                evidence_rows=member_rows,
                claims=member_claims,
                rights_clear=rights_clear,
            )
            hypothesis = repo.save_hypothesis(
                session,
                vertical=slug,
                cluster_id=draft.cluster_id,
                title=draft.title,
                buyer=draft.buyer,
                job=draft.job,
                pain=draft.pain,
                current_workaround=draft.current_workaround,
                current_paid_alternative=draft.current_paid_alternative,
                incumbent_failures=draft.incumbent_failures,
                evidence_for=draft.evidence_for,
                evidence_against=draft.evidence_against,
                channel=draft.channel,
                suggested_form=draft.suggested_form,
                pricing_mechanism=draft.pricing_mechanism,
                smallest_paid_test=draft.smallest_paid_test,
                fastest_mvp=draft.fastest_mvp,
                compliance_status=draft.compliance_status,
            )
            hypothesis_count += 1
            features = aggregate_features(member_rows, member_claims)
            price_band = band_from_claims(member_claims)
            candidates = []
            for form in generate_forms(draft):
                result = score_idea(features, draft, form["form"], self.rubric)
                candidates.append((result, form))
            candidates.sort(key=lambda pair: pair[0].total, reverse=True)
            for result, form in candidates[:3]:
                idea = repo.save_idea(
                    session,
                    vertical=slug,
                    hypothesis_id=hypothesis.id,
                    form=form["form"],
                    pitch=form["pitch"],
                    pricing_mechanism=form["pricing_mechanism"],
                    smallest_paid_test=form["smallest_paid_test"],
                    mvp_sketch=form["mvp_sketch"],
                    score_total=result.total,
                    score_dimensions=result.dimensions,
                    gates=result.gates,
                    band=result.band,
                )
                repo.save_snapshot(session, idea, features, result.total)
                spec = draft_experiment_spec(
                    idea,
                    hypothesis,
                    self.rubric.get("experiment_defaults") or {},
                    price_band=price_band,
                )
                repo.save_experiment(session, idea, spec)
                idea_count += 1

        if groups:
            for group, cluster_row in zip(groups, cluster_rows, strict=True):
                member_rows = [by_id[mid] for mid in group.member_ids if mid in by_id]
                compile_one(cluster_row, member_rows, group.label)
        elif originals:
            # too little evidence to cluster — one whole-vertical hypothesis
            compile_one(None, originals, "all evidence (below clustering floor)")

        return {
            "hypothesis_count": hypothesis_count,
            "idea_count": idea_count,
            "extraction_mode": extraction_mode,
        }


# re-exported for CLI convenience (CollectSummary lives in report.py)
__all__ = [
    "CollectBatch",
    "CollectSummary",
    "Pipeline",
    "PipelineBatch",
    "PipelineResult",
    "purge_expired",
]
