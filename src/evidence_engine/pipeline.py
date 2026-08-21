"""Pipeline orchestration.

collect -> normalize/store -> embed -> dedupe -> intent/segments -> claims ->
cluster -> hypotheses -> forms -> scoring+snapshots -> experiments -> report.

Every consumption stage re-checks the policy gate for its purpose
(STORE_DERIVED, AGGREGATE, LOCAL_INFERENCE) — permitted-to-retrieve does not
imply permitted-to-use.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
from evidence_engine.nlp.segment import split_sentences
from evidence_engine.policy import PolicyRegistry, Purpose
from evidence_engine.sources.registry import build_adapters
from evidence_engine.store import init_db, make_engine, make_session_factory
from evidence_engine.store import repository as repo

logger = get_logger("pipeline")

EMBED_BATCH = 16


@dataclass
class CollectSummary:
    vertical: str
    runs: int = 0
    records: int = 0
    by_source: dict[str, dict] = field(default_factory=dict)


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

        with self.session_factory() as session:
            for position, query_text in enumerate(vertical.get("queries", [])):
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
            for group in groups[:10]:
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

            ideas_count = 0
            hypotheses = self._compile_hypotheses(
                session, slug, groups, cluster_rows, originals, by_id, claims_rows
            )
            ideas_count = hypotheses["idea_count"]

            session.commit()

        report_path = self._write_report(slug, collect_summary, embed_mode)
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
            {row.source for row in originals}, Purpose.LOCAL_INFERENCE
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
        if claims_rows and claims_rows[0].model_version != "heuristic-0.1":
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
                    idea, hypothesis, self.rubric.get("experiment_defaults") or {}
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

    # -- report ---------------------------------------------------------------
    def _write_report(self, slug: str, collect_summary: CollectSummary, embed_mode: str) -> Path:
        with self.session_factory() as session:
            markdown = self._render_report(session, slug, collect_summary, embed_mode)
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        path = self.settings.reports_dir / f"{slug}-{stamp}.md"
        path.write_text(markdown, encoding="utf-8")
        logger.info("report written: %s", path)
        return path

    def _render_report(
        self, session, slug: str, collect_summary: CollectSummary, embed_mode: str
    ) -> str:
        vertical = load_verticals(self.settings).get(slug, {})
        all_rows = repo.evidence_for_vertical(session, slug, originals_only=False)
        originals = [row for row in all_rows if not row.is_duplicate_of]
        hypotheses = repo.hypotheses_for_vertical(session, slug)
        ideas = sorted(
            repo.ideas_for_vertical(session, slug), key=lambda i: i.score_total, reverse=True
        )
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        lines: list[str] = [
            f"# Opportunity report — {vertical.get('name', slug)} (`{slug}`)",
            "",
            f"Generated {now} · evidence-engine 0.1.0 · policy v{self.policy.policy_version}",
            "",
            "> Scores are priors from documented heuristics, not validation.",
            "> Advance only on costly action: deposit, paid pilot, repeat payment.",
            "",
            "## Collection",
            "",
            "| source | runs | records | status |",
            "|---|---:|---:|---|",
        ]
        for name, entry in collect_summary.by_source.items():
            status = entry.get("status", "ok")
            reason = entry.get("reason")
            lines.append(
                f"| {name} | {entry.get('runs', 0)} | {entry.get('records', 0)} "
                f"| {status}{': ' + reason[:80] if reason and status != 'ok' else ''} |"
            )

        dup_count = len(all_rows) - len(originals)
        domains = len({row.domain for row in originals if row.domain})
        lines += [
            "",
            "## Evidence",
            "",
            f"- total collected: **{len(all_rows)}** · duplicates removed: **{dup_count}** "
            f"· independent evidence: **{len(originals)}** · unique domains: **{domains}**",
            f"- embeddings: {embed_mode} ({self.embedder.model_name})",
            "",
            "| intent | share of independent evidence |",
            "|---|---:|",
        ]
        counter = Counter(
            label for row in originals for label in (row.intent_labels or [])
        )
        for label, count in counter.most_common():
            share = count / max(len(originals), 1)
            lines.append(f"| {label} | {share:.0%} ({count}) |")

        lines += ["", "## Hypotheses", ""]
        for hypothesis in hypotheses:
            lines.append(f"### {hypothesis.title}")
            lines.append("")
            lines.append(f"- Buyer: {hypothesis.buyer or '—'}")
            lines.append(f"- Job: {hypothesis.job or '—'}")
            lines.append(f"- Pain: {hypothesis.pain or '—'}")
            lines.append(f"- Current workaround: {hypothesis.current_workaround or '—'}")
            lines.append(
                f"- Paid alternative: {hypothesis.current_paid_alternative or '—'}"
            )
            lines.append(f"- Channel: {hypothesis.channel or '—'}")
            lines.append(
                f"- Evidence: {len(hypothesis.evidence_for or [])} for / "
                f"{len(hypothesis.evidence_against or [])} against · "
                f"compliance: {hypothesis.compliance_status}"
            )
            lines.append("")

        lines += [
            "## Scored ideas",
            "",
            "| idea | form | score | band | failed gates |",
            "|---|---|---:|---|---|",
        ]
        for idea in ideas[:20]:
            failed = [name for name, ok in (idea.gates or {}).items() if not ok]
            lines.append(
                f"| `{idea.id}` | {idea.form} | {idea.score_total:.0f} | "
                f"{idea.band} | {', '.join(failed) or '—'} |"
            )

        for idea in ideas[:3]:
            hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
            if hypothesis is None:
                continue
            defaults = self.rubric.get("experiment_defaults") or {}
            spec = draft_experiment_spec(idea, hypothesis, defaults)
            lines += [
                "",
                f"## Next up: `{idea.id}` — {idea.pitch}",
                "",
                f"- Smallest paid test: {idea.smallest_paid_test}",
                f"- Pricing mechanism: {idea.pricing_mechanism}",
                f"- CAC ceiling: ${spec['economics_guardrail']['cac_ceiling']}"
                f" ({defaults.get('cac_payback_months', 6)}-month payback)",
                f"- Dimensions: {idea.score_dimensions}",
            ]

        lines += [
            "",
            "## Compliance footer",
            "",
            "| source | enabled | store_derived | commercial_use | retention |",
            "|---|---|---|---|---|",
        ]
        for name, source_policy in sorted(self.policy.all_sources().items()):
            rights = source_policy.rights
            lines.append(
                f"| {name} | {source_policy.enabled} | {rights.store_derived} "
                f"| {rights.commercial_use} | {source_policy.retention_days or '—'}d |"
            )
        lines += [
            "",
            "_evidence_engine — rights-aware evidence-to-revenue "
            "(docs/design/evidence-to-revenue.md)_",
        ]
        return "\n".join(lines) + "\n"


# re-exported for CLI convenience
__all__ = ["CollectSummary", "Pipeline", "PipelineResult", "purge_expired"]
