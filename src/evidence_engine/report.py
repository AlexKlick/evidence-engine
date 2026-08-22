"""Opportunity report rendering + writing.

Standalone on purpose: `ee report --vertical X` regenerates a report from the
store without collecting anything. The pipeline calls the same functions right
after a run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine import __version__
from evidence_engine.config import Settings, load_verticals
from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.ideas.competitors import (
    alternatives_overview,
    render_competitors_markdown,
)
from evidence_engine.ideas.review import review_queue
from evidence_engine.logging_setup import get_logger
from evidence_engine.policy import PolicyRegistry
from evidence_engine.store import repository as repo
from evidence_engine.store.models import EvidenceEvent, Query, SourceRun

logger = get_logger("report")


@dataclass
class CollectSummary:
    vertical: str
    runs: int = 0
    records: int = 0
    by_source: dict[str, dict] = field(default_factory=dict)


def collect_summary_from_db(session: Session, slug: str) -> CollectSummary:
    """Rebuild the collection summary from persisted source_run rows."""
    stmt = select(SourceRun).join(Query, SourceRun.query_id == Query.id).where(
        Query.vertical == slug
    )
    summary = CollectSummary(vertical=slug)
    for run in session.execute(stmt).scalars():
        entry = summary.by_source.setdefault(
            run.source, {"runs": 0, "records": 0, "status": "ok"}
        )
        entry["runs"] += 1
        entry["records"] += run.record_count or 0
        if run.status != "ok":
            entry["status"] = run.status
            entry["reason"] = run.reason
        summary.runs += 1
        summary.records += run.record_count or 0
    return summary


def stored_embed_mode(originals: list) -> str:
    """Embedding status for a store-only regeneration."""
    if any(row.embedding for row in originals):
        return "stored"
    return "none"


def velocity_summary(session: Session, slug: str) -> dict | None:
    """Recent evidence rate vs historical baseline, per day.

    The doc's velocity (recent normalized evidence rate / historical rate) on
    its daily-snapshot granularity. Returns None until evidence spans >= 2 days.
    """
    rows = session.execute(
        select(EvidenceEvent).where(EvidenceEvent.vertical == slug)
    ).scalars()
    by_day: dict[str, int] = {}
    for row in rows:
        day = row.fetched_at.date().isoformat()
        by_day[day] = by_day.get(day, 0) + 1
    days = sorted(by_day)
    if len(days) < 2:
        return None
    recent_day = days[-1]
    recent = by_day[recent_day]
    baseline = sum(by_day[day] for day in days[:-1]) / (len(days) - 1)
    return {
        "days": len(days),
        "recent_day": recent_day,
        "recent": recent,
        "baseline": round(baseline, 1),
        "velocity": round(recent / baseline, 2) if baseline > 0 else None,
    }


def render_vertical_report(
    session: Session,
    slug: str,
    *,
    settings: Settings,
    policy: PolicyRegistry,
    rubric: dict,
    embedding_model: str,
    collect_summary: CollectSummary,
    embed_mode: str,
) -> str:
    vertical = load_verticals(settings).get(slug, {})
    all_rows = repo.evidence_for_vertical(session, slug, originals_only=False)
    originals = [row for row in all_rows if not row.is_duplicate_of]
    hypotheses = repo.hypotheses_for_vertical(session, slug)
    ideas = sorted(
        repo.ideas_for_vertical(session, slug),
        key=lambda idea: idea.score_total,
        reverse=True,
    )
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# Opportunity report — {vertical.get('name', slug)} (`{slug}`)",
        "",
        f"Generated {now} · evidence-engine {__version__} · "
        f"policy v{policy.policy_version}",
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
    velocity = velocity_summary(session, slug)
    velocity_line = (
        f"- velocity: {velocity['velocity']}x "
        f"({velocity['recent_day']}: {velocity['recent']} vs baseline "
        f"{velocity['baseline']}/day over {velocity['days']} days)"
        if velocity
        else "- velocity: n/a (single day of evidence so far)"
    )
    lines += [
        "",
        "## Evidence",
        "",
        f"- total collected: **{len(all_rows)}** · duplicates removed: **{dup_count}** "
        f"· independent evidence: **{len(originals)}** · unique domains: **{domains}**",
        f"- embeddings: {embed_mode} ({embedding_model})",
        velocity_line,
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
        lines += [f"### {hypothesis.title}", ""]
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
        defaults = rubric.get("experiment_defaults") or {}
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

    overview = alternatives_overview(session, slug)
    if overview.direct_software:
        lines += ["", "## Competitor pressure", ""]
        lines += render_competitors_markdown(overview)

    queue = review_queue(session, slug, rubric=rubric)
    if queue:
        lines += [
            "",
            "## Review queue — hard gates open",
            "",
            "| hypothesis | best score | ideas | missing gates |",
            "|---|---:|---:|---|",
        ]
        for entry in queue[:5]:
            missing = (
                "_awaiting review sign-off (paid_validation capped)_"
                if entry.get("awaiting_review")
                else ", ".join(entry["missing"])
            )
            lines.append(
                f"| `{entry['hypothesis_id']}` ({entry['title'][:40]}) "
                f"| {entry['best_score']:.0f} | {entry['ideas']} "
                f"| {missing} |"
            )

    lines += [
        "",
        "## Compliance footer",
        "",
        "| source | enabled | store_derived | commercial_use | retention |",
        "|---|---|---|---|---|",
    ]
    for name, source_policy in sorted(policy.all_sources().items()):
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


def write_report(markdown: str, settings: Settings, slug: str) -> Path:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = settings.reports_dir / f"{slug}-{stamp}.md"
    suffix = 2
    while path.exists():  # same-second regeneration -> suffix, never overwrite
        path = settings.reports_dir / f"{slug}-{stamp}-{suffix}.md"
        suffix += 1
    path.write_text(markdown, encoding="utf-8")
    logger.info("report written: %s", path)
    return path
