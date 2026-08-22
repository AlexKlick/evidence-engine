"""Competitor map — derived only from already-collected, entitled evidence.

The founding doc's competitor table (entry_price, pricing_unit, ...) needs
per-domain crawling of competitor pages, which stays rights-gated
(web_crawler, per-domain policy). What the current evidence already supports:
incumbent names extracted by claims, how often each is mentioned, switching
pressure, and which personas complain. The doc's three alternative classes
are derivable today: direct software (incumbents), manual workaround
(workaround claims), do nothing (anti-demand evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from evidence_engine.store.models import EvidenceEvent, PainClaim


@dataclass
class CompetitorEntry:
    name: str
    mention_count: int = 0
    switching_count: int = 0
    personas: list[str] = field(default_factory=list)
    gap_signals: list[str] = field(default_factory=list)

    @property
    def switching_share(self) -> float:
        return self.switching_count / self.mention_count if self.mention_count else 0.0


@dataclass
class AlternativesOverview:
    direct_software: list[CompetitorEntry]
    manual_workaround_claims: int
    do_nothing_evidence: int


def vertical_claims(session: Session, vertical: str) -> list[PainClaim]:
    stmt = select(PainClaim).join(
        EvidenceEvent, PainClaim.evidence_id == EvidenceEvent.id
    ).where(EvidenceEvent.vertical == vertical)
    return list(session.execute(stmt).scalars())


def competitor_map(session: Session, vertical: str) -> list[CompetitorEntry]:
    """Group claims by extracted incumbent; rank by switching pressure."""
    claims = vertical_claims(session, vertical)
    by_name: dict[str, CompetitorEntry] = {}
    for claim in claims:
        for incumbent in claim.incumbents or []:
            key = incumbent.strip().lower()
            if not key:
                continue
            entry = by_name.setdefault(key, CompetitorEntry(name=key))
            entry.mention_count += 1
            if claim.claim_type == "switching":
                entry.switching_count += 1
            if claim.persona and claim.persona not in entry.personas:
                entry.personas.append(claim.persona)
            if claim.obstacle and claim.obstacle not in entry.gap_signals:
                entry.gap_signals.append(claim.obstacle)
    return sorted(
        by_name.values(),
        key=lambda entry: (entry.switching_count, entry.mention_count),
        reverse=True,
    )


def alternatives_overview(session: Session, vertical: str) -> AlternativesOverview:
    """The doc's three alternative classes, from current evidence."""
    claims = vertical_claims(session, vertical)
    manual = sum(1 for claim in claims if claim.current_workaround)
    evidence = list(
        session.execute(
            select(EvidenceEvent).where(EvidenceEvent.vertical == vertical)
        ).scalars()
    )
    do_nothing = sum(
        1 for row in evidence if "anti_demand" in (row.intent_labels or [])
    )
    return AlternativesOverview(
        direct_software=competitor_map(session, vertical),
        manual_workaround_claims=manual,
        do_nothing_evidence=do_nothing,
    )


def render_competitors_markdown(overview: AlternativesOverview, top: int = 8) -> list[str]:
    """Markdown block for the report / `ee competitors`."""
    lines = [
        "| incumbent | mentions | switching | switching share | personas |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in overview.direct_software[:top]:
        lines.append(
            f"| {entry.name} | {entry.mention_count} | {entry.switching_count} "
            f"| {entry.switching_share:.0%} | {', '.join(entry.personas[:2]) or '—'} |"
        )
    lines += [
        "",
        f"Alternative classes: direct software: **{len(overview.direct_software)}** · "
        f"manual workarounds: **{overview.manual_workaround_claims}** claims · "
        f"do-nothing signals: **{overview.do_nothing_evidence}** evidence",
        "",
        "> Pricing/feature detail per competitor needs the rights-gated web_crawler",
        "> (per-domain policy) — not fetched here.",
    ]
    return lines
