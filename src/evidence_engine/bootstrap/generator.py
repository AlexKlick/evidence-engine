"""Project bootstrapper: winning idea -> testable-value project folder.

Optimizes for testable value, not architectural completeness. evidence.md
contains only permitted references (titles/urls/claims) — no raw content
redisplay for sources with redisplay_raw_content: false.
"""

from __future__ import annotations

from pathlib import Path


def _evidence_md(idea, hypothesis, evidence_rows) -> str:
    lines = [
        "# Evidence",
        "",
        f"Idea: `{idea.id}` ({idea.form}) — hypothesis `{hypothesis.id}`",
        "",
        "> Permitted references only. Claims cite evidence ids; raw content is",
        "> not redisplayed (rights.redisplay_raw_content = false for v1 sources).",
        "",
        "| evidence id | title | url | source | intents |",
        "|---|---|---|---|---|",
    ]
    for row in evidence_rows:
        intents = ", ".join(row.intent_labels or []) or "-"
        title = (row.title or "")[:80].replace("|", "/")
        lines.append(f"| {row.id} | {title} | {row.canonical_url} | {row.source} | {intents} |")
    against = hypothesis.evidence_against or []
    if against:
        lines += ["", "## Disconfirming evidence", ""]
        lines += [f"- `{evidence_id}` (anti-demand signal)" for evidence_id in against]
    lines += [
        "",
        f"- Evidence for: {len(hypothesis.evidence_for or [])} · against: {len(against)}",
    ]
    return "\n".join(lines) + "\n"


def _prd_md(idea, hypothesis) -> str:
    return f"""# PRD — {idea.pitch}

- **Buyer**: {hypothesis.buyer or "(fill in — hard gate)"}
- **Job**: {hypothesis.job or "(fill in)"}
- **Pain**: {hypothesis.pain or "(fill in)"}
- **Current workaround**: {hypothesis.current_workaround or "unknown"}
- **Paid alternative**: {hypothesis.current_paid_alternative or "none identified"}
- **Why incumbents fail**: {hypothesis.incumbent_failures or "unknown"}

## Wedge (testable value only)

{idea.mvp_sketch
 or hypothesis.fastest_mvp
 or "Concierge fulfillment first; automate repeated steps."}

## Non-goals (v0)

Everything that is not the central paid workflow.

## Acceptance

One user pays for the core workflow outcome within the experiment window.
"""


def _pricing_md(idea, spec) -> str:
    guard = spec["economics_guardrail"]
    price = guard.get("price_monthly")
    ceiling = guard.get("cac_ceiling")
    anchor = (
        f"${price}/month, {guard['gross_margin']:.0%} margin"
        if price is not None
        else "not set — no price evidence collected yet"
    )
    ceiling_line = (
        f"**${ceiling}**"
        if ceiling is not None
        else "**not set** (start gates on `ee experiments start --price`)"
    )
    lines = [
        "# Pricing",
        "",
        f"- Mechanism: {idea.pricing_mechanism}",
        f"- Anchor price: {anchor}",
        f"- CAC ceiling ({guard['cac_payback_months']}-month payback): {ceiling_line}",
        f"- Smallest paid test: {idea.smallest_paid_test}",
    ]
    if guard.get("price_provenance"):
        lines.append(f"- Price provenance: {guard['price_provenance']}")
    lines += [
        "",
        "> Anchor price is evidence-derived (median of collected price signals)",
        "> or operator-reviewed — never a silent default, and not a bid.",
    ]
    return "\n".join(lines) + "\n"


def _acquisition_md(hypothesis) -> str:
    channel = hypothesis.channel or "(none identified — hard gate)"
    return f"""# Acquisition

- Channel: {channel}
- Playbook order: high-intent evidence -> visible price -> paid concierge ->
  repeatable workflow -> narrow software -> scalable acquisition
- Founder-led outbound only from lawful business-contact channels; never from
  mined personal disclosures.
"""


def _risks_md(hypothesis) -> str:
    return f"""# Risks

- Compliance: {hypothesis.compliance_status}
- Evidence against demand: {len(hypothesis.evidence_against or [])} anti-demand signal(s)
- Engagement is not validation: advance only on costly action (deposit, paid pilot, repeat payment).
- Rights: if any source in lineage loses entitlement, delete via lineage walk
  (`ee deletions purge`) — do not keep stale derived data.
"""


def scaffold_idea_project(
    out_dir: Path,
    idea,
    hypothesis,
    evidence_rows: list,
    experiment_spec: dict,
) -> Path:
    """Write the doc's /idea/{id}/ folder structure; returns the directory."""
    root = Path(out_dir)
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "analytics").mkdir(exist_ok=True)
    (root / "tests").mkdir(exist_ok=True)
    (root / "deployment").mkdir(exist_ok=True)

    def write(rel: str, content: str) -> None:
        (root / rel).write_text(content, encoding="utf-8")

    write("evidence.md", _evidence_md(idea, hypothesis, evidence_rows))
    write("prd.md", _prd_md(idea, hypothesis))
    write("risks.md", _risks_md(hypothesis))
    write("pricing.md", _pricing_md(idea, experiment_spec))
    write("acquisition.md", _acquisition_md(hypothesis))
    write(
        "experiment.yaml",
        "\n".join(
            f"{key}: {value if not isinstance(value, (dict, list)) else value}"
            for key, value in experiment_spec.items()
        )
        + "\n",
    )
    write("app/README.md", "# App — minimal paid workflow only\n")
    write(
        "analytics/events.md",
        "# Analytics events\n\n- view, variant_seen, cta_click, checkout_start,\n"
        "- payment_success, activation, retention_60d\n",
    )
    write(
        "tests/README.md",
        "# Acceptance tests\n\n- core workflow completes\n- checkout path charges\n",
    )
    write(
        "deployment/README.md",
        "# Deployment\n\nLoopback first; public exposure is a separate decision.\n",
    )
    return root
