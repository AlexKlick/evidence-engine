"""Landing-page draft generator (shared content model + markdown emitter)."""

from __future__ import annotations

from dataclasses import dataclass

from evidence_engine.ideas.forms import FORMS, _outcome


@dataclass(frozen=True)
class LandingContent:
    """Variant-independent landing copy shared by the markdown + HTML emitters."""

    pitch: str
    buyer: str
    job: str
    pain: str
    price_monthly: float | None  # evidence-derived; None until price signals exist
    cac_ceiling: float | None
    payback_months: int
    primary_metric: str
    stop_condition: str
    headline_a: str  # feature framing
    body_a: str
    headline_b: str  # outcome framing
    body_b: str


def synthesize_pitch(idea, hypothesis) -> str:
    """Re-derive the pitch from CURRENT hypothesis fields + the form template.

    idea.pitch is frozen at pipeline time and once carried verbatim scraped
    text (lane B); it is never trusted for publication. Unknown forms fall
    back to a neutral line built from job/pain.
    """
    form = getattr(idea, "form", None)
    spec = next((s for s in FORMS if s.form == form), None)
    if spec is not None:
        incumbent = (
            getattr(hypothesis, "current_paid_alternative", None) or "A->B"
        )
        return spec.pitch_template.format(
            outcome=_outcome(hypothesis), incumbent=incumbent
        )
    job = getattr(hypothesis, "job", None) or "the recurring job"
    pain = getattr(hypothesis, "pain", None) or "the manual work"
    return f"{form or 'offer'} that handles: {job} — {pain}"


def landing_content(idea, hypothesis, spec: dict) -> LandingContent:
    """Build the shared copy: reviewed hypothesis fields + rubric arithmetic."""
    buyer = getattr(hypothesis, "buyer", None) or "the buyer"
    job = getattr(hypothesis, "job", None) or "the recurring job"
    pain = getattr(hypothesis, "pain", None) or "the manual work"
    pitch = synthesize_pitch(idea, hypothesis)
    guardrail = spec["economics_guardrail"]
    return LandingContent(
        pitch=pitch,
        buyer=buyer,
        job=job,
        pain=pain,
        price_monthly=guardrail["price_monthly"],
        cac_ceiling=guardrail["cac_ceiling"],
        payback_months=guardrail["cac_payback_months"],
        primary_metric=spec["primary_metric"],
        stop_condition=str(spec.get("stop_condition")),
        headline_a=pitch,
        body_a=f"Handles {job} automatically. Built for {buyer}.",
        headline_b=f"Stop losing time to {pain}",
        body_b=(
            f"{buyer.title() if isinstance(buyer, str) else 'You'} get {job} "
            "handled every week — or you don't pay. First result in days."
        ),
    )


def render_landing_markdown(idea, hypothesis, spec: dict) -> str:
    """Message-test-ready landing draft: outcome vs feature framing + price."""
    content = landing_content(idea, hypothesis, spec)
    return f"""# Landing draft — {content.pitch}

> Variant A (feature framing) vs Variant B (outcome framing), 50/50
> randomized. Primary metric: {content.primary_metric}.
> Economics guardrail: CAC ceiling **${content.cac_ceiling}**
> (price ${content.price_monthly}/mo × margin × {content.payback_months}-month payback).

## Variant A — feature framing

# {content.headline_a}

{content.body_a}

**${content.price_monthly}/month** · [Start paid pilot](#checkout)

## Variant B — outcome framing

# {content.headline_b}

{content.body_b}

**${content.price_monthly}/month** · [Start paid pilot](#checkout)

## Checkout (both variants)

- Visible price BEFORE the CTA (price-visibility test)
- Paid-pilot CTA or refundable deposit — measure payment, not signups
- Analytics events: `view`, `variant_seen`, `cta_click`, `checkout_start`,
  `payment_success`

## Stop conditions (predeclared)

- {content.stop_condition}
- No early winner from casual peeking; evaluate at planned sample size.
"""
