"""Landing-page draft generator (markdown scaffold; A/B message variants)."""

from __future__ import annotations


def render_landing_markdown(idea, hypothesis, spec: dict) -> str:
    """Message-test-ready landing draft: outcome vs feature framing + price."""
    buyer = getattr(hypothesis, "buyer", None) or "the buyer"
    job = getattr(hypothesis, "job", None) or "the recurring job"
    pain = getattr(hypothesis, "pain", None) or "the manual work"
    price = spec["economics_guardrail"]["price_monthly"]
    ceiling = spec["economics_guardrail"]["cac_ceiling"]
    pitch = getattr(idea, "pitch", "") or f"{getattr(idea, 'form', 'offer')}"
    return f"""# Landing draft — {pitch}

> Variant A (feature framing) vs Variant B (outcome framing), 50/50
> randomized. Primary metric: {spec["primary_metric"]}.
> Economics guardrail: CAC ceiling **${ceiling}**
> (price ${price}/mo × margin × {spec["economics_guardrail"]["cac_payback_months"]}-month payback).

## Variant A — feature framing

# {pitch}

Handles {job} automatically. Built for {buyer}.

**${price}/month** · [Start paid pilot](#checkout)

## Variant B — outcome framing

# Stop losing time to {pain}

{buyer.title() if isinstance(buyer, str) else "You"} get {job} handled every
week — or you don't pay. First result in days.

**${price}/month** · [Start paid pilot](#checkout)

## Checkout (both variants)

- Visible price BEFORE the CTA (price-visibility test)
- Paid-pilot CTA or refundable deposit — measure payment, not signups
- Analytics events: `view`, `variant_seen`, `cta_click`, `checkout_start`,
  `payment_success`

## Stop conditions (predeclared)

- {spec.get("stop_condition")}
- No early winner from casual peeking; evaluate at planned sample size.
"""
