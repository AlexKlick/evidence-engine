"""Self-contained static landing HTML (one file per A/B variant) + events schema.

Rights posture: every rendered string is synthesized from store_derived
evidence (reviewed hypothesis fields + rubric arithmetic) — no raw source
content is redisplayed. Files are inert: inline CSS only, zero scripts, no
external requests. Serving/exposure is an operator decision made outside
this module (see ADR-0005).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from evidence_engine.experiments.landing import LandingContent

EVENT_NAMES = (
    "view",
    "variant_seen",
    "cta_click",
    "checkout_start",
    "payment_success",
)

_STYLE = """\
body { font-family: system-ui, sans-serif; margin: 0; color: #1a1a2e; }
main { max-width: 640px; margin: 0 auto; padding: 64px 24px 32px; }
h1 { font-size: 2rem; line-height: 1.2; }
p.lead { font-size: 1.1rem; color: #333; }
p.price { font-size: 1.25rem; font-weight: 600; margin: 24px 0 8px; }
a.cta { display: inline-block; margin: 8px 0 32px; padding: 12px 24px;
        background: #1a1a2e; color: #ffffff; text-decoration: none;
        font-weight: 600; }
footer { border-top: 1px solid #dddddd; margin-top: 48px; padding-top: 16px;
         font-size: 0.85rem; color: #666666; }
"""


def events_schema() -> list[dict]:
    """Analytics contract for the landing (documented, not implemented)."""
    return [
        {"name": "view", "when": "page load", "payload": {"variant": "a|b"}},
        {
            "name": "variant_seen",
            "when": "headline visible",
            "payload": {"variant": "a|b"},
        },
        {
            "name": "cta_click",
            "when": "Start paid pilot clicked",
            "payload": {"variant": "a|b"},
        },
        {
            "name": "checkout_start",
            "when": "#checkout anchor reached",
            "payload": {"variant": "a|b", "price_monthly": "number"},
        },
        {
            "name": "payment_success",
            "when": "deposit/pilot payment completed",
            "payload": {"variant": "a|b", "amount": "number"},
        },
    ]


def render_landing_html(content: LandingContent, variant: str) -> str:
    """One self-contained page per variant ('a' | 'b'); escaped copy, no JS."""
    if variant == "a":
        headline, body, framing = content.headline_a, content.body_a, "feature framing"
    else:
        headline, body, framing = content.headline_b, content.body_b, "outcome framing"
    esc = html.escape
    price_month = (
        f"${content.price_monthly}/month"
        if content.price_monthly is not None
        else "price TBD"
    )
    price_mo = (
        f"${content.price_monthly}/mo" if content.price_monthly is not None else "TBD"
    )
    cac = f"${content.cac_ceiling}" if content.cac_ceiling is not None else "unset"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(content.pitch)}</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
<h1>{esc(headline)}</h1>
<p class="lead">{esc(body)}</p>
<p class="price">{price_month} — visible before the CTA.</p>
<a class="cta" href="#checkout" id="checkout">Start paid pilot</a>
<footer>
Variant {variant.upper()} ({esc(framing)}) · 50/50 randomized ·
primary metric: {esc(content.primary_metric)} ·
CAC ceiling {cac}
(price {price_mo} × margin × {content.payback_months}-mo payback) ·
stop condition: {esc(content.stop_condition)}
</footer>
</main>
</body>
</html>
"""


def write_landing_html(target: Path, content: LandingContent) -> list[Path]:
    """Write landing-a.html, landing-b.html, events.json under target."""
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for variant in ("a", "b"):
        path = target / f"landing-{variant}.html"
        path.write_text(render_landing_html(content, variant), encoding="utf-8")
        paths.append(path)
    events = target / "events.json"
    events.write_text(
        json.dumps({"events": events_schema()}, indent=2) + "\n", encoding="utf-8"
    )
    paths.append(events)
    return paths
