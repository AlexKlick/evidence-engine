# ADR-0005: Static landing export, no server

Date: 2026-08-21
Status: accepted

## Context

The validation ladder needs a payment-test artifact: a landing page with the
price visible before the CTA, A/B message variants, and a predeclared
analytics event contract. The founding doc's "experiment factory" describes
templates, checkout, and analytics as the Phase B serving layer — but nothing
in the engine has traffic to serve yet, the payments provider is deliberately
an unconfigured stub pending a `commercial_use` rights review, and standing
up a web server adds an always-on surface (and an exposure decision) before
any of that is earned.

## Decision

1. Landing pages are exported as **static, self-contained HTML** — one file
   per A/B variant (`landing-a.html`, `landing-b.html`), inline CSS, zero
   `<script>`, zero external URLs/requests, previewable from `file://`.
2. The analytics contract is **documented, not implemented**: `events.json`
   lists the five ladder events (`view`, `variant_seen`, `cta_click`,
   `checkout_start`, `payment_success`) with payload shapes. No runtime
   emits them yet; wiring a sink is a later, separate decision.
3. Copy comes from the shared `LandingContent` model (reviewed hypothesis
   fields + rubric arithmetic), `html.escape`d on render. All strings are
   synthesized from `store_derived` evidence — no raw source content is
   redisplayed.
4. Serving/exposure (hosting the files anywhere reachable) is an operator
   decision made outside this repo, when there is a real experiment to run.
   Nothing in the engine listens on a port.

## Consequences

- The engine emits deploy-anywhere artifacts with no server to operate, and
  the A/B copy stays single-sourced between markdown and HTML emitters.
- No metrics flow until an analytics sink is wired — acceptable: the first
  experiments are concierge-style (the operator observes and records
  outcomes via `ee outcomes record`), not self-serve funnels.
- If/when a served experiment is warranted, add a separate ADR for the
  serving posture (the workspace has a hardened public-gateway standard for
  exactly this) rather than growing this module into a server.

## Addendum — 2026-08-21: exposure decided

The operator approved serving the first landing experiment (options-chain
price test, `exp_446adf139ed8`) on the workspace public hub:

- `scripts/serve_landing.py` — read-only allowlist static server (exactly
  `landing-a.html`, `landing-b.html`, `events.json` + `/healthz`; GET/HEAD
  only; 404 for anything else including traversal) on loopback `:18087`,
  run by the user unit `evidence-landing.service` owned by this repo.
- Route `/apps/options-chain/` on `alexklick.ngrok.app` via the hub
  (opencode-stack `server-surfaces.json` entry `evidence-landing`, mirroring
  the fractal-page open-static-public allowance).
- The ADR's "no server inside the engine" stance stands: the server is a
  3-file allowlist in `scripts/`, not a serving framework — it cannot serve
  anything that is not one of the exported artifacts.
