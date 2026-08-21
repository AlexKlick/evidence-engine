# evidence_engine

A rights-aware **evidence-to-revenue** engine: mine search demand from
permitted sources, turn evidence into auditable problem hypotheses and product
ideas, score them, and stage willingness-to-pay experiments.

Founding design document: [`docs/design/evidence-to-revenue.md`](docs/design/evidence-to-revenue.md).
The architecture, source-rights analysis, scoring rubric and validation ladder
all come from there; this repo is its implementation.

## Core idea

The moat is not a scraped corpus — it is the closed loop:

```
market evidence → auditable problem hypotheses → paid experiments
→ observed revenue outcomes → better ranking → faster product bootstrap
```

Three rules make it durable:

- **No evidence, no idea.** Every promoted hypothesis cites concrete evidence IDs.
- **No score without a channel.** A valuable problem with no way to reach the buyer stays in research.
- **No source without an entitlement.** Adapters default to **off**; rights come
  from source contracts, never from "it is public".

## Source posture (v1)

| Source | Status | Why |
|---|---|---|
| SearXNG `:8018` (self-hosted) | **enabled** — personal research use | loopback, keyless; upstream terms not reviewed for commercial resale (`commercial_use: false`) |
| Google Ads `KeywordPlanIdeaService` | gated off | needs API developer token + approved customer ID |
| Commercial SERP provider | gated off | needs contracted provider with reviewed provenance/usage terms |
| YouTube | gated off | developer policy restricts API-Data aggregation; transcripts are rights-gated |
| Reddit | gated off | commercial use + derived-data revenue require a separate agreement |
| Independent web crawler | gated off | per-domain policy needed; robots.txt ≠ authorization (RFC 9309) |

See `config/source_policies.yaml` — the entitlement registry every adapter and
pipeline stage consults (default-deny, versioned, snapshotted onto each
evidence record at fetch time).

## Quickstart

```bash
uv sync --extra dev          # or: uv sync --extra dev --extra cluster --extra pg
uv run ee doctor             # health: DB, SearXNG :8018, LLM :18000, embeddings :6900
uv run ee policy show        # entitlement registry + reasons
uv run ee collect --vertical local-ai-tooling --limit 5
uv run ee pipeline --vertical local-ai-tooling --limit 5   # → reports/<vertical>-<date>.md
uv run ee ideas list
uv run ee bootstrap --idea <idea-id> --out ideas/          # scaffold a project folder
```

`--no-llm` runs extraction with the heuristic extractor instead of the local
model. All model traffic stays on loopback (`:18000` text-main, `:6900`
embeddings) per the workstation `MODEL_CONTRACT.json`.

## Layout

```
config/            evidence_engine.yaml, source_policies.yaml, scoring_rubric.yaml, seed_verticals.yaml
src/evidence_engine/
  policy/          entitlement registry + purpose gate (default-deny)
  sources/         adapters: searxng (live) + google_ads/serp_provider/youtube/reddit/web_crawler (gated stubs)
  store/           SQLAlchemy evidence graph: query→run→evidence→claims→clusters→hypotheses→ideas→experiments→outcomes
  nlp/             normalize, dedupe (exact/near/semantic), intent taxonomy, embeddings, pain-claim extraction, clustering
  ideas/           hypothesis compiler, product forms, weighted rubric + hard gates, feature snapshots
  experiments/     experiment specs, landing-page draft, payments stub, outcome recorder
  bootstrap/       per-idea project scaffold generator (PRD/evidence/pricing/acquisition/experiment.yaml)
  ranking/         no-hindsight feature snapshots + calibration (stub; learns from outcomes later)
  deletion/        retention TTLs + lineage-aware deletion propagation
  pipeline.py      orchestration: collect → normalize → dedupe → embed → classify → extract → cluster → score → report
  cli.py           typer CLI (`evidence-engine` / `ee`)
tests/             pytest suite (policy gate, adapters, dedupe, intent, scoring, deletion lineage, CLI)
scripts/gate.sh    local gate: ruff + pytest with logged counts (NO hosted CI — hard rule)
```

## Storage

Default SQLite (`data/evidence_engine.db`, gitignored). Set `database.dsn` in
`config/evidence_engine.yaml` for Postgres — the host cluster at
`127.0.0.1:5432` has pgvector installed (see ADR-0004 for the native-vector
upgrade path; the scaffold stores embeddings as JSON with in-process cosine).

## Validation ladder (what "validated" means)

Search views are weak evidence. Bands escalate only with costly action:
signup → interview → price-shown demo → deposit / paid concierge pilot →
repeat payment. Scoring output is a **prior**, never proof — the ranking
module exists to replace handcrafted weights with first-party outcomes once
experiments exist.

## Roadmap (from the founding doc)

- **Phase A (this scaffold)** — Google-first local pipeline + first-party outcomes.
- **Phase B** — revenue-generating research sprints on top of the platform.
- **Phase C** — automate what the paid service repeats into SaaS modules.
- **Phase D** — Reddit only under explicit commercial terms; YouTube only for rights-cleared uses.
- **Phase E** — calibrated ranking trained on owned outcome data.

## Gates

```bash
./scripts/gate.sh    # ruff check + pytest; log with counts under gate-logs/
```

No GitHub Actions / hosted CI (workspace hard rule). The local gate is the
only validation authority.
