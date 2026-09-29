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
uv run ee pipeline --all --limit 5                         # every configured vertical
uv run ee report --vertical local-ai-tooling               # regenerate report from the store
uv run ee ideas list [--band collect_more]
# decision support: what to review, who you'd displace, what to ship
uv run ee review --queue                                   # failing hard gates OR awaiting sign-off, ranked
# paid_validation is an operator-reviewed state: pipeline scoring caps at interview
# even when the score crosses 80 — re-affirm fields via `ee review -H` to unlock
uv run ee competitors --vertical local-ai-tooling          # incumbent pressure from evidence
uv run ee landing --idea <idea-id> --html                   # A/B draft + static landing-a/b.html + events.json (ADR-0005)
# operator loop: review the human gate, record outcomes, watch calibration
uv run ee review --hypothesis <hyp-id> --buyer "..." --channel search --smallest-paid-test "..."
uv run ee outcomes record --experiment <exp-id> --kind deposit_paid --value '{"amount":50}'
uv run ee experiments list --status draft                     # predeclared spend caps
uv run ee experiments start --experiment <exp-id>             # draft→running (prints the cap)
uv run ee experiments stop --experiment <exp-id> --decision advance
uv run ee calibration
uv run ee bootstrap --idea <idea-id> --out ideas/          # scaffold a project folder
```

`--no-llm` runs extraction with the heuristic extractor. Extraction defaults
to the loopback lanes (`:18000` text-main, `:6900` embeddings) per the
workstation `MODEL_CONTRACT.json`; while GPU 0 is unavailable, set
`EE_LLM_PROVIDER=minimax` (with `ANTHROPIC_AUTH_TOKEN_MINIMAX2` sourced from
`~/.claude/.env`) to extract via hosted `MiniMax-M3.1-Flash-Preview` (the
default since 2026-09-28; it named `MiniMax-M3` before) — authorized for derived
searxng data under policy v2's `external_inference_authorization`. M3.1 always
thinks: requests carry `output_config.effort` (`llm.minimax.effort`, default
`high`; `EE_LLM_EFFORT` overrides) plus reasoning headroom on `max_tokens`, and
a response naming a different model is refused.

**Repository history note.** A small number of early commits referenced operator-specific URLs (a personal SearXNG origin, an ngrok tunnel, and routing profile names) that are no longer present in `HEAD`. They were removed from source as the public posture hardened, but the commits remain in the git log for forensic continuity. Force-pushing the history would break clones and forks and is not warranted; reviewers can ignore those URLs in `git log -p` outputs.

## Nightly operator loop

`scripts/nightly.sh` runs `ee pipeline --all --no-llm` (deterministic — no
key or loopback dependency) plus the retention purge, under a flock so runs
never overlap, logging to `gate-logs/nightly-<stamp>.log` and exiting
non-zero when any vertical fails. Install once:

```bash
crontab -e
# 17 3 * * * /path/to/evidence_engine/scripts/nightly.sh
```

A second collection day is what makes the report's velocity line compute
(evidence-rate ratio vs the day-1 baseline). LLM extraction in the nightly is
opt-in via `EE_NIGHTLY_USE_LLM=1` in the service/cron environment.

Where crontab is unavailable (e.g. `/var/spool/cron` denied), use a
**systemd user timer** instead — `evidence-engine-nightly.timer` at 03:17
local, `Persistent=true`:

```bash
mkdir -p ~/.config/systemd/user
# service: ExecStart=<repo>/scripts/nightly.sh, WorkingDirectory=<repo>
# timer: OnCalendar=*-*-* 03:17:00, Persistent=true
systemctl --user daemon-reload && systemctl --user enable --now evidence-engine-nightly.timer
systemctl --user list-timers evidence-engine-nightly.timer
```

## Layout

```
config/            evidence_engine.yaml, source_policies.yaml, scoring_rubric.yaml, seed_verticals.yaml
src/evidence_engine/
  policy/          entitlement registry + purpose gate (default-deny)
  sources/         adapters: searxng (live) + google_ads/serp_provider/youtube/reddit/web_crawler (gated stubs)
  store/           SQLAlchemy evidence graph: query→run→evidence→claims→clusters→hypotheses→ideas→experiments→outcomes
  nlp/             normalize, dedupe (exact/near/semantic), intent taxonomy, embeddings, pain-claim extraction, clustering
  ideas/           hypothesis compiler, product forms, weighted rubric + hard gates, feature snapshots, human review
  experiments/     experiment specs, landing-page draft, payments stub, outcome recorder
  bootstrap/       per-idea project scaffold generator (PRD/evidence/pricing/acquisition/experiment.yaml)
  ranking/         no-hindsight feature snapshots + calibration (stub; learns from outcomes later)
  deletion/        retention TTLs + lineage-aware deletion propagation
  pipeline.py      orchestration: collect → normalize → dedupe → embed → classify → extract → cluster → score → report
  cli.py           typer CLI (`evidence-engine` / `ee`)
tests/             pytest suite (policy gate, adapters, dedupe, intent, scoring, deletion lineage, CLI)
scripts/gate.sh    local gate: ruff + pytest with logged counts (NO hosted CI — hard rule)
scripts/nightly.sh nightly operator ritual: pipeline --all + retention purge (flock, cron)
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

## Status

Research scaffold, developed in the open. It runs a real nightly loop and has
driven live willingness-to-pay experiments, but it is not a packaged product:
adapters beyond self-hosted SearXNG are deliberately gated off, and the
serving/analytics layer is intentionally minimal (see ADR-0005).

Two design commitments are worth knowing before you read the code:

- **Prices are never invented.** Experiment economics come from parsed
  evidence or an explicit operator override, never a default — a hardcoded
  placeholder once set a real public price and spend cap, and the
  `price_provenance` field plus a lifecycle gate exist to prevent that class
  of bug.
- **Published copy is never copied.** Landing artifacts are synthesized from
  reviewed hypothesis fields, and a redisplay guard rejects any artifact
  sharing a long verbatim span with cited source evidence.

## License

MIT — see [LICENSE](LICENSE).
- **Phase C** — automate what the paid service repeats into SaaS modules.
- **Phase D** — Reddit only under explicit commercial terms; YouTube only for rights-cleared uses.
- **Phase E** — calibrated ranking trained on owned outcome data.

## Gates

```bash
./scripts/gate.sh    # ruff check + pytest; log with counts under gate-logs/
```

No GitHub Actions / hosted CI (workspace hard rule). The local gate is the
only validation authority.
