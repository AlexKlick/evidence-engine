# evidence_engine — project instructions

Rights-aware evidence-to-revenue engine. Founding design doc:
`docs/design/evidence-to-revenue.md` (authoritative for intent; code is the
source of truth for behavior).

## Hard rules

- **No GitHub Actions / hosted CI.** Validation = `./scripts/gate.sh`
  (ruff + pytest, counts logged to `gate-logs/`). Never propose CI configs.
- **Adapters default to off.** Any new source needs a row in
  `config/source_policies.yaml` with rights from the actual source contract.
  Default-deny is enforced in `policy/` — don't bypass the gate.
- **Local models only** for inference over collected evidence: text-main
  `http://127.0.0.1:18000/v1` (Qwen3.8-27B), embeddings
  `http://127.0.0.1:6900/v1` (Qwen3-Embedding-0.6B, 1024-dim). Endpoints
  mirror `~/documents/MODEL_CONTRACT.json`. External hosted-model inference
  over collected data is a rights decision, not a convenience — keep
  `external_inference: false` until reviewed.
- No scraping of Google/YouTube/Reddit directly. SearXNG (self-hosted,
  loopback `:8018`) is the only live collection path in v1, personal-research
  posture (`commercial_use: false`).

## Conventions

- Python 3.12, uv, src layout, SQLAlchemy 2.0 typed ORM, pydantic v2, typer CLI.
- Type annotations required; `ruff check .` clean before commit; tests in `tests/`.
- Logging (stdlib `logging`), not print, inside library code. CLI output via typer.
- Data (`data/`, `reports/`, `ideas/`) is gitignored local state.
- Evidence lineage is sacred: features must cite `evidence_ids`; deletion must
  walk lineage (`deletion/service.py`), never ad-hoc deletes.

## Key entry points

- CLI: `uv run ee --help` (doctor, policy, collect, pipeline, report, ideas,
  review, outcomes, calibration, bootstrap, deletions, verticals).
- Pipeline orchestration: `src/evidence_engine/pipeline.py`; report rendering
  is standalone in `src/evidence_engine/report.py` (`ee report` regenerates
  without collecting).
- Human review (`ee review`) rescores ideas against STORED feature snapshots —
  features never change retroactively; only gates/bands respond to review.
- Entitlement registry: `config/source_policies.yaml` + `src/evidence_engine/policy/`.

## Gates

`./scripts/gate.sh` before handing anything over. Commit when green
(right-size process: gate once, no ceremony).
