#!/usr/bin/env bash
# Nightly operator ritual: full pipeline over every vertical + retention purge.
# Local-only (NO hosted CI — hard rule). Suggested crontab entry:
#   17 3 * * * /home/alexk/documents/evidence_engine/scripts/nightly.sh
#
# LLM extraction stays OFF by default (deterministic, no key/loopback
# dependency). Set EE_NIGHTLY_USE_LLM=1 in the cron environment to opt in —
# with EE_LLM_PROVIDER=minimax that path needs ANTHROPIC_AUTH_TOKEN_MINIMAX2
# exported too; never put secrets in this repo.
#
# flock guards nightly-vs-nightly overlap only; an interactive CLI run during
# the nightly can still hit SQLite's single-writer lock (accepted, one
# operator).
set -uo pipefail
cd "$(dirname "$0")/.."

# cron/systemd environments ship a minimal PATH; uv lives in ~/.local/bin.
export PATH="${HOME:-/home/alexk}/.local/bin:${PATH}"

mkdir -p gate-logs data
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="gate-logs/nightly-${STAMP}.log"

exec 9>data/nightly.lock
if ! flock -n 9; then
  echo "nightly: lock held — previous run still active, skipping" | tee -a "$LOG"
  exit 0
fi

LLM_FLAG="--no-llm"
[ "${EE_NIGHTLY_USE_LLM:-0}" = "1" ] && LLM_FLAG=""

FAIL=0
{
  echo "== evidence-engine nightly ${STAMP} =="
} > "$LOG"
uv run ee pipeline --all ${LLM_FLAG} >> "$LOG" 2>&1 || FAIL=1
# Compliance duty — runs even when collection failed.
uv run ee deletions purge >> "$LOG" 2>&1 || FAIL=1
echo "nightly exit: ${FAIL}" >> "$LOG"

tail -n 5 "$LOG"
if [ "$FAIL" -ne 0 ]; then
  echo "NIGHTLY: FAIL (full log: ${LOG})"
  exit 1
fi
echo "NIGHTLY: PASS (${LOG})"
