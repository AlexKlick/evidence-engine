#!/usr/bin/env bash
# Local gate — the only validation authority (NO hosted CI, hard rule).
# ruff + pytest, full output captured to a log; summary read from the log.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p gate-logs
STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="gate-logs/gate-${STAMP}.log"

{
  echo "== evidence-engine gate ${STAMP} =="
  echo "== ruff =="
} > "$LOG"

FAIL=0

uv run ruff check src tests >> "$LOG" 2>&1 || FAIL=1
ruff_line=$(grep -cE '\[[A-Z]+[0-9]+\]' "$LOG" || true)

{
  echo "== pytest =="
} >> "$LOG"
uv run pytest >> "$LOG" 2>&1 || FAIL=1

summary=$(grep -E '^[0-9]+ (passed|failed)|=+ .* =+|(passed|failed|error|skipped)' "$LOG" | tail -n 3)

echo "ruff findings: ${ruff_line:-0}"
echo "pytest summary:"
echo "${summary}"
echo "full log: ${LOG}"

if [ "$FAIL" -ne 0 ]; then
  echo "GATE: FAIL"
  tail -n 40 "$LOG"
  exit 1
fi
echo "GATE: PASS"
