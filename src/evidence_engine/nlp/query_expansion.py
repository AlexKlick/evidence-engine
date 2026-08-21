"""Query expansion (the doc's `seed_expand` pipeline job).

Seed queries -> a broader, intent-mixed query set -> versioned yaml registry.

Rights posture: the ONLY input to expansion is operator-authored seed queries
(config/seed_verticals.yaml) — never collected evidence — so hosted-LLM
expansion does not exercise the EXTERNAL_INFERENCE right (no source data
leaves the host). Expanded queries land in a committed, human-editable yaml
sidecar; pruning the file before collecting is the approval step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from evidence_engine.logging_setup import get_logger
from evidence_engine.nlp.llm import LLMClient, LLMJsonError, LLMUnavailableError

logger = get_logger("nlp.query_expansion")

MAX_QUERY_CHARS = 80
DEFAULT_EXPANSION_LIMIT = 10

SYSTEM_PROMPT = """\
You expand market-research query sets for a specific niche.
Return ONLY a JSON object: {"queries": [{"text": "...", "intent": "..."}]}.

Rules:
- Every query must plausibly be typed by someone IN that niche with the problem.
- Mix intents and label each one: problem_aware | workaround | comparison |
  transactional | switching | feature_request.
- Max 8 words per query, natural search-engine phrasing, no quotes/brands spam.
- Do NOT repeat or trivially rephrase the given queries.
- Produce at most the requested number of queries."""


@dataclass
class ExpandedQuery:
    text: str
    intent: str
    origin: str = ""


def norm_query(text: str) -> str:
    """Case/whitespace-normalized form for dedupe."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def expanded_path(settings, slug: str) -> Path:
    return settings.config_dir / "expanded_queries" / f"{slug}.yaml"


def load_expanded(settings, slug: str) -> list[ExpandedQuery]:
    """Load the operator-editable expansion sidecar (missing file -> empty)."""
    path = expanded_path(settings, slug)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = []
    for entry in data.get("queries", []):
        if isinstance(entry, dict) and entry.get("text"):
            queries.append(
                ExpandedQuery(
                    text=str(entry["text"]),
                    intent=str(entry.get("intent", "unlabeled")),
                )
            )
        elif isinstance(entry, str):
            queries.append(ExpandedQuery(text=entry, intent="unlabeled"))
    return queries


def save_expanded(
    settings, slug: str, queries: list[ExpandedQuery], model: str
) -> Path:
    """Write the sidecar — committed config, so expansion is reviewable in git."""
    path = expanded_path(settings, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "origin": f"llm-expansion ({model}) — prune freely; collected on next run",
        "queries": [
            {"text": q.text, "intent": q.intent}
            for q in queries
        ],
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("expanded queries written: %s (%d)", path, len(queries))
    return path


def _sanitize(
    raw: object, known: set[str], limit: int, origin: str
) -> list[ExpandedQuery]:
    if isinstance(raw, dict):
        raw = raw.get("queries", [])
    if not isinstance(raw, list):
        return []
    result: list[ExpandedQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text or len(text) > MAX_QUERY_CHARS:
            continue
        if norm_query(text) in known:
            continue
        known.add(norm_query(text))
        result.append(
            ExpandedQuery(
                text=text,
                intent=str(item.get("intent", "unlabeled")),
                origin=origin,
            )
        )
        if len(result) >= limit:
            break
    return result


def expand_queries(
    llm: LLMClient,
    slug: str,
    seeds: list[str],
    existing: list[str],
    limit: int = DEFAULT_EXPANSION_LIMIT,
) -> list[ExpandedQuery]:
    """Expand a vertical's seed queries; dedupes against seeds + existing."""
    known = {norm_query(q) for q in [*seeds, *existing] if q}
    user = (
        f"Niche slug: {slug}\n"
        f"Seed queries (do not repeat):\n"
        + "\n".join(f"- {q}" for q in seeds)
        + f"\n\nProduce up to {limit} new queries as JSON now."
    )
    try:
        raw = llm.chat_json(SYSTEM_PROMPT, user, max_tokens=2048)
    except (LLMUnavailableError, LLMJsonError) as exc:
        logger.warning("query expansion failed: %s", exc)
        return []
    return _sanitize(raw, known, limit, origin=llm.model_name)


def effective_queries(vertical: dict, settings, slug: str) -> list[str]:
    """Seed queries + pruned expansion sidecar, deduped, seeds first."""
    seeds = list(vertical.get("queries", []))
    seen = {norm_query(q) for q in seeds}
    merged = list(seeds)
    for expanded in load_expanded(settings, slug):
        key = norm_query(expanded.text)
        if key not in seen:
            seen.add(key)
            merged.append(expanded.text)
    return merged
