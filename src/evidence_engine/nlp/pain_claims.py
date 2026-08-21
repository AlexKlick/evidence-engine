"""Pain-claim extraction: local LLM (structured, evidence-cited) with a
deterministic heuristic fallback.

The critical field is evidence_ids — an LLM statement without traceable
source evidence is brainstorming, not research. Claims citing unknown
evidence ids are dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from evidence_engine.logging_setup import get_logger
from evidence_engine.nlp.intent import classify
from evidence_engine.nlp.llm import LLMClient, LLMJsonError, LLMUnavailableError

logger = get_logger("nlp.pain_claims")

HEURISTIC_MODEL_VERSION = "heuristic-0.1"

SYSTEM_PROMPT = """\
You extract structured market-research claims from search-result evidence.
Return ONLY a JSON object: {"claims": [...]}.

Each claim object has exactly these fields:
  evidence_id (string, MUST be one of the provided ids),
  claim_type ("pain" | "workaround" | "wish" | "switching" | "praise"),
  persona (string or null),
  job (string or null),
  obstacle (string or null),
  consequence (string or null),
  current_workaround (string or null),
  urgency (number 0-1),
  commercial_intent ("unaware" | "problem_aware" | "solution_aware" | "comparison"),
  price_or_budget_signal (string or null),
  incumbents (list of strings),
  confidence (number 0-1).

Rules:
- Use null/empty when the text does not support a field. NEVER invent facts.
- Only produce a claim when the text expresses a problem, workaround, wish,
  or switching intent. Skip pure navigation/homepage results.
- urgency reflects lost time/money/risk language, not emotion."""


@dataclass
class ExtractionResult:
    claims: list[dict[str, Any]]
    mode: str  # "llm" | "heuristic" | "mixed"


def _user_prompt(evidences: list[Any]) -> str:
    lines = [
        json.dumps(
            {"id": e.id, "title": e.title, "snippet": (e.snippet or "")[:600]},
            ensure_ascii=False,
        )
        for e in evidences
    ]
    return "Evidence:\n" + "\n".join(lines) + (
        "\n\nExtract claims as JSON now. Remember: only claims supported by "
        "the evidence, each citing its evidence_id."
    )


def _sanitize_claims(raw: Any, valid_ids: set[str], model_version: str) -> list[dict]:
    if isinstance(raw, dict):
        raw = raw.get("claims", [])
    if not isinstance(raw, list):
        return []
    claims: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id")
        if evidence_id not in valid_ids:
            continue  # untraceable -> drop
        claim = dict(item)
        claim["evidence_ids"] = [evidence_id]
        claim["model_version"] = model_version
        claims.append(claim)
    return claims


def extract_with_llm(
    llm: LLMClient, evidences: list[Any], batch_size: int, limit: int
) -> list[dict[str, Any]]:
    """Batched LLM extraction over evidences (capped at `limit`)."""
    selected = evidences[:limit]
    claims: list[dict[str, Any]] = []
    valid_ids = {e.id for e in selected}
    for start in range(0, len(selected), batch_size):
        batch = selected[start : start + batch_size]
        raw = llm.chat_json(SYSTEM_PROMPT, _user_prompt(batch))
        claims.extend(_sanitize_claims(raw, valid_ids, llm.model_name))
    return claims


def heuristic_claims(evidences: list[Any]) -> list[dict[str, Any]]:
    """Deterministic fallback: claims only for problem/feature/workaround text."""
    claims: list[dict[str, Any]] = []
    wanted = {"problem_aware", "feature_request", "workaround"}
    for evidence in evidences:
        text = f"{evidence.title} {evidence.snippet}"
        labels = set(classify(text))
        if not labels & wanted:
            continue
        obstacle = evidence.title if evidence.title.endswith(("?",)) else (
            evidence.snippet[:200] if evidence.snippet else evidence.title
        )
        claims.append(
            {
                "evidence_id": evidence.id,
                "claim_type": "workaround" if "workaround" in labels else "pain",
                "persona": None,
                "job": None,
                "obstacle": obstacle,
                "consequence": None,
                "current_workaround": None,
                "urgency": 0.4,
                "commercial_intent": "problem_aware",
                "price_or_budget_signal": None,
                "incumbents": [],
                "confidence": 0.3,
                "evidence_ids": [evidence.id],
                "model_version": HEURISTIC_MODEL_VERSION,
            }
        )
    return claims


def extract(
    evidences: list[Any],
    llm: LLMClient | None = None,
    batch_size: int = 6,
    limit: int = 24,
) -> ExtractionResult:
    """LLM-first with graceful heuristic fallback."""
    if llm is not None and evidences:
        try:
            claims = extract_with_llm(llm, evidences, batch_size, limit)
            if claims:
                return ExtractionResult(claims=claims, mode="llm")
            logger.info("llm extraction returned no claims; falling back to heuristic")
        except (LLMUnavailableError, LLMJsonError) as exc:
            logger.warning("llm extraction failed (%s); using heuristic fallback", exc)
    return ExtractionResult(claims=heuristic_claims(evidences), mode="heuristic")
