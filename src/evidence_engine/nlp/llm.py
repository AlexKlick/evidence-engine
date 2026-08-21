"""LLM client for the loopback text-main lane (:18000).

Local inference only — governed by each source's rights.local_inference flag.
JSON contract: strict prompt + fence-stripping parse + one retry.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from evidence_engine.config import Settings
from evidence_engine.logging_setup import get_logger

logger = get_logger("nlp.llm")

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMUnavailableError(RuntimeError):
    pass


class LLMJsonError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", (text or "").strip())


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    def health(self, timeout: float = 5.0) -> bool:
        try:
            response = httpx.get(
                f"{self._settings.llm_base_url.rstrip('/')}/models", timeout=timeout
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        try:
            response = httpx.post(
                f"{self._settings.llm_base_url.rstrip('/')}/chat/completions",
                json={
                    "model": self._settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self._settings.llm_timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"] or ""
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(str(exc)) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(f"malformed chat response: {exc}") from exc

    def chat_json(
        self, system: str, user: str, max_tokens: int = 2048, temperature: float = 0.0
    ) -> Any:
        """Chat expecting a JSON object; one repair retry; parse or raise."""
        first = self.chat(system, user, max_tokens=max_tokens, temperature=temperature)
        candidates = [_strip_fences(first)]
        retry = self.chat(
            system,
            user + "\n\nReturn ONLY the JSON object. No prose, no code fences.",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        candidates.append(_strip_fences(retry))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # tolerate trailing prose around a {...} or [...] block
                match = re.search(r"[\[{].*[\]}]", candidate, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
        raise LLMJsonError("model did not return parseable JSON after retry")
