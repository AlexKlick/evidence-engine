"""LLM client: loopback text-main lane (:18000) or hosted MiniMax (M3).

Provider is config/env selected (`llm.provider` / EE_LLM_PROVIDER). Which
rights purpose applies follows the provider:

  local   -> rights.local_inference      (loopback, ADR-0001 default)
  minimax -> rights.external_inference   (operator-authorized, policy v2)

JSON contract: strict prompt + fence-stripping parse + one retry.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from evidence_engine.config import Settings
from evidence_engine.logging_setup import get_logger
from evidence_engine.policy import Purpose

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

    # -- provider awareness ---------------------------------------------------
    @property
    def provider(self) -> str:
        return self._settings.llm_provider

    @property
    def is_external(self) -> bool:
        return self._settings.llm_provider != "local"

    @property
    def inference_purpose(self) -> Purpose:
        """The rights purpose this provider's inference requires."""
        return Purpose.EXTERNAL_INFERENCE if self.is_external else Purpose.LOCAL_INFERENCE

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    def _api_key(self) -> str:
        if not self.is_external:
            return ""
        key = os.environ.get(self._settings.llm_api_key_env, "")
        if not key:
            raise LLMUnavailableError(
                f"{self._settings.llm_api_key_env} not set — required for "
                f"provider {self.provider!r}"
            )
        return key

    def _headers(self) -> dict[str, str]:
        if not self.is_external:
            return {}
        # Anthropic-compatible endpoint (MiniMax Coding Plan coverage).
        return {
            "x-api-key": self._api_key(),
            "anthropic-version": "2023-06-01",
        }

    # -- API ------------------------------------------------------------------
    def health(self, timeout: float = 5.0) -> bool:
        try:
            base = self._settings.llm_base_url.rstrip("/")
            # external provider: model list lives on the openplatform root
            models_url = base.replace("/anthropic/v1", "/v1") + "/models"
            response = httpx.get(
                models_url,
                headers={"Authorization": f"Bearer {self._api_key()}"},
                timeout=timeout,
            )
            return response.status_code == 200
        except (httpx.HTTPError, LLMUnavailableError):
            return False

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        base = self._settings.llm_base_url.rstrip("/")
        try:
            if self.is_external:
                # Anthropic Messages shape. MiniMax may emit thinking blocks
                # (M2 always, M3 sometimes) — only text blocks carry the answer.
                response = httpx.post(
                    f"{base}/messages",
                    headers=self._headers(),
                    json={
                        "model": self._settings.llm_model,
                        "system": system,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{"role": "user", "content": user}],
                    },
                    timeout=self._settings.llm_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                blocks = payload.get("content") or []
                return "".join(
                    block.get("text", "")
                    for block in blocks
                    if block.get("type") == "text"
                )
            response = httpx.post(
                f"{base}/chat/completions",
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
