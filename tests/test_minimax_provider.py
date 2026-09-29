"""MiniMax (M3.1) external provider: config resolution, key handling, rights gating.

External hosted inference is a RIGHTS decision, not a convenience: the
pipeline checks EXTERNAL_INFERENCE per source and falls back to the heuristic
extractor when denied (ADR-0001 degradation preserved).
"""

from __future__ import annotations

import json
import textwrap

import httpx
import pytest
import respx

from conftest import FakeEmbedder, FakeLLM
from evidence_engine.config import Settings
from evidence_engine.nlp.llm import LLMClient, LLMUnavailableError
from evidence_engine.pipeline import Pipeline
from evidence_engine.policy import PolicyRegistry, Purpose
from evidence_engine.store.models import PainClaim

VERTICAL = "local-ai-tooling"
MESSAGES_URL = "https://api.minimax.io/anthropic/v1/messages"


class FakeExternalLLM(FakeLLM):
    model_name = "MiniMax-M3.1-Flash-Preview"
    inference_purpose = Purpose.EXTERNAL_INFERENCE


def test_settings_resolve_minimax_provider(monkeypatch) -> None:
    monkeypatch.setenv("EE_LLM_PROVIDER", "minimax")
    monkeypatch.delenv("EE_LLM_MODEL", raising=False)
    monkeypatch.delenv("EE_LLM_EFFORT", raising=False)
    settings = Settings.load()
    assert settings.llm_provider == "minimax"
    assert settings.llm_base_url == "https://api.minimax.io/anthropic/v1"
    # owner 2026-09-28: every M3 pick moves to M3.1; it always thinks
    assert settings.llm_model == "MiniMax-M3.1-Flash-Preview"
    assert settings.minimax_model == "MiniMax-M3.1-Flash-Preview"
    assert settings.minimax_effort == "high"
    assert settings.minimax_reasoning_headroom == 4096
    assert settings.llm_api_key_env == "ANTHROPIC_AUTH_TOKEN_MINIMAX2"
    client = LLMClient(settings)
    assert client.is_external
    assert client.inference_purpose is Purpose.EXTERNAL_INFERENCE


def test_settings_dataclass_default_is_m31() -> None:
    assert Settings().minimax_model == "MiniMax-M3.1-Flash-Preview"
    assert Settings().minimax_effort == "high"


def test_effort_env_override_and_no_thinking_disable(monkeypatch) -> None:
    monkeypatch.setenv("EE_LLM_EFFORT", "LOW")
    assert Settings.load().minimax_effort == "low"
    for bad in ("none", "disabled", "off"):
        monkeypatch.setenv("EE_LLM_EFFORT", bad)
        with pytest.raises(ValueError, match="effort"):
            Settings.load()


def _external(settings: Settings, monkeypatch) -> LLMClient:
    settings.llm_provider = "minimax"
    settings.llm_base_url = "https://api.minimax.io/anthropic/v1"
    settings.llm_model = "MiniMax-M3.1-Flash-Preview"
    monkeypatch.setenv(settings.llm_api_key_env, "test-key-not-real")
    return LLMClient(settings)


def test_m31_request_names_effort_and_leaves_reasoning_room(settings, monkeypatch) -> None:
    client = _external(settings, monkeypatch)
    with respx.mock:
        route = respx.post(MESSAGES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "MiniMax-M3.1-Flash-Preview",
                    "content": [
                        {"type": "thinking", "thinking": "weighing the quotes"},
                        {"type": "text", "text": "{\"ok\": true}"},
                    ],
                },
            )
        )
        assert client.chat("system", "user", max_tokens=2048) == "{\"ok\": true}"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "MiniMax-M3.1-Flash-Preview"
    assert body["output_config"] == {"effort": "high"}
    assert body["max_tokens"] == 2048 + 4096
    assert "thinking" not in body


def test_substituted_model_is_refused(settings, monkeypatch) -> None:
    client = _external(settings, monkeypatch)
    with respx.mock:
        respx.post(MESSAGES_URL).mock(
            return_value=httpx.Response(
                200,
                json={"model": "MiniMax-M3", "content": [{"type": "text", "text": "x"}]},
            )
        )
        with pytest.raises(LLMUnavailableError, match="response named MiniMax-M3"):
            client.chat("system", "user")


def test_settings_default_stays_local(monkeypatch) -> None:
    monkeypatch.delenv("EE_LLM_PROVIDER", raising=False)
    settings = Settings.load()
    assert settings.llm_provider == "local"
    assert "127.0.0.1:18000" in settings.llm_base_url
    assert not LLMClient(settings).is_external


def test_external_provider_requires_key(monkeypatch, settings) -> None:
    settings.llm_provider = "minimax"
    monkeypatch.delenv(settings.llm_api_key_env, raising=False)
    client = LLMClient(settings)
    with pytest.raises(LLMUnavailableError, match=settings.llm_api_key_env):
        client.chat("system", "user")


def test_external_extraction_allowed_by_policy_v2(
    fake_adapters, session_factory, settings
) -> None:
    settings.llm_provider = "minimax"
    result = Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=FakeExternalLLM(),
    ).run(VERTICAL, limit=3, use_llm=True)
    assert result.extraction_mode == "llm"
    with session_factory() as session:
        claims = session.query(PainClaim).all()
        assert claims
        assert all(
            claim.model_version == "MiniMax-M3.1-Flash-Preview" for claim in claims
        )


def test_external_extraction_denied_without_right_falls_back(
    fake_adapters, session_factory, settings, tmp_path
) -> None:
    config = tmp_path / "policies.yaml"
    config.write_text(
        textwrap.dedent(
            """
            policy_version: 9
            sources:
              searxng:
                adapter: searxng
                enabled: true
                rights:
                  collect_enabled: true
                  store_derived: true
                  aggregate: true
                  local_inference: true
                  external_inference: false
            """
        ),
        encoding="utf-8",
    )
    policy = PolicyRegistry.load(config)
    settings.llm_provider = "minimax"
    result = Pipeline(
        settings=settings,
        policy=policy,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=FakeExternalLLM(),
    ).run(VERTICAL, limit=3, use_llm=True)
    assert result.extraction_mode == "heuristic"  # gate denied -> graceful fallback
