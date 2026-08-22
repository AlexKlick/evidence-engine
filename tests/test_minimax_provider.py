"""MiniMax (M3) external provider: config resolution, key handling, rights gating.

External hosted inference is a RIGHTS decision, not a convenience: the
pipeline checks EXTERNAL_INFERENCE per source and falls back to the heuristic
extractor when denied (ADR-0001 degradation preserved).
"""

from __future__ import annotations

import textwrap

import pytest

from conftest import FakeEmbedder, FakeLLM
from evidence_engine.config import Settings
from evidence_engine.nlp.llm import LLMClient, LLMUnavailableError
from evidence_engine.pipeline import Pipeline
from evidence_engine.policy import PolicyRegistry, Purpose
from evidence_engine.store.models import PainClaim

VERTICAL = "local-ai-tooling"


class FakeExternalLLM(FakeLLM):
    model_name = "MiniMax-M3"
    inference_purpose = Purpose.EXTERNAL_INFERENCE


def test_settings_resolve_minimax_provider(monkeypatch) -> None:
    monkeypatch.setenv("EE_LLM_PROVIDER", "minimax")
    settings = Settings.load()
    assert settings.llm_provider == "minimax"
    assert settings.llm_base_url == "https://api.minimax.io/anthropic/v1"
    assert settings.llm_model == "MiniMax-M3"
    assert settings.llm_api_key_env == "ANTHROPIC_AUTH_TOKEN_MINIMAX2"
    client = LLMClient(settings)
    assert client.is_external
    assert client.inference_purpose is Purpose.EXTERNAL_INFERENCE


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
        assert all(claim.model_version == "MiniMax-M3" for claim in claims)


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
