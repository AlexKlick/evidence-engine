"""Shared fixtures: tmp settings, policy, sqlite store, fake model clients."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evidence_engine.config import REPO_ROOT, Settings
from evidence_engine.nlp.embeddings import EmbeddingClient
from evidence_engine.policy import PolicyRegistry, Purpose
from evidence_engine.store import init_db, make_engine, make_session_factory

FAKE_DIM = 16


def fake_vector(text: str, salt: int = 0) -> list[float]:
    """Deterministic pseudo-embedding: one-hot bucket + per-item wobble."""
    bucket = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (FAKE_DIM - 1)
    wobble = (int(hashlib.md5(f"{text}{salt}".encode()).hexdigest(), 16) % 100) / 1000
    vector = [0.01] * FAKE_DIM
    vector[bucket] = 1.0 + wobble
    return vector


class FakeEmbedder(EmbeddingClient):
    """Offline stand-in: clusters form per query-bucket."""

    model_name = "fake-embed"

    def __init__(self) -> None:  # deliberately no Settings/super: offline fake
        pass

    def health(self, timeout: float = 4.0) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [fake_vector(text) for text in texts]


class FakeLLM:
    """Offline stand-in returning one claim per evidence, citing its id."""

    model_name = "fake-llm"
    inference_purpose = Purpose.LOCAL_INFERENCE  # class attr: fine as property stand-in

    def __init__(self) -> None:
        self.calls: list[str] = []

    def health(self, timeout: float = 5.0) -> bool:
        return True

    def chat_json(self, system: str, user: str, **kwargs):
        import re

        self.calls.append(user)
        ids = re.findall(r'"id":\s*"(ev_[0-9a-f]+)"', user)
        return {
            "claims": [
                {
                    "evidence_id": evidence_id,
                    "claim_type": "pain",
                    "persona": "solo operator",
                    "job": "keep local models running",
                    "obstacle": f"manual fiddling around {evidence_id}",
                    "consequence": "hours lost weekly",
                    "current_workaround": "spreadsheet",
                    "urgency": 0.8,
                    "commercial_intent": "solution_aware",
                    "price_or_budget_signal": None,
                    "incumbents": ["BigTool"],
                    "confidence": 0.8,
                }
                for evidence_id in ids
            ]
        }


class FakeAdapterModule:  # helper namespace for pipeline tests
    pass


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Isolated copy of the real config: tests may write sidecars (e.g.
    # expanded_queries) without polluting the repo's config dir.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "evidence_engine.yaml",
        "source_policies.yaml",
        "scoring_rubric.yaml",
        "seed_verticals.yaml",
    ):
        source = REPO_ROOT / "config" / name
        (config_dir / name).write_text(source.read_text(encoding="utf-8"))
    return Settings(
        config_dir=config_dir,
        sqlite_path=tmp_path / "test.db",
        reports_dir=tmp_path / "reports",
        ideas_dir=tmp_path / "ideas",
        searxng_delay=0.0,
        use_llm_extraction=False,
        embeddings_dim=FAKE_DIM,
    )


@pytest.fixture
def policy(settings: Settings) -> PolicyRegistry:
    return PolicyRegistry.load(settings.policies_path)


@pytest.fixture
def session_factory(settings: Settings):
    engine = make_engine(settings.db_url)
    init_db(engine)
    factory = make_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def session(session_factory):
    with session_factory() as db_session:
        yield db_session


@pytest.fixture
def fake_adapters(monkeypatch) -> None:
    """Replace the adapter registry with an offline fake 'searxng'."""
    import evidence_engine.pipeline as pipeline_module
    from evidence_engine.sources.base import SourceAdapter, SourceBatch, SourceRecord

    class FakeLiveAdapter(SourceAdapter):
        name = "searxng"
        version = "test"

        def _collect(self, query: str, limit: int) -> SourceBatch:
            import re as _re

            records = [
                SourceRecord(
                    url=(
                        "https://site"
                        f"{index}.com/docs/{_re.sub(r'[^a-z0-9]+', '', query.lower())}"
                    ),
                    title=f"how do i {query} step {index} spreadsheet",
                    snippet=(
                        "manual workaround takes forever; pricing per month is too "
                        "expensive, best alternative to BigTool comparison"
                    ),
                    engine="fake",
                )
                for index in range(3)
            ]
            return SourceBatch(
                source=self.name,
                adapter_version=self.version,
                status="ok",
                records=records,
            )

    monkeypatch.setattr(
        pipeline_module,
        "build_adapters",
        lambda policy, settings, names=None: {"searxng": FakeLiveAdapter(policy)},
    )
