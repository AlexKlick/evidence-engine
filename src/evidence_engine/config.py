"""Runtime configuration loading (config/evidence_engine.yaml + friends).

Env overrides:
  EVIDENCE_ENGINE_CONFIG — path to a main config yaml
  EE_DB_DSN              — force a database DSN (wins over yaml)
  EE_CONFIG_DIR          — directory holding the yaml files
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


@dataclass
class Settings:
    """Typed view over the yaml config files."""

    config_dir: Path = DEFAULT_CONFIG_DIR
    database_dsn: str = ""
    sqlite_path: Path = REPO_ROOT / "data" / "evidence_engine.db"
    searxng_base_url: str = "http://127.0.0.1:8018"
    searxng_timeout: float = 10.0
    searxng_max_results: int = 10
    searxng_delay: float = 0.5
    searxng_language: str = "en"
    llm_base_url: str = "http://127.0.0.1:18000/v1"
    llm_model: str = "Qwen/Qwen3.8-27B"
    # "local" (loopback) | "minimax" (hosted; needs EXTERNAL_INFERENCE right)
    llm_provider: str = "local"
    llm_api_key_env: str = "ANTHROPIC_AUTH_TOKEN_MINIMAX2"
    # MiniMax Coding Plan covers the Anthropic-compatible endpoint; the
    # OpenAI-style /v1/chat/completions path is pay-as-you-go (1008 balance).
    minimax_base_url: str = "https://api.minimax.io/anthropic/v1"
    minimax_model: str = "MiniMax-M3"
    llm_timeout: float = 240.0
    llm_max_batch: int = 6
    llm_max_evidences: int = 24
    embeddings_base_url: str = "http://127.0.0.1:6900/v1"
    embeddings_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embeddings_dim: int = 1024
    embeddings_timeout: float = 60.0
    semantic_duplicate_threshold: float = 0.97
    cluster_similarity_threshold: float = 0.72
    use_llm_extraction: bool = True
    reports_dir: Path = REPO_ROOT / "reports"
    ideas_dir: Path = REPO_ROOT / "ideas"
    default_sources: list[str] = field(default_factory=lambda: ["searxng"])
    default_max_results: int = 10

    # -- derived paths ----------------------------------------------------
    @property
    def policies_path(self) -> Path:
        return self.config_dir / "source_policies.yaml"

    @property
    def rubric_path(self) -> Path:
        return self.config_dir / "scoring_rubric.yaml"

    @property
    def verticals_path(self) -> Path:
        return self.config_dir / "seed_verticals.yaml"

    @property
    def db_url(self) -> str:
        if dsn := os.environ.get("EE_DB_DSN") or self.database_dsn:
            return dsn
        return f"sqlite:///{self.sqlite_path}"

    # -- factories ---------------------------------------------------------
    @classmethod
    def load(cls, config_dir: Path | None = None) -> Settings:
        config_dir = Path(
            os.environ.get("EE_CONFIG_DIR") or config_dir or DEFAULT_CONFIG_DIR
        )
        env_cfg = os.environ.get("EVIDENCE_ENGINE_CONFIG")
        raw = (
            _load_yaml(Path(env_cfg))
            if env_cfg
            else _load_yaml(config_dir / "evidence_engine.yaml")
        )

        db = raw.get("database", {}) or {}
        sqlite_path = Path(db.get("sqlite_path") or "data/evidence_engine.db")
        if not sqlite_path.is_absolute():
            sqlite_path = REPO_ROOT / sqlite_path

        searx = raw.get("searxng", {}) or {}
        llm = raw.get("llm", {}) or {}
        minimax = llm.get("minimax", {}) or {}
        emb = raw.get("embeddings", {}) or {}
        pipe = raw.get("pipeline", {}) or {}
        paths = raw.get("paths", {}) or {}
        defaults = raw.get("defaults", {}) or {}

        def _dir(key: str, fallback: Path, env: str = "") -> Path:
            override = os.environ.get(env) if env else None
            p = Path(override or paths.get(key) or fallback.name)
            return p if p.is_absolute() else REPO_ROOT / p

        # LLM provider resolution: local loopback by default; "minimax" routes
        # extraction to the hosted MiniMax API (external-inference rights apply).
        provider = (os.environ.get("EE_LLM_PROVIDER") or llm.get("provider") or "local").lower()
        if provider == "minimax":
            base_default = minimax.get("base_url", "https://api.minimax.io/v1")
            model_default = minimax.get("model", "MiniMax-M3")
        else:
            provider = "local"
            base_default = llm.get("base_url", "http://127.0.0.1:18000/v1")
            model_default = llm.get("model", "Qwen/Qwen3.8-27B")

        return cls(
            config_dir=config_dir,
            database_dsn=db.get("dsn") or "",
            sqlite_path=sqlite_path,
            searxng_base_url=searx.get("base_url", "http://127.0.0.1:8018"),
            searxng_timeout=float(searx.get("timeout_seconds", 10.0)),
            searxng_max_results=int(searx.get("max_results_per_query", 10)),
            searxng_delay=float(searx.get("delay_seconds", 0.5)),
            searxng_language=searx.get("language", "en"),
            llm_provider=provider,
            llm_base_url=os.environ.get("EE_LLM_BASE_URL") or base_default,
            llm_model=os.environ.get("EE_LLM_MODEL") or model_default,
            llm_api_key_env=minimax.get(
                "api_key_env", "ANTHROPIC_AUTH_TOKEN_MINIMAX2"
            ),
            minimax_base_url=minimax.get(
                "base_url", "https://api.minimax.io/anthropic/v1"
            ),
            minimax_model=minimax.get("model", "MiniMax-M3"),
            llm_timeout=float(llm.get("timeout_seconds", 240.0)),
            llm_max_batch=int(llm.get("max_batch_evidences", 6)),
            llm_max_evidences=int(llm.get("max_evidences_per_run", 24)),
            embeddings_base_url=emb.get("base_url", "http://127.0.0.1:6900/v1"),
            embeddings_model=emb.get("model", "Qwen/Qwen3-Embedding-0.6B"),
            embeddings_dim=int(emb.get("dim", 1024)),
            embeddings_timeout=float(emb.get("timeout_seconds", 60.0)),
            semantic_duplicate_threshold=float(pipe.get("semantic_duplicate_threshold", 0.97)),
            cluster_similarity_threshold=float(pipe.get("cluster_similarity_threshold", 0.72)),
            use_llm_extraction=bool(pipe.get("use_llm_extraction", True)),
            reports_dir=_dir("reports_dir", REPO_ROOT / "reports", env="EE_REPORTS_DIR"),
            ideas_dir=_dir("ideas_dir", REPO_ROOT / "ideas", env="EE_IDEAS_DIR"),
            default_sources=list(defaults.get("sources", ["searxng"])),
            default_max_results=int(defaults.get("max_results_per_query", 10)),
        )


def load_verticals(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return {slug: vertical-dict} from seed_verticals.yaml."""
    data = _load_yaml(settings.verticals_path)
    return dict(data.get("verticals", {}))


def load_rubric(settings: Settings) -> dict[str, Any]:
    """Return the parsed scoring rubric."""
    return _load_yaml(settings.rubric_path)
