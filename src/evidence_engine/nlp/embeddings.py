"""Embedding client for the loopback lane (:6900, OpenAI-compatible)."""

from __future__ import annotations

import httpx

from evidence_engine.config import Settings
from evidence_engine.logging_setup import get_logger

logger = get_logger("nlp.embeddings")


class EmbeddingUnavailableError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def health(self, timeout: float = 4.0) -> bool:
        try:
            response = httpx.get(
                f"{self._settings.embeddings_base_url.rstrip('/')}/models",
                timeout=timeout,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; raises EmbeddingUnavailableError."""
        if not texts:
            return []
        try:
            response = httpx.post(
                f"{self._settings.embeddings_base_url.rstrip('/')}/embeddings",
                json={"input": texts, "model": self._settings.embeddings_model},
                timeout=self._settings.embeddings_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload.get("data", []), key=lambda d: d.get("index", 0))
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(texts):
                raise EmbeddingUnavailableError(
                    f"embedding count mismatch: {len(vectors)} != {len(texts)}"
                )
            return vectors
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(str(exc)) from exc

    @property
    def model_name(self) -> str:
        return self._settings.embeddings_model
