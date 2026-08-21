# ADR-0003: Local model substitution for the NLP stack

Date: 2026-08-21
Status: accepted

## Context

The founding doc's reference stack: SentenceTransformers embeddings,
UMAP + HDBSCAN clustering, BERTopic-style topic representations, and a
provider-agnostic LLM for structured extraction. The workstation provides
loopback OpenAI-compatible lanes instead (text-main `:18000` Qwen3.8-27B,
embeddings `:6900` Qwen3-Embedding-0.6B 1024-dim) and GPU headroom is limited
(GPU 0 ~fully occupied by text-main; GPU 1 has ~5.7 GB spare; embedding lane
is CPU-only).

## Decision

1. Embeddings come from the `:6900` lane via an OpenAI-compatible client
   (`nlp/embeddings.py`), not a bundled SentenceTransformers model. 1024-dim.
2. Pain-claim extraction uses the `:18000` lane with a strict JSON contract,
   fence-stripping parser and one retry; a deterministic heuristic extractor
   is the always-available fallback (`nlp/pain_claims.py`). Quality of the
   local model is a prior to calibrate, not a blocker.
3. Clustering ships a dependency-free greedy cosine-threshold fallback
   (`nlp/clustering.py`). UMAP + HDBSCAN are an optional extra
   (`uv sync --extra cluster`) selected automatically when importable.
4. `local_inference` vs `external_inference` are separate rights on each
   source: loopback models are governed by the former; switching to any
   hosted API requires the latter to be true.

## Consequences

- Zero new model hosting cost; pipeline degrades gracefully when lanes are
  down (doctor reports, heuristics take over, report notes the mode).
- Embedding-model swaps change vector space; cluster refreshes should re-run
  end-to-end. The embedding model name is stored per run for this reason.
- Native pgvector ANN search and BERTopic-style topic labels remain future
  upgrades; the schema and interfaces do not block them.
