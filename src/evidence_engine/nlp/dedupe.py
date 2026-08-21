"""Three-level deduplication: exact hash -> near-duplicate -> semantic.

A naive counter turns viral repetition into fake demand; the pipeline counts
only independent evidence (originals), duplicates stay linked for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_engine.nlp.normalize import cosine


@dataclass
class DedupeResult:
    exact_pairs: list[tuple[str, str]] = field(default_factory=list)
    near_pairs: list[tuple[str, str]] = field(default_factory=list)
    semantic_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_pairs(self) -> list[tuple[str, str, str]]:
        pairs = [("exact", a, b) for a, b in self.exact_pairs]
        pairs += [("near", a, b) for a, b in self.near_pairs]
        pairs += [("semantic", a, b) for a, b in self.semantic_pairs]
        return pairs

    @property
    def duplicate_ids(self) -> set[str]:
        return {dup for dup, _ in self.exact_pairs + self.near_pairs + self.semantic_pairs}


def find_duplicates(
    events: list,
    semantic_threshold: float = 0.97,
) -> DedupeResult:
    """events: EvidenceEvent rows (id, content_hash, near_dup_key, embedding)."""
    result = DedupeResult()
    by_hash: dict[str, str] = {}
    by_near: dict[str, str] = {}
    originals: list[tuple[str, list[float]]] = []

    for event in events:
        if event.content_hash in by_hash:
            result.exact_pairs.append((event.id, by_hash[event.content_hash]))
            continue
        by_hash[event.content_hash] = event.id

        near = event.near_dup_key or ""
        if near and near in by_near:
            result.near_pairs.append((event.id, by_near[near]))
            continue
        if near:
            by_near[near] = event.id

        vector = event.embedding
        if vector:
            match_id = None
            best = 0.0
            for original_id, original_vec in originals:
                similarity = cosine(vector, original_vec)
                if similarity > best:
                    best, match_id = similarity, original_id
            if match_id is not None and best >= semantic_threshold:
                result.semantic_pairs.append((event.id, match_id))
                continue
            originals.append((event.id, vector))
        else:
            originals.append((event.id, []))

    return result
