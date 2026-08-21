"""Clustering with a dependency-free greedy-cosine default.

`density_cluster` uses UMAP + HDBSCAN (extras: uv sync --extra cluster) when
importable and the corpus is large enough; otherwise it degrades to greedy
cosine thresholds. Cluster labels combine top tokens + dominant intent —
buying-stage axes a pure topic model would miss.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from evidence_engine.nlp.normalize import cosine

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with", "how",
    "do", "i", "is", "are", "my", "your", "it", "that", "this", "be", "can",
    "you", "best", "get", "using", "use", "from", "at", "when", "what", "why",
    "not", "but", "if", "into", "than", "then", "will", "should", "does",
    "free", "vs", "com", "www", "https", "http",
}
_TOKEN = re.compile(r"[a-z][a-z0-9+-]{2,}")
_DENSITY_MIN_ITEMS = 40


@dataclass
class ClusterGroup:
    member_ids: list[str] = field(default_factory=list)
    label: str = ""


def top_tokens(titles: list[str], limit: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for title in titles:
        for token in _TOKEN.findall((title or "").lower()):
            if token not in _STOPWORDS:
                counts[token] += 1
    return [token for token, _ in counts.most_common(limit)]


def dominant_intent(label_sets: list[list[str]]) -> str:
    counts: Counter[str] = Counter()
    for labels in label_sets:
        counts.update(labels)
    return counts.most_common(1)[0][0] if counts else "unlabeled"


def make_label(titles: list[str], label_sets: list[list[str]]) -> str:
    tokens = top_tokens(titles)
    intent = dominant_intent(label_sets)
    return " ".join(tokens) + f" [{intent}]" if tokens else f"[{intent}]"


def greedy_cluster(
    items: list[tuple[str, list[float]]], threshold: float
) -> list[list[str]]:
    """Order-preserving greedy clustering by cosine to running centroid."""
    clusters: list[tuple[list[str], list[float]]] = []
    for item_id, vector in items:
        best_index, best_score = -1, 0.0
        for index, (_, centroid) in enumerate(clusters):
            score = cosine(vector, centroid) if vector and centroid else 0.0
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0 and best_score >= threshold:
            members, centroid = clusters[best_index]
            members.append(item_id)
            if vector and centroid:
                clusters[best_index] = (
                    members,
                    [c + v for c, v in zip(centroid, vector, strict=True)],
                )
        else:
            clusters.append(([item_id], list(vector)))
    return [members for members, _ in clusters]


def density_cluster(
    items: list[tuple[str, list[float]]], threshold: float
) -> list[list[str]]:
    """UMAP+HDBSCAN when available and warranted; greedy otherwise.

    Noise points become singletons (cluster_evidence drops groups < 2 anyway).
    Only runs when every item has a vector, so label indices stay aligned.
    """
    vectors = [vector for _, vector in items]
    if len(vectors) >= _DENSITY_MIN_ITEMS and all(vectors):
        try:
            import hdbscan  # type: ignore[import-not-found]
            import umap  # type: ignore[import-not-found]

            ids = [item_id for item_id, _ in items]
            reduced = umap.UMAP(
                n_components=min(10, max(2, len(vectors) // 10)),
                metric="cosine",
                random_state=42,
            ).fit_transform(vectors)
            labels = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(reduced)
            groups: dict[int, list[str]] = {}
            for item_id, label in zip(ids, labels, strict=True):
                groups.setdefault(int(label), []).append(item_id)
            noise = groups.pop(-1, [])
            result = [members for _, members in sorted(groups.items())]
            result.extend([member] for member in noise)
            return result
        except ImportError:
            pass
    return greedy_cluster(items, threshold)


def cluster_evidence(
    evidence_rows: list,
    threshold: float,
    use_density: bool = True,
) -> list[ClusterGroup]:
    """Cluster embedded, non-duplicate evidence; label from titles + intents."""
    items = [
        (row.id, row.embedding or [])
        for row in evidence_rows
        if not row.is_duplicate_of
    ]
    groups = (
        density_cluster(items, threshold) if use_density else greedy_cluster(items, threshold)
    )
    by_id = {row.id: row for row in evidence_rows}
    result: list[ClusterGroup] = []
    for member_ids in groups:
        if len(member_ids) < 2:
            continue  # singletons are noise at scaffold scale
        rows = [by_id[mid] for mid in member_ids if mid in by_id]
        label = make_label(
            [f"{r.title} {r.snippet}" for r in rows],
            [list(r.intent_labels or []) for r in rows],
        )
        result.append(ClusterGroup(member_ids=member_ids, label=label))
    result.sort(key=lambda group: len(group.member_ids), reverse=True)
    return result
