"""Ideas package: hypothesis compiler, product forms, scoring."""

from evidence_engine.ideas.hypothesis import HypothesisDraft, build_hypothesis
from evidence_engine.ideas.scoring import ScoreResult, aggregate_features, score_idea

__all__ = [
    "HypothesisDraft",
    "ScoreResult",
    "aggregate_features",
    "build_hypothesis",
    "score_idea",
]
