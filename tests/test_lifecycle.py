"""Experiment lifecycle: validated draft->running->stopped transitions."""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder
from evidence_engine.experiments.lifecycle import (
    start_experiment,
    stop_experiment,
    validate_transition,
)
from evidence_engine.pipeline import Pipeline
from evidence_engine.store.models import Experiment

VERTICAL = "local-ai-tooling"


def seed_experiment(session_factory, settings) -> str:
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        return experiment.id


def test_validate_transition_matrix() -> None:
    validate_transition("draft", "running")
    validate_transition("running", "stopped")
    for current, target in (
        ("draft", "stopped"),
        ("draft", "draft"),
        ("running", "running"),
        ("stopped", "running"),
        ("stopped", "stopped"),
    ):
        with pytest.raises(ValueError):
            validate_transition(current, target)
    with pytest.raises(ValueError, match="legal transitions: running"):
        validate_transition("draft", "stopped")
    with pytest.raises(ValueError, match="unknown status"):
        validate_transition("draft", "paused")


def test_start_then_stop_roundtrip(
    fake_adapters, session_factory, settings
) -> None:
    experiment_id = seed_experiment(session_factory, settings)
    with session_factory() as session:
        row = start_experiment(session, experiment_id)
        assert row.status == "running"
        session.commit()
    with session_factory() as session:
        with pytest.raises(ValueError):
            start_experiment(session, experiment_id)  # already running
        row = stop_experiment(session, experiment_id, decision="advance")
        assert row.status == "stopped"
        assert row.decision == "advance"
        session.commit()
    with session_factory() as session:
        with pytest.raises(ValueError):
            stop_experiment(session, experiment_id)  # terminal
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "stopped"


def test_stop_of_draft_raises(fake_adapters, session_factory, settings) -> None:
    experiment_id = seed_experiment(session_factory, settings)
    with (
        session_factory() as session,
        pytest.raises(ValueError, match="legal transitions: running"),
    ):
        stop_experiment(session, experiment_id)


def test_lifecycle_unknown_experiment_raises(session_factory) -> None:
    with session_factory() as session:
        with pytest.raises(KeyError):
            start_experiment(session, "exp_missing")
        with pytest.raises(KeyError):
            stop_experiment(session, "exp_missing")
