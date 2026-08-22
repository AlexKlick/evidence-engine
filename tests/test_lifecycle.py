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
from evidence_engine.store import repository as repo
from evidence_engine.store.models import Experiment

VERTICAL = "local-ai-tooling"


def seed_experiment(session_factory, settings, with_price: bool = True) -> str:
    """Heuristic fake run -> no price evidence; stamp one unless asked not to."""
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        if with_price:
            spec = dict(experiment.spec or {})
            guardrail = dict(spec.get("economics_guardrail") or {})
            guardrail.update(
                price_monthly=15, cac_ceiling=76.5,
                price_provenance="derived:median(n=1,p25=15,p75=15)",
            )
            spec["economics_guardrail"] = guardrail
            spec["maximum_spend"] = 76.5
            experiment.spec = spec  # reassign: JSON column change tracking
            session.commit()
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


def test_start_refuses_when_spec_has_no_price(
    fake_adapters, session_factory, settings
) -> None:
    """Hard gate: no price on the guardrail -> no spend, name the fix."""
    experiment_id = seed_experiment(session_factory, settings, with_price=False)
    with (
        session_factory() as session,
        pytest.raises(ValueError, match="--price"),
    ):
        start_experiment(session, experiment_id)
    with session_factory() as session:  # nothing transitioned on refusal
        refreshed = session.get(Experiment, experiment_id)
        assert refreshed is not None
        assert refreshed.status == "draft"


def test_lifecycle_unknown_experiment_raises(session_factory) -> None:
    with session_factory() as session:
        with pytest.raises(KeyError):
            start_experiment(session, "exp_missing")
        with pytest.raises(KeyError):
            stop_experiment(session, "exp_missing")


# -- codex adversarial review (gpt-5.6-sol): economics must be valid, not just present


@pytest.mark.parametrize(
    "price",
    [0, -5, float("nan"), float("inf")],
)
def test_start_rejects_nonfinite_or_nonpositive_price(
    session_factory, settings, price: float
) -> None:
    experiment_id = seed_experiment(session_factory, settings, with_price=False)
    with session_factory() as session:
        row = session.get(repo.Experiment, experiment_id)
        spec = dict(row.spec or {})
        guardrail = dict(spec.get("economics_guardrail") or {})
        guardrail["price_monthly"] = price
        spec["economics_guardrail"] = guardrail
        row.spec = spec
        session.commit()
        with pytest.raises(ValueError, match="finite positive"):
            start_experiment(session, experiment_id)


def test_start_rejects_inconsistent_cap(session_factory, settings) -> None:
    """cap/maximum_spend must match price x margin x months — a hand-edited or
    drifting spec must not start with unrelated economics."""
    experiment_id = seed_experiment(session_factory, settings)
    with session_factory() as session:
        row = session.get(repo.Experiment, experiment_id)
        spec = dict(row.spec or {})
        guardrail = dict(spec["economics_guardrail"])
        guardrail["cac_ceiling"] = 999.0
        spec["economics_guardrail"] = guardrail
        spec["maximum_spend"] = 999.0
        row.spec = spec
        session.commit()
        with pytest.raises(ValueError, match="cap"):
            start_experiment(session, experiment_id)


def test_start_rejects_price_without_provenance(session_factory, settings) -> None:
    """The pre-2026-08-22 shape: an unsourced 99.0 that passes `is not None`.

    474 specs were drafted with the illustrative $99 and a $504.90 cap before
    the pricing module reached draft_experiment_spec. Presence alone let every
    one of them start, which is the exact failure the anti-default guarantee
    is for.
    """
    experiment_id = seed_experiment(session_factory, settings)
    with session_factory() as session:
        row = session.get(repo.Experiment, experiment_id)
        spec = dict(row.spec or {})
        guardrail = dict(spec["economics_guardrail"])
        guardrail["price_monthly"] = 99.0
        guardrail["price_provenance"] = None
        guardrail["cac_ceiling"] = 504.9  # internally consistent: 99 x 0.85 x 6
        spec["economics_guardrail"] = guardrail
        spec["maximum_spend"] = 504.9
        row.spec = spec
        session.commit()
        with pytest.raises(ValueError, match="no price_provenance"):
            start_experiment(session, experiment_id)
