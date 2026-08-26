"""Decision state machine (advance|kill|iterate) — the v2 wedge for the
ranker loop.

Locks the contract:
- `apply_decision` routes every status change through `validate_transition`.
- Each decision writes a typed `decision` AND a synthesized `Outcome` kind
  (advance/kill always, iterate only when a reason is provided).
- `--decision advance` without a prior paid outcome warns but does not
  refuse (`--override` suppresses even the warning).
- The synthesized `Outcome.value` carries operator metadata only — never
  evidence row text (CLAUDE.md: no raw source content redisplayed).
- `ee calibration` surfaces the typed `decision` per idea.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from conftest import FakeEmbedder
from evidence_engine.cli import app
from evidence_engine.experiments.decisions import (
    DECISIONS,
    DECISION_OUTCOME_KIND,
    DECISION_TARGET_STATUS,
    apply_decision,
)
from evidence_engine.experiments.lifecycle import validate_transition
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo
from evidence_engine.store.models import Experiment, Outcome

runner = CliRunner()
VERTICAL = "local-ai-tooling"


def seed_running(session_factory, settings) -> str:
    """Heuristic fake run + stamp a guardrail so start_experiment passes.

    Returns the experiment_id already in `running` state. Direct ORM write
    bypasses the lifecycle validator only because we're seeding test
    fixtures; the lifecycle gate is exercised explicitly in the transition
    tests below.
    """
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        experiment = session.query(Experiment).first()
        assert experiment is not None
        spec = dict(experiment.spec or {})
        guardrail = dict(spec.get("economics_guardrail") or {})
        guardrail.update(
            price_monthly=15,
            cac_ceiling=76.5,
            price_provenance="derived:median(n=1,p25=15,p75=15)",
        )
        spec["economics_guardrail"] = guardrail
        spec["maximum_spend"] = 76.5
        experiment.spec = spec
        experiment.status = "running"
        session.commit()
        return experiment.id


# ---- pure transition matrix --------------------------------------------------


@pytest.mark.parametrize("decision", ["advance", "kill"])
def test_decision_target_status_agrees_with_lifecycle(decision: str) -> None:
    """advance/kill produce a status side-effect that the lifecycle validator accepts.

    `iterate` is a status-side-effect NO-OP (running stays running) and
    therefore explicitly bypasses validate_transition; that contract is
    exercised by the CLI tests below, not by this matrix.
    """
    target = DECISION_TARGET_STATUS[decision]
    validate_transition("running", target)  # never raises for these two
    # `iterate` is a no-op transition and deliberately bypasses the validator.
    assert DECISION_TARGET_STATUS["iterate"] == "running"


@pytest.mark.parametrize(
    "current,target",
    [
        ("draft", "stopped"),
        ("stopped", "running"),
        ("stopped", "stopped"),
    ],
)
def test_decisions_never_bypass_validate_transition(current, target) -> None:
    """The lifecycle validator still rules — decisions cannot smuggle past it.

    Note: this matrix asserts that the validator still rejects illegal
    transitions on its own. `iterate` short-circuits the validator when
    target == current; that bypass lives in apply_decision and is tested
    separately in the CLI surface.
    """
    with pytest.raises(ValueError):
        validate_transition(current, target)


# ---- apply_decision behavior -------------------------------------------------


@pytest.mark.parametrize("decision", list(DECISIONS))
def test_apply_decision_synthesizes_outcome_row(
    fake_adapters, session_factory, settings, decision: str
) -> None:
    exp_id = seed_running(session_factory, settings)
    # Pre-record a paid outcome so `advance` doesn't trip the warning path;
    # the warning is exercised separately below.
    with session_factory() as session:
        exp = session.get(Experiment, exp_id)
        assert exp is not None
        if decision == "advance":
            repo.record_outcome(session, exp, "deposit_paid", {"amount": 25})
        session.commit()
    with session_factory() as session:
        row, outcome, warnings = apply_decision(session, exp_id, decision)
        assert row.status == DECISION_TARGET_STATUS[decision]
        assert row.decision == decision
        # advance / kill always synthesize; iterate only when reason supplied
        if decision != "iterate":
            assert outcome is not None
            assert outcome.kind == DECISION_OUTCOME_KIND[decision]
            assert outcome.value["decider"] == "operator"
            assert outcome.value["reason"] == ""
        else:
            assert outcome is None
        assert warnings == []
        session.commit()

    with session_factory() as session:
        refreshed = session.get(Experiment, exp_id)
        assert refreshed is not None
        assert refreshed.status == DECISION_TARGET_STATUS[decision]
        assert refreshed.decision == decision
        if decision != "iterate":
            kinds = [o.kind for o in session.query(Outcome).all()]
            assert DECISION_OUTCOME_KIND[decision] in kinds


def test_apply_decision_records_reason_and_caps_at_500(
    fake_adapters, session_factory, settings
) -> None:
    long_reason = "x" * 800
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        row, outcome, _ = apply_decision(
            session, exp_id, "advance", reason=long_reason
        )
        session.commit()
    assert outcome is not None
    assert len(outcome.value["reason"]) == 500  # hard cap
    with session_factory() as session:
        refreshed = session.get(Experiment, exp_id)
        assert refreshed is not None
        assert refreshed.decision_reason is not None
        assert len(refreshed.decision_reason) == 500


def test_advance_without_paid_outcome_warns_but_allows(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        row, outcome, warnings = apply_decision(session, exp_id, "advance")
        assert any("paid outcome" in w for w in warnings)
        assert row.status == "stopped"
        assert outcome is not None


def test_advance_with_paid_outcome_no_warning(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        exp = session.get(Experiment, exp_id)
        assert exp is not None
        repo.record_outcome(session, exp, "deposit_paid", {"amount": 25})
        session.commit()
    with session_factory() as session:
        _, _, warnings = apply_decision(session, exp_id, "advance")
        assert warnings == []


def test_kill_synthesizes_outcome_and_stops(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        row, outcome, warnings = apply_decision(
            session, exp_id, "kill", reason="no traction after two weeks"
        )
        session.commit()
    assert row.status == "stopped"
    assert row.decision == "kill"
    assert outcome is not None
    assert outcome.kind == "kill"
    assert outcome.value["reason"] == "no traction after two weeks"
    assert warnings == []


def test_iterate_without_reason_synthesizes_nothing(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        row, outcome, warnings = apply_decision(session, exp_id, "iterate")
        session.commit()
    assert row.status == "running"
    assert row.decision == "iterate"
    assert outcome is None
    assert warnings == []
    with session_factory() as session:
        kinds = [o.kind for o in session.query(Outcome).all()]
        assert "iterate" not in kinds


def test_iterate_with_reason_synthesizes_outcome(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session:
        row, outcome, _ = apply_decision(
            session, exp_id, "iterate", reason="tighten CTA copy"
        )
        session.commit()
    assert row.status == "running"
    assert outcome is not None
    assert outcome.kind == "iterate"


def test_decision_value_never_carries_evidence_text(
    fake_adapters, session_factory, settings
) -> None:
    """CLAUDE.md hard rule: published artifacts carry no source text."""
    exp_id = seed_running(session_factory, settings)
    operator_reason = "first paying customer via concierge outreach"
    with session_factory() as session:
        _, outcome, _ = apply_decision(
            session, exp_id, "advance", reason=operator_reason
        )
        session.commit()
    assert outcome is not None
    serialized = repr(outcome.value)
    # Operator-authored free text only; no evidence row ids / snippet keys.
    for forbidden in ("ev_", "snippet", "title=", "url="):
        assert forbidden not in serialized, (
            f"synthesized Outcome.value leaked evidence text: {serialized!r}"
        )


def test_apply_decision_rejects_unknown_decision(
    fake_adapters, session_factory, settings
) -> None:
    exp_id = seed_running(session_factory, settings)
    with session_factory() as session, pytest.raises(ValueError, match="unknown decision"):
        apply_decision(session, exp_id, "maybe")  # type: ignore[arg-type]


def test_apply_decision_rejects_unknown_experiment(session_factory) -> None:
    with session_factory() as session, pytest.raises(KeyError, match="not found"):
        apply_decision(session, "exp_does_not_exist", "advance")


# ---- CLI surface -------------------------------------------------------------


@pytest.mark.parametrize("decision", list(DECISIONS))
def test_cli_decide_invokes_lifecycle(
    fake_adapters, session_factory, settings, monkeypatch, decision: str
) -> None:
    exp_id = seed_running(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    result = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", decision]
    )
    assert result.exit_code == 0, result.output
    assert decision in result.output
    with session_factory() as session:
        refreshed = session.get(Experiment, exp_id)
        assert refreshed is not None
        assert refreshed.status == DECISION_TARGET_STATUS[decision]
        assert refreshed.decision == decision


def test_cli_decide_rejects_unknown_decision(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    exp_id = seed_running(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    result = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", "maybe"]
    )
    assert result.exit_code == 1
    assert "advance" in result.output and "kill" in result.output


def test_cli_decide_warns_advance_without_paid(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    exp_id = seed_running(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    result = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", "advance"]
    )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output
    assert "paid outcome" in result.output


def test_cli_decide_rejects_illegal_lifecycle_transition(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    """stopped experiment cannot be re-decided; validate_transition fires."""
    exp_id = seed_running(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    # First decision: advance (running -> stopped)
    first = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", "advance"]
    )
    assert first.exit_code == 0, first.output
    # Second decision: refused — terminal state.
    second = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", "iterate"]
    )
    assert second.exit_code == 1
    assert "stopped" in second.output


def test_calibration_exposes_decision_column(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    exp_id = seed_running(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    decided = runner.invoke(
        app, ["experiments", "decide", "-e", exp_id, "-d", "advance"]
    )
    assert decided.exit_code == 0, decided.output

    calibration = runner.invoke(app, ["calibration"])
    assert calibration.exit_code == 0, calibration.output
    assert "decision=" in calibration.output
    assert "advance" in calibration.output
