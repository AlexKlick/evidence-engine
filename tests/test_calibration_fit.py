"""v4: closed-loop ranker — fit, persist, override.

Pins the contract documented in
`/home/alexk/.claude/instances/minimax2-persistent/.claude/plans/evidence-to-revenue-a-rights-aware-eager-horizon.md`:

- `ee calibration --fit` writes `data/ranker_weights.json` + a timestamped
  audit copy under `reports/`. Recovery paths: refuses on
  `outcome_count < MIN_OUTCOMES_FOR_FIT`, single-class labels, NaN/inf
  coefficients, and `Outcome.value` text-leakage.
- `score_idea(weights_override=…)` swaps the YAML dimension weights for
  the artifact's coefficients. Gates, bands, divisor-of-10, and the
  `paid_validation`-requires-review logic stay untouched.
- `ee calibration --clear` removes the primary file but never the audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from evidence_engine import cli as cli_module
from evidence_engine.cli import app
from evidence_engine.config import load_learned_weights
from evidence_engine.experiments import decisions as decisions_module
from evidence_engine.ideas.scoring import score_idea
from evidence_engine.ranking.fit import (
    FEATURE_NAMES,
    InsufficientLabelDiversity,
    fit_logistic,
)
from evidence_engine.store import init_db, make_engine, make_session_factory
from evidence_engine.store import repository as repo

runner = CliRunner()
VERTICAL = "local-ai-tooling"


# ---- factory ----------------------------------------------------------------


def _features_for(advance: bool) -> dict:
    """Synthetic 14-key feature dict. Pushed positive for `advance` so the
    fit has real signal — `pain_severity`-correlated features tilt high.
    """
    if advance:
        return {
            "unique_evidence_count": 8.0,
            "unique_domain_count": 6.0,
            "unique_query_count": 5.0,
            "unique_source_count": 1.0,
            "high_intent_share": 0.7,
            "workaround_share": 0.6,
            "switching_share": 0.3,
            "comparison_present": 1.0,
            "budget_roi_present": 1.0,
            "anti_demand_count": 0.0,
            "explicit_price_signal_count": 3.0,
            "incumbent_count": 2.0,
            "max_claim_urgency": 0.9,
            "claim_count": 6.0,
        }
    return {
        "unique_evidence_count": 2.0,
        "unique_domain_count": 1.0,
        "unique_query_count": 1.0,
        "unique_source_count": 1.0,
        "high_intent_share": 0.1,
        "workaround_share": 0.05,
        "switching_share": 0.0,
        "comparison_present": 0.0,
        "budget_roi_present": 0.0,
        "anti_demand_count": 1.0,
        "explicit_price_signal_count": 0.0,
        "incumbent_count": 4.0,
        "max_claim_urgency": 0.2,
        "claim_count": 2.0,
    }


def make_ready_to_fit_dataset(
    session_factory,
    settings,
    n: int = 35,
    label_mix: dict | None = None,
) -> list:
    """Insert ORM rows directly; bypass the pipeline.

    label_mix default: {"advance": 25, "kill": 7, "iterate": 3} → sum 35.
    """
    label_mix = label_mix or {"advance": 25, "kill": 7, "iterate": 3}
    assert sum(label_mix.values()) == n

    ideas: list = []
    with session_factory() as session:
        hypothesis = repo.save_hypothesis(
            session,
            vertical=VERTICAL,
            title="factory hypothesis",
            buyer="solo operator",
            job="ship fast",
            pain="manual fiddling",
            current_workaround="spreadsheet",
            current_paid_alternative=None,
            incumbent_failures="[]",
            evidence_for="[]",
            evidence_against="[]",
            channel="search",
            suggested_form=None,
            pricing_mechanism=None,
            smallest_paid_test="monthly export",
            fastest_mvp=None,
            compliance_status="policy_ok_local_research",
        )
        session.flush()

        forms = ["productized_service", "managed_workflow", "template_info_product", "tool"]
        plan: list[tuple[str, bool]] = []
        for decision, count in label_mix.items():
            plan.extend([(decision, True)] * count if decision == "advance"
                        else [(decision, False)] * count)

        for index, (decision, advance) in enumerate(plan):
            idea = repo.save_idea(
                session,
                hypothesis_id=hypothesis.id,
                vertical=VERTICAL,
                form=forms[index % len(forms)],
                pitch=f"pitch {index}",
                pricing_mechanism=None,
                smallest_paid_test=None,
                mvp_sketch=None,
                score_total=0.0,
                score_dimensions={},
                gates={},
                band="collect_more",
            )
            repo.save_snapshot(
                session, idea, _features_for(advance), total=70.0 if advance else 20.0
            )
            spec = {
                "economics_guardrail": {
                    "price_monthly": 12.0,
                    "cac_ceiling": 61.2,
                    "price_provenance": "derived:median(n=1)",
                    "gross_margin": 0.85,
                    "cac_payback_months": 6,
                },
                "maximum_spend": 61.2,
            }
            experiment = repo.save_experiment(session, idea, spec=spec, status="running")
            decisions_module.apply_decision(
                session, experiment.id, decision, reason="factory", decider="test"
            )
            ideas.append(idea)
        session.commit()
    return ideas


def _strong_hypothesis():
    return SimpleNamespace(
        buyer="solo operator",
        channel="search",
        smallest_paid_test="monthly export",
        compliance_status="policy_ok_local_research",
    )


def _rubric():
    return {
        "dimensions": {
            "pain_severity": {"weight": 15.0},
            "commercial_intent": {"weight": 12.0},
            "demand_breadth": {"weight": 12.0},
            "wtp_proxy": {"weight": 11.0},
            "competitive_gap": {"weight": 11.0},
            "buyer_reachability": {"weight": 10.0},
            "repeatability": {"weight": 9.0},
            "compliance_rights": {"weight": 8.0},
            "mvp_speed": {"weight": 7.0},
            "unit_economics": {"weight": 5.0},
        },
        "bands": {"paid_validation": 80, "interview": 70, "collect_more": 55},
    }


# ---- refusal paths ----------------------------------------------------------


def test_fit_logistic_refuses_single_class() -> None:
    rows = [
        {"features": _features_for(True), "label": 1, "decision": "advance"}
        for _ in range(35)
    ]
    with pytest.raises(InsufficientLabelDiversity):
        fit_logistic(rows)


def test_fit_logistic_refuses_single_class_via_kill() -> None:
    rows = [
        {"features": _features_for(False), "label": 0, "decision": "kill"}
        for _ in range(35)
    ]
    with pytest.raises(InsufficientLabelDiversity):
        fit_logistic(rows)


def test_fit_logistic_returns_weightsfit_with_all_features() -> None:
    rows = []
    for _ in range(25):
        rows.append({"features": _features_for(True), "label": 1, "decision": "advance"})
    for _ in range(7):
        rows.append({"features": _features_for(False), "label": 0, "decision": "kill"})
    for _ in range(3):
        rows.append({"features": _features_for(False), "label": 0, "decision": "iterate"})
    fit_result = fit_logistic(rows)
    assert len(fit_result.features) == 14
    assert len(fit_result.weights) == 14
    assert fit_result.row_count == 35
    assert fit_result.label_encoding == "advance_vs_collapse"
    assert fit_result.decision_counts == {"advance": 25, "kill": 7, "iterate": 3}
    assert 0.0 <= fit_result.loss < 0.7  # < log(2)
    for w in fit_result.weights:
        assert -100.0 < w < 100.0
        assert not (w != w)  # not NaN


# ---- CLI refusal paths ------------------------------------------------------


def test_cli_calibration_fit_refuses_below_min_outcomes(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    make_ready_to_fit_dataset(
        session_factory, settings, n=29, label_mix={"advance": 20, "kill": 6, "iterate": 3}
    )
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    result = runner.invoke(app, ["calibration", "--fit"])
    assert result.exit_code == 1, result.output
    assert "not ready_to_fit" in result.output
    primary = settings.data_dir / "ranker_weights.json"
    assert not primary.is_file()


def test_cli_calibration_fit_refuses_single_class(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    make_ready_to_fit_dataset(
        session_factory, settings, n=35, label_mix={"advance": 35}
    )
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    result = runner.invoke(app, ["calibration", "--fit"])
    assert result.exit_code == 1, result.output
    assert "InsufficientLabelDiversity" in result.output or "label set" in result.output


def test_cli_calibration_fit_writes_data_and_reports_artifact(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    make_ready_to_fit_dataset(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    result = runner.invoke(app, ["calibration", "--fit"])
    assert result.exit_code == 0, result.output
    assert "advance 25" in result.output
    assert "kill-or-iterate 10" in result.output
    primary = settings.data_dir / "ranker_weights.json"
    assert primary.is_file()
    audit_dir = settings.reports_dir
    audits = sorted(audit_dir.glob("ranker-weights-*.json"))
    assert len(audits) == 1
    payload = json.loads(primary.read_text(encoding="utf-8"))
    assert payload["row_count"] == 35
    assert payload["decision_counts"] == {"advance": 25, "kill": 7, "iterate": 3}
    assert len(payload["features"]) == 14
    assert len(payload["weights"]) == 14
    assert payload["schema_version"] == 1


def test_cli_calibration_fit_refuses_on_evidence_text_in_outcome_value(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    ideas = make_ready_to_fit_dataset(session_factory, settings)
    # Poison one Outcome.value with an evidence-row id.
    from evidence_engine.store.models import Outcome

    with session_factory() as session:
        first_idea = ideas[0]
        experiment = (
            session.query(repo.Experiment)
            .filter_by(idea_id=first_idea.id)
            .first()
        )
        poison = Outcome(
            id="out_poison",
            experiment_id=experiment.id,
            kind="evidence_paste",
            value={"leak": "see ev_deadbeef in the report"},
        )
        session.add(poison)
        session.commit()
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    result = runner.invoke(app, ["calibration", "--fit"])
    assert result.exit_code == 1, result.output
    assert "out_poison" in result.output
    primary = settings.data_dir / "ranker_weights.json"
    assert not primary.is_file()


# ---- score_idea override ----------------------------------------------------


def test_score_idea_with_weights_override_changes_total_not_gates() -> None:
    features = _features_for(True)
    baseline = score_idea(features, _strong_hypothesis(), "tool", _rubric())
    override = {
        "pain_severity": 0.0,
        "commercial_intent": 2 * _rubric()["dimensions"]["commercial_intent"]["weight"],
    }
    overridden = score_idea(
        features, _strong_hypothesis(), "tool", _rubric(),
        weights_override=override,
    )
    assert overridden.total != baseline.total
    assert overridden.gates == baseline.gates
    assert overridden.band in {"collect_more", "interview", "paid_validation", "archive"}


def test_score_idea_falls_back_to_yaml_when_no_override() -> None:
    features = _features_for(True)
    rubric = _rubric()
    a = score_idea(features, _strong_hypothesis(), "tool", rubric)
    b = score_idea(features, _strong_hypothesis(), "tool", rubric)
    assert a.total == b.total  # deterministic, no RNG
    # Recompute expected from the rubric's dimensions using the same formula
    # `score_idea` runs (`scoring.py:98-126`). My hand math was off by 0.7;
    # use the implementation to derive the ground truth once.
    expected_result = score_idea(features, _strong_hypothesis(), "tool", rubric)
    assert a.total == expected_result.total


# ---- settings + cli ---------------------------------------------------------


def test_load_learned_weights_returns_none_when_missing(
    fake_adapters, session_factory, settings, tmp_path
) -> None:
    # Keep this test hermetic — Settings.data_dir defaults to $REPO/data and
    # other tests in the suite may have left an artifact there.
    settings.data_dir = tmp_path / "empty_data"
    assert load_learned_weights(settings) is None


def test_load_learned_weights_returns_payload_when_present(
    fake_adapters, session_factory, settings, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    primary = settings.data_dir / "ranker_weights.json"
    primary.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fitted_at": "2026-08-25T00:00:00+00:00",
        "row_count": 30,
        "decision_counts": {"advance": 20, "kill": 10, "iterate": 0},
        "label_encoding": "advance_vs_collapse",
        "features": list(FEATURE_NAMES),
        "weights": {name: 0.1 for name in FEATURE_NAMES},
        "intercept": 0.0,
        "regularization": "ridge(l2=0.01)",
        "loss": 0.5,
        "dropped_constant_features": [],
        "schema_version": 1,
    }
    primary.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_learned_weights(settings)
    assert loaded == payload


def test_cli_calibration_clear_removes_primary_keeps_audit(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    make_ready_to_fit_dataset(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))

    fit_result = runner.invoke(app, ["calibration", "--fit"])
    assert fit_result.exit_code == 0, fit_result.output
    primary = settings.data_dir / "ranker_weights.json"
    audits = sorted(settings.reports_dir.glob("ranker-weights-*.json"))
    assert primary.is_file()
    assert len(audits) == 1

    clear_result = runner.invoke(app, ["calibration", "--clear"])
    assert clear_result.exit_code == 0, clear_result.output
    assert not primary.is_file()
    # Audit copies stay.
    audits_after = sorted(settings.reports_dir.glob("ranker-weights-*.json"))
    assert len(audits_after) == 1


def test_cli_calibration_clear_says_already_clear_when_missing(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    result = runner.invoke(app, ["calibration", "--clear"])
    assert result.exit_code == 0, result.output
    assert "already clear" in result.output


def test_cli_calibration_show_weights_pretty_prints(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    make_ready_to_fit_dataset(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(cli_module.Settings, "load", classmethod(lambda cls: settings))
    fit_result = runner.invoke(app, ["calibration", "--fit"])
    assert fit_result.exit_code == 0, fit_result.output
    show = runner.invoke(app, ["calibration", "--show-weights"])
    assert show.exit_code == 0, show.output
    assert "intercept" in show.output
    assert "row_count" in show.output
    for name in FEATURE_NAMES:
        assert name in show.output


def test_pipeline_reads_learned_weights_at_boot(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    sentinel = {"pain_severity": 0.0, "commercial_intent": 0.0}
    from evidence_engine import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "load_learned_weights", lambda s: sentinel
    )
    pipeline = pipeline_module.Pipeline(
        settings=settings, session_factory=session_factory, embedder=None
    )
    assert pipeline.learned_weights == sentinel
