"""v6: `ee calibration --show-weights --idea ID` — print header + table +
score diff vs YAML rubric for one idea.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evidence_engine import cli as cli_module
from evidence_engine.cli import app

runner = CliRunner()


@pytest.fixture
def primed_settings(fake_adapters, session_factory, settings, tmp_path, monkeypatch):
    """Seed the v4 fit fixture and hermeticise data/reports dirs."""
    from test_calibration_fit import make_ready_to_fit_dataset

    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    make_ready_to_fit_dataset(session_factory, settings)
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    # Run the fit so the artifact exists.
    fit_result = runner.invoke(app, ["calibration", "--fit"])
    assert fit_result.exit_code == 0, fit_result.output
    return settings


def test_show_weights_with_idea_filter_prints_header(
    primed_settings, session_factory
) -> None:
    from evidence_engine.store import repository as repo

    with session_factory() as session:
        idea_id = session.query(repo.ProductIdea).first().id
    result = runner.invoke(
        app, ["calibration", "--show-weights", "--idea", idea_id]
    )
    assert result.exit_code == 0, result.output
    assert f"filter: idea_id={idea_id}" in result.output
    assert "weights from" in result.output


def test_show_weights_idea_filter_score_diff_appears(
    primed_settings, session_factory
) -> None:
    from evidence_engine.store import repository as repo

    with session_factory() as session:
        idea_id = session.query(repo.ProductIdea).first().id
    result = runner.invoke(
        app, ["calibration", "--show-weights", "--idea", idea_id]
    )
    assert result.exit_code == 0, result.output
    assert "score_idea(pre-fit, yaml):" in result.output
    assert "score_idea(post-fit, learned):" in result.output
    assert "delta:" in result.output


def test_show_weights_without_idea_keeps_legacy_output(primed_settings) -> None:
    result = runner.invoke(app, ["calibration", "--show-weights"])
    assert result.exit_code == 0, result.output
    assert "filter:" not in result.output
    assert "intercept" in result.output


def test_idea_without_show_weights_errors(primed_settings) -> None:
    result = runner.invoke(app, ["calibration", "--idea", "idea_smoke"])
    assert result.exit_code == 1
    assert "--idea requires --show-weights" in result.output
