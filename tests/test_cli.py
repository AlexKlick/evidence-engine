"""CLI smoke tests (config-only commands; no db/network)."""

from __future__ import annotations

from typer.testing import CliRunner

from evidence_engine.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "evidence-engine" in result.output


def test_policy_show_lists_sources_and_reasons() -> None:
    result = runner.invoke(app, ["policy", "show"])
    assert result.exit_code == 0
    assert "searxng" in result.output
    assert "ENABLED" in result.output
    assert "reddit" in result.output
    assert "separate agreement" in result.output


def test_policy_lint_ok_on_repo_config() -> None:
    result = runner.invoke(app, ["policy", "lint"])
    assert result.exit_code == 0
    assert "ok" in result.output


def test_verticals_lists_seed_slugs() -> None:
    result = runner.invoke(app, ["verticals"])
    assert result.exit_code == 0
    for slug in ("local-ai-tooling", "solo-dev-saas", "prediction-market-research"):
        assert slug in result.output


def test_help_mentions_core_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "collect", "pipeline", "ideas", "bootstrap", "deletions"):
        assert command in result.output
