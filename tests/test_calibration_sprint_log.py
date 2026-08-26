"""v6: `ee calibration --sprint-log` — append a timestamped block to
gate-logs/sprint.md. Provenance: each line names its source.
"""

from __future__ import annotations

from typer.testing import CliRunner

from conftest import FakeEmbedder
from evidence_engine import cli as cli_module
from evidence_engine.cli import app
from evidence_engine.experiments import decisions as decisions_module
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo

runner = CliRunner()
VERTICAL = "local-ai-tooling"


def _seed_one_running(session_factory, settings) -> None:
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        exp = session.query(repo.Experiment).first()
        assert exp is not None
        exp.status = "running"
        session.commit()
        decisions_module.apply_decision(
            session, exp.id, "advance", reason="smoke", decider="test"
        )
        session.commit()


def test_sprint_log_creates_file_with_header_when_missing(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    settings.gate_logs_dir = tmp_path / "gate-logs"
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    result = runner.invoke(app, ["calibration", "--sprint-log"])
    assert result.exit_code == 0, result.output
    target = settings.gate_logs_dir / "sprint.md"
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Sprint log\n")
    assert "## Sprint snapshot — " in text
    assert "ready_to_fit=False" in text


def test_sprint_log_appends_to_existing_file(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    settings.gate_logs_dir = tmp_path / "gate-logs"
    settings.gate_logs_dir.mkdir(parents=True, exist_ok=True)
    target = settings.gate_logs_dir / "sprint.md"
    target.write_text("# Sprint log\n\nSENTINEL_LINE\n", encoding="utf-8")

    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    result = runner.invoke(app, ["calibration", "--sprint-log"])
    assert result.exit_code == 0, result.output
    text = target.read_text(encoding="utf-8")
    assert "SENTINEL_LINE" in text
    assert text.count("## Sprint snapshot — ") == 1


def test_sprint_log_block_carries_provenance_lines(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    settings.data_dir = tmp_path / "data"
    settings.reports_dir = tmp_path / "reports"
    settings.gate_logs_dir = tmp_path / "gate-logs"
    _seed_one_running(session_factory, settings)

    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    result = runner.invoke(app, ["calibration", "--sprint-log"])
    assert result.exit_code == 0, result.output
    text = (settings.gate_logs_dir / "sprint.md").read_text(encoding="utf-8")
    # Every provenance line surfaces its source.
    assert "source: `ee calibration` snapshot" in text
    assert "source: `data/ranker_weights.json`" in text
    assert "source: `ee calibration` per-vertical running counts" in text
    assert "source: `ee calibration` decision distribution" in text
    assert "advance=1 · kill=0 · iterate=0" in text
