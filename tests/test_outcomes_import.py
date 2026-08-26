"""v6: `ee outcomes import --csv` — batch outcome ingest with the
same leakage gate as `ee calibration --fit`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from conftest import FakeEmbedder
from evidence_engine import cli as cli_module
from evidence_engine.cli import app
from evidence_engine.experiments import decisions as decisions_module
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo

runner = CliRunner()
VERTICAL = "local-ai-tooling"


def _seed_running_experiment(session_factory, settings) -> str:
    """Seed a pipeline run + force one experiment into `running` and one
    into `stopped` (so `--status` transitions have somewhere legal to go).
    Returns the running experiment_id.
    """
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        experiments = session.query(repo.Experiment).all()
        assert experiments, "pipeline run produced no experiments"
        running = experiments[0]
        running.status = "running"
        # Use `apply_decision` to leave the row in a known state.
        decisions_module.apply_decision(
            session, running.id, "iterate", reason="seed", decider="test"
        )
        session.commit()
        return running.id


def _seed_two_running(session_factory, settings) -> tuple[str, str]:
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        exps = session.query(repo.Experiment).order_by(repo.Experiment.created_at).all()
        assert len(exps) >= 2
        exps[0].status = "running"
        exps[1].status = "running"
        session.commit()
        return exps[0].id, exps[1].id


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    import csv as csv_mod

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_mod.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return path


# ---- happy path -----------------------------------------------------------


def test_import_happy_path_records_three_outcomes(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            [exp_id, "deposit_paid", '{"amount": 49}'],
            [exp_id, "interview_completed", '{"duration_min": 30}'],
            [exp_id, "paid_pilot_started", '{"weeks": 2}'],
        ],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 0, result.output
    assert "imported 3 · skipped 0" in result.output
    with session_factory() as session:
        rows = (
            session.query(repo.Outcome)
            .filter_by(experiment_id=exp_id)
            .all()
        )
        # 1 from the iterate seed + 3 imported = 4
        assert len(rows) == 4
        kinds = {r.kind for r in rows}
        assert kinds == {"deposit_paid", "interview_completed", "paid_pilot_started", "iterate"}


# ---- refusal paths -------------------------------------------------------


def test_import_refuses_unknown_kind_with_row_number(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            [exp_id, "deposit_paid", '{"amount": 1}'],
            [exp_id, "not_a_kind", "{}"],
        ],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "refuse row 2" in result.output
    assert "unknown kind" in result.output


def test_import_refuses_malformed_value_json(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            [exp_id, "deposit_paid", '{"amount": 1}'],
            [exp_id, "deposit_paid", "{not json"],
        ],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "refuse row 2" in result.output
    assert "malformed JSON" in result.output


def test_import_refuses_non_object_value(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [[exp_id, "deposit_paid", "[1,2,3]"]],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "JSON object" in result.output


def test_import_refuses_text_leakage_evidence_id(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            [exp_id, "deposit_paid", '{"amount": 1}'],
            [exp_id, "interview_completed", '{"note": "see ev_deadbeef in the report"}'],
        ],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "refuse row 2" in result.output
    assert "text leakage" in result.output


def test_import_refuses_text_leakage_long_string(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    exp_id = _seed_running_experiment(session_factory, settings)
    long_text = "x" * 250
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [[exp_id, "interview_completed", f'{{"note": "{long_text}"}}']],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "refuse row 1" in result.output
    assert "len 250" in result.output


def test_import_refuses_illegal_status_transition(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    # Seed a stopped experiment (via iterate+advance flow keeps one stopped).
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        experiments = session.query(repo.Experiment).all()
        assert experiments
        stopped = experiments[0]
        stopped.status = "stopped"
        session.commit()
        exp_id = stopped.id

    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [[exp_id, "deposit_paid", '{"amount": 1}']],
    )
    # stopped → stopped is illegal (terminal state).
    result = runner.invoke(
        app,
        ["outcomes", "import", "--csv", str(csv), "--status", "stopped"],
    )
    assert result.exit_code == 1, result.output
    assert "legal transitions" in result.output


def test_import_refuses_unknown_experiment_with_row_number(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    _seed_running_experiment(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            ["exp_does_not_exist", "deposit_paid", '{"amount": 1}'],
            ["exp_also_missing", "deposit_paid", '{"amount": 1}'],
        ],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1, result.output
    assert "refuse row 1" in result.output
    assert "not found" in result.output


def test_import_keep_going_skips_and_continues(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    a, b = _seed_two_running(session_factory, settings)
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["experiment_id", "kind", "value"],
        [
            [a, "deposit_paid", '{"amount": 1}'],
            ["exp_missing_1", "deposit_paid", '{"amount": 1}'],   # bad
            [b, "interview_completed", '{"duration_min": 30}'],
            [b, "not_a_kind", '{}'],                              # bad
            [a, "paid_pilot_started", '{"weeks": 2}'],
        ],
    )
    result = runner.invoke(
        app, ["outcomes", "import", "--csv", str(csv), "--keep-going"]
    )
    assert result.exit_code == 1, result.output
    assert "imported 3 · skipped 2" in result.output
    assert "refuse row 2" in result.output
    assert "refuse row 4" in result.output


# ---- argument validation -------------------------------------------------


def test_import_rejects_bad_header(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    csv = _write_csv(
        tmp_path / "outcomes.csv",
        ["exp", "kind", "value"],  # wrong header
        [["exp_x", "deposit_paid", "{}"]],
    )
    result = runner.invoke(app, ["outcomes", "import", "--csv", str(csv)])
    assert result.exit_code == 1
    assert "header must be" in result.output


def test_import_rejects_missing_csv(
    fake_adapters, session_factory, settings, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("EE_DB_DSN", settings.db_url)
    monkeypatch.setattr(
        cli_module.Settings, "load", classmethod(lambda cls: settings)
    )
    result = runner.invoke(
        app, ["outcomes", "import", "--csv", str(tmp_path / "nope.csv")]
    )
    assert result.exit_code == 1
    assert "not found" in result.output
