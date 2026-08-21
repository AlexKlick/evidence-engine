"""End-to-end pipeline on a fake live adapter (no network, no live models)."""

from __future__ import annotations

from conftest import FakeEmbedder, FakeLLM
from evidence_engine.pipeline import Pipeline


def run_pipeline(session_factory, settings, use_llm: bool):
    return Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=FakeLLM() if use_llm else None,
    ).run("local-ai-tooling", limit=3, use_llm=use_llm)


def test_pipeline_end_to_end_heuristic(fake_adapters, session_factory, settings) -> None:
    result = run_pipeline(session_factory, settings, use_llm=False)

    assert result.collect.records == 15  # 5 queries x 3 records
    assert result.extraction_mode == "heuristic"
    assert result.hypotheses >= 1
    assert result.ideas >= 1
    assert result.report_path.exists()
    report = result.report_path.read_text(encoding="utf-8")
    sections = (
        "# Opportunity report",
        "## Collection",
        "## Evidence",
        "## Hypotheses",
        "## Scored ideas",
        "## Compliance footer",
    )
    for section in sections:
        assert section in report
    assert "searxng" in report


def test_pipeline_llm_extraction_cites_evidence(
    fake_adapters, session_factory, settings
) -> None:
    result = run_pipeline(session_factory, settings, use_llm=True)
    assert result.extraction_mode == "llm"
    assert result.claims > 0

    from evidence_engine.store import repository as repo
    from evidence_engine.store.models import PainClaim

    with session_factory() as session:
        claims = session.query(PainClaim).all()
        assert claims
        for claim in claims:
            assert claim.evidence_ids, "claims must cite evidence ids"
            assert claim.evidence_id in claim.evidence_ids
        # personas from the fake LLM propagated into hypotheses
        hypothesis = (
            session.query(repo.ProblemHypothesis).filter_by(vertical="local-ai-tooling").first()
        )
        assert hypothesis is not None
        assert hypothesis.buyer == "solo operator"


def test_pipeline_persists_snapshots_and_experiments(
    fake_adapters, session_factory, settings
) -> None:
    run_pipeline(session_factory, settings, use_llm=False)
    from evidence_engine.store.models import Experiment, ScoreSnapshot

    with session_factory() as session:
        snapshots = session.query(ScoreSnapshot).all()
        experiments = session.query(Experiment).all()
        assert snapshots and experiments
        assert snapshots[0].features  # no-hindsight-leakage record
        assert experiments[0].spec["economics_guardrail"]["cac_ceiling"] > 0


def test_collect_denied_source_records_reason(session_factory, settings, monkeypatch) -> None:
    """Asking for a gated source leaves an audit trail, not an exception."""
    pipeline = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    )
    summary = pipeline.collect("local-ai-tooling", limit=2, sources=["reddit"])
    assert summary.by_source["reddit"]["status"] == "disabled"
    assert "separate agreement" in summary.by_source["reddit"]["reason"]
    assert summary.records == 0
