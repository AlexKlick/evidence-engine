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
        # heuristic run carries no price signals: NO price, NO cap — the old
        # $99 -> $504.90 default is gone (start gates on --price instead)
        guardrail = experiments[0].spec["economics_guardrail"]
        assert guardrail["price_monthly"] is None
        assert guardrail["cac_ceiling"] is None
        assert guardrail["price_provenance"] is None
        assert experiments[0].spec["maximum_spend"] is None


def test_pipeline_derives_price_from_claim_signals(
    fake_adapters, session_factory, settings
) -> None:
    """Claims' price signals flow into the spec as a derived median + band."""
    from conftest import PricedFakeLLM
    from evidence_engine.store.models import Experiment

    Pipeline(
        settings=settings,
        session_factory=session_factory,
        embedder=FakeEmbedder(),
        llm=PricedFakeLLM(),
    ).run("local-ai-tooling", limit=3, use_llm=True)

    with session_factory() as session:
        experiments = session.query(Experiment).all()
        assert experiments
        for experiment in experiments:
            guardrail = experiment.spec["economics_guardrail"]
            # every claim: $9.99/mo or $99/yr -> band [8.25, 9.99] -> median 9.12 -> 9
            assert guardrail["price_monthly"] == 9
            assert guardrail["cac_ceiling"] == 45.9  # 9 * 0.85 * 6
            assert experiment.spec["maximum_spend"] == 45.9
            assert guardrail["price_provenance"].startswith("derived:median(n=")


def test_collect_denied_source_records_reason(session_factory, settings, monkeypatch) -> None:
    """Asking for a gated source leaves an audit trail, not an exception."""
    pipeline = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    )
    summary = pipeline.collect("local-ai-tooling", limit=2, sources=["reddit"])
    assert summary.by_source["reddit"]["status"] == "disabled"
    assert "separate agreement" in summary.by_source["reddit"]["reason"]
    assert summary.records == 0


def test_run_all_processes_every_configured_vertical(
    fake_adapters, session_factory, settings
) -> None:
    batch = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run_all(limit=2, use_llm=False)
    # seed-file order: local-ai-tooling, solo-dev-saas, prediction-market-research
    assert [result.vertical for result in batch.results] == [
        "local-ai-tooling",
        "solo-dev-saas",
        "prediction-market-research",
    ]
    assert batch.failed == {}
    assert all(result.report_path.exists() for result in batch.results)


def test_run_all_isolates_vertical_failures(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    pipeline = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    )
    original = pipeline.run

    def failing(slug, limit=None, use_llm=None):
        if slug == "solo-dev-saas":
            raise RuntimeError("boom")
        return original(slug, limit=limit, use_llm=use_llm)

    monkeypatch.setattr(pipeline, "run", failing)
    batch = pipeline.run_all(limit=2, use_llm=False)
    assert [result.vertical for result in batch.results] == [
        "local-ai-tooling",
        "prediction-market-research",
    ]
    assert set(batch.failed) == {"solo-dev-saas"}
    assert batch.failed["solo-dev-saas"]


def test_collect_all_returns_summary_per_vertical(
    fake_adapters, session_factory, settings
) -> None:
    batch = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).collect_all(limit=2)
    assert len(batch.summaries) == 3
    assert batch.failed == {}
    assert all(summary.records > 0 for summary in batch.summaries)


def test_pipeline_caps_clusters_at_ten(
    fake_adapters, session_factory, settings, monkeypatch
) -> None:
    """Clustering past the top-10 cap must compile cleanly, not zip-crash.

    Found live: an expanded query set produced 12 groups while cluster_rows
    stayed capped at 10 — zip(..., strict=True) raised ValueError.
    """
    from types import SimpleNamespace

    import evidence_engine.pipeline as pipeline_module
    from evidence_engine.store import repository as repo

    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run("local-ai-tooling", limit=3, use_llm=False)

    with session_factory() as session:
        ids = [
            row.id
            for row in repo.evidence_for_vertical(session, "local-ai-tooling")
            if not row.is_duplicate_of
        ]
    assert len(ids) >= 2, "fake seed run must yield originals to split"

    groups = [
        SimpleNamespace(label=f"forced group {index}", member_ids=[ids[index % len(ids)]])
        for index in range(12)  # more groups than the top-10 cap
    ]
    monkeypatch.setattr(
        pipeline_module, "cluster_evidence", lambda *args, **kwargs: groups
    )

    result = Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run("local-ai-tooling", limit=2, use_llm=False)
    assert result.clusters == 10  # capped, not crashed
    assert result.hypotheses == 10
