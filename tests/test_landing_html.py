"""Static landing HTML export: self-contained variants + events contract."""

from __future__ import annotations

import json

from conftest import FakeEmbedder
from evidence_engine.config import load_rubric
from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.experiments.landing import (
    landing_content,
    render_landing_markdown,
)
from evidence_engine.experiments.landing_html import (
    EVENT_NAMES,
    render_landing_html,
    write_landing_html,
)
from evidence_engine.pipeline import Pipeline
from evidence_engine.store import repository as repo

VERTICAL = "local-ai-tooling"


def content_for(session_factory, settings, mutate=None):
    Pipeline(
        settings=settings, session_factory=session_factory, embedder=FakeEmbedder()
    ).run(VERTICAL, limit=3, use_llm=False)
    with session_factory() as session:
        idea = sorted(
            repo.ideas_for_vertical(session, VERTICAL),
            key=lambda idea: idea.score_total,
            reverse=True,
        )[0]
        hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
        if mutate:
            mutate(hypothesis)
        spec = draft_experiment_spec(
            idea, hypothesis, load_rubric(settings).get("experiment_defaults") or {}
        )
        content = landing_content(idea, hypothesis, spec)
    return content


def test_html_variants_self_contained(
    fake_adapters, session_factory, settings, tmp_path
) -> None:
    content = content_for(session_factory, settings)
    paths = write_landing_html(tmp_path, content)
    assert [path.name for path in paths] == [
        "landing-a.html",
        "landing-b.html",
        "events.json",
    ]
    for variant in ("a", "b"):
        text = (tmp_path / f"landing-{variant}.html").read_text(encoding="utf-8")
        assert "<script" not in text.lower(), "static pages must carry no JS"
        assert "src=" not in text, "no external assets"
        assert 'href="http' not in text, "no external links (CTA is #checkout)"
        assert "<style>" in text, "inline CSS present"
        # price-visibility test: price appears before the CTA in source order
        assert text.index(f"${content.price_monthly}") < text.index("Start paid pilot")
        assert f"Variant {variant.upper()}" in text
    # variants differ in framing headline (the <title> carries the shared pitch)
    page_a = (tmp_path / "landing-a.html").read_text(encoding="utf-8")
    page_b = (tmp_path / "landing-b.html").read_text(encoding="utf-8")
    h1_a = page_a.split("<h1>")[1].split("</h1>")[0]
    h1_b = page_b.split("<h1>")[1].split("</h1>")[0]
    assert h1_a != h1_b
    events = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in events["events"]] == list(EVENT_NAMES)


def test_events_schema_lists_five_events() -> None:
    from evidence_engine.experiments.landing_html import events_schema

    assert tuple(entry["name"] for entry in events_schema()) == EVENT_NAMES
    for entry in events_schema():
        assert entry["when"] and isinstance(entry["payload"], dict)


def test_landing_html_escapes_evidence_derived_copy(
    fake_adapters, session_factory, settings
) -> None:
    def mutate(hypothesis):
        hypothesis.buyer = "Solo <operator> & co"

    content = content_for(session_factory, settings, mutate=mutate)
    page_a = render_landing_html(content, "a")
    page_b = render_landing_html(content, "b")
    # feature framing uses the buyer as-is; outcome framing title-cases it
    assert "&lt;operator&gt;" in page_a
    assert "&lt;Operator&gt;" in page_b
    assert "<operator>" not in page_a + page_b  # no raw markup injection


def test_shared_content_model_consistent(
    fake_adapters, session_factory, settings, tmp_path
) -> None:
    content = content_for(session_factory, settings)
    with session_factory() as session:
        idea = sorted(
            repo.ideas_for_vertical(session, VERTICAL),
            key=lambda idea: idea.score_total,
            reverse=True,
        )[0]
        hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
        spec = draft_experiment_spec(
            idea, hypothesis, load_rubric(settings).get("experiment_defaults") or {}
        )
        markdown = render_landing_markdown(idea, hypothesis, spec)
    write_landing_html(tmp_path, content)
    page = (tmp_path / "landing-a.html").read_text(encoding="utf-8")

    assert f"${content.price_monthly}/month" in markdown
    assert f"${content.price_monthly}/month" in page
    assert content.pitch in markdown
    assert content.pitch in page
