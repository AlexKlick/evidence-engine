"""Typer CLI (`evidence-engine` / `ee`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from evidence_engine import __version__
from evidence_engine.config import Settings, load_rubric, load_verticals
from evidence_engine.deletion.service import purge_expired
from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.experiments.outcomes import OUTCOME_KINDS
from evidence_engine.ideas.review import apply_review
from evidence_engine.logging_setup import configure_logging
from evidence_engine.pipeline import Pipeline
from evidence_engine.policy import PolicyRegistry
from evidence_engine.ranking.calibration import assemble_calibration_dataset
from evidence_engine.report import (
    collect_summary_from_db,
    render_vertical_report,
    stored_embed_mode,
    write_report,
)
from evidence_engine.store import init_db, make_engine, make_session_factory
from evidence_engine.store import repository as repo

app = typer.Typer(
    help="Rights-aware evidence-to-revenue engine.", no_args_is_help=True, add_completion=False
)
policy_app = typer.Typer(help="Inspect the source entitlement registry.", no_args_is_help=True)
deletions_app = typer.Typer(help="Retention + lineage deletion.", no_args_is_help=True)
outcomes_app = typer.Typer(help="Record first-party experiment outcomes.", no_args_is_help=True)
app.add_typer(policy_app, name="policy")
app.add_typer(deletions_app, name="deletions")
app.add_typer(outcomes_app, name="outcomes")

Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="debug logging")]
Quiet = Annotated[bool, typer.Option("--quiet", "-q", help="warnings only")]


def _pipeline() -> Pipeline:
    return Pipeline()


@app.command()
def doctor(
    offline: Annotated[bool, typer.Option("--offline", help="skip network checks")] = False,
) -> None:
    """Health: DB, policy registry, SearXNG, text-main LLM, embeddings."""
    import httpx

    settings = Settings.load()
    policy = PolicyRegistry.load(settings.policies_path)
    checks: list[tuple[str, bool, str]] = []

    try:
        engine = make_engine(settings.db_url)
        init_db(engine)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        checks.append(("database", True, settings.db_url.split("@")[-1]))
    except Exception as exc:  # noqa: BLE001
        checks.append(("database", False, str(exc)[:100]))

    problems = policy.lint()
    checks.append(
        (
            "policy registry",
            not problems,
            f"v{policy.policy_version}, {len(policy.all_sources())} sources",
        )
    )

    if not offline:
        from evidence_engine.nlp.llm import LLMClient

        llm_client = LLMClient(settings)
        llm_ok = llm_client.health()
        checks.append(
            (
                f"llm [{settings.llm_provider}]",
                llm_ok,
                f"{settings.llm_model}",
            )
        )
        for name, url in [
            (
                "searxng",
                f"{settings.searxng_base_url}/search?q=health%20check&format=json",
            ),
            ("embeddings :6900", f"{settings.embeddings_base_url}/models"),
        ]:
            try:
                response = httpx.get(url, timeout=6.0)
                checks.append((name, response.status_code == 200, f"{response.status_code}"))
            except httpx.HTTPError as exc:
                checks.append((name, False, str(exc)[:100]))

    width = max(len(name) for name, _, _ in checks)
    all_ok = True
    for name, ok, detail in checks:
        all_ok &= ok
        typer.echo(f"{'ok  ' if ok else 'FAIL'} {name:<{width}}  {detail}")
    if problems:
        typer.echo("policy lint problems:")
        for problem in problems:
            typer.echo(f"  - {problem}")
    raise typer.Exit(0 if all_ok else 1)


@policy_app.command("show")
def policy_show() -> None:
    """List sources: enabled, rights highlights, disabled reasons."""
    settings = Settings.load()
    policy = PolicyRegistry.load(settings.policies_path)
    typer.echo(f"policy_version: {policy.policy_version}")
    for name, source in sorted(policy.all_sources().items()):
        rights = source.rights
        state = "ENABLED " if source.enabled else "gated   "
        typer.echo(
            f"{state} {name:<14} collect={rights.collect_enabled} "
            f"store_derived={rights.store_derived} aggregate={rights.aggregate} "
            f"local_inf={rights.local_inference} external_inf={rights.external_inference} "
            f"commercial={rights.commercial_use} retention={source.retention_days or '—'}d"
        )
        if not source.enabled and source.disabled_reason:
            typer.echo(f"         reason: {source.disabled_reason.strip()[:120]}")


@policy_app.command("lint")
def policy_lint() -> None:
    """Registry hygiene problems."""
    settings = Settings.load()
    policy = PolicyRegistry.load(settings.policies_path)
    problems = policy.lint()
    if not problems:
        typer.echo("policy registry: ok")
        return
    for problem in problems:
        typer.echo(f"- {problem}")
    raise typer.Exit(1)


@app.command()
def collect(
    vertical: Annotated[str, typer.Option("--vertical", "-y", help="vertical slug")],
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=30)] = 10,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="source name (repeatable)"),
    ] = None,
) -> None:
    """Run enabled adapters for a vertical (policy gate enforced)."""
    pipeline = _pipeline()
    summary = pipeline.collect(vertical, limit=limit, sources=list(source) if source else None)
    typer.echo(
        f"collected {summary.records} records over {summary.runs} runs "
        f"for {summary.vertical}"
    )
    for name, entry in summary.by_source.items():
        typer.echo(f"  {name}: {entry.get('status', 'ok')} ({entry.get('records', 0)} records)")


@app.command("pipeline")
def pipeline_run(
    vertical: Annotated[str, typer.Option("--vertical", "-y", help="vertical slug")],
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=30)] = 8,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="heuristic extraction only")] = False,
) -> None:
    """Collect + process + score + report (full loop)."""
    result = _pipeline().run(vertical, limit=limit, use_llm=not no_llm)
    typer.echo(f"vertical:        {result.vertical}")
    typer.echo(f"records:         {result.collect.records}")
    typer.echo(f"independent:     {result.originals} (duplicates: {result.duplicates})")
    typer.echo(f"claims:          {result.claims} ({result.extraction_mode})")
    typer.echo(f"clusters:        {result.clusters}")
    typer.echo(f"hypotheses/ideas: {result.hypotheses} / {result.ideas}")
    typer.echo(f"report:          {result.report_path}")


@app.command()
def report(
    vertical: Annotated[str, typer.Option("--vertical", "-y", help="vertical slug")],
) -> None:
    """Regenerate the opportunity report from the store (no collection)."""
    settings = Settings.load()
    policy = PolicyRegistry.load(settings.policies_path)
    rubric = load_rubric(settings)
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        summary = collect_summary_from_db(session, vertical)
        originals = repo.evidence_for_vertical(session, vertical)
        markdown = render_vertical_report(
            session,
            vertical,
            settings=settings,
            policy=policy,
            rubric=rubric,
            embedding_model=settings.embeddings_model,
            collect_summary=summary,
            embed_mode=stored_embed_mode(originals),
        )
    path = write_report(markdown, settings, vertical)
    typer.echo(f"report: {path}")


@app.command()
def review(
    hypothesis: Annotated[str, typer.Option("--hypothesis", "-H", help="hyp_... id")],
    buyer: Annotated[str | None, typer.Option("--buyer", "-b")] = None,
    channel: Annotated[str | None, typer.Option("--channel", "-c")] = None,
    paid_test: Annotated[
        str | None, typer.Option("--smallest-paid-test", "-t")
    ] = None,
    compliance: Annotated[
        str | None,
        typer.Option("--compliance", help="policy_ok_local_research | rights_review_required"),
    ] = None,
    job: Annotated[str | None, typer.Option("--job")] = None,
    pain: Annotated[str | None, typer.Option("--pain")] = None,
) -> None:
    """Human review: set hypothesis fields, then rescore its ideas (gates update)."""
    settings = Settings.load()
    rubric = load_rubric(settings)
    engine = make_engine(settings.db_url)
    init_db(engine)
    updates = {
        "buyer": buyer,
        "channel": channel,
        "smallest_paid_test": paid_test,
        "compliance_status": compliance,
        "job": job,
        "pain": pain,
    }
    with make_session_factory(engine)() as session:
        try:
            outcome = apply_review(session, hypothesis, rubric, updates)
            session.commit()
        except (KeyError, ValueError) as exc:
            typer.echo(f"error: {exc}")
            raise typer.Exit(1) from exc
    if outcome.applied:
        typer.echo(f"applied: {outcome.applied}")
    else:
        typer.echo("no fields updated (nothing to do)")
        return
    typer.echo(f"rescored {len(outcome.ideas)} ideas:")
    for entry in outcome.ideas:
        failed = ",".join(entry["failed_gates"]) or "pass"
        typer.echo(
            f"  {entry['id']}  {entry['total']:5.1f}  {entry['band']:<15} "
            f"{entry['form']:<24} gates:{failed}"
        )


@outcomes_app.command("record")
def outcomes_record(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="exp_... id")],
    kind: Annotated[
        str,
        typer.Option("--kind", "-k", help=f"one of: {', '.join(OUTCOME_KINDS)}"),
    ],
    value: Annotated[str, typer.Option("--value", help="JSON object")] = "{}",
    status: Annotated[str | None, typer.Option("--status", help="experiment status")] = None,
    decision: Annotated[str | None, typer.Option("--decision", help="advance/kill/...")] = None,
) -> None:
    """Record a first-party outcome against an experiment (closes the loop)."""
    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    try:
        payload = json.loads(value) if value else {}
    except json.JSONDecodeError as exc:
        typer.echo(f"--value is not valid JSON: {exc}")
        raise typer.Exit(1) from exc
    if not isinstance(payload, dict):
        typer.echo("--value must be a JSON object")
        raise typer.Exit(1)
    with make_session_factory(engine)() as session:
        experiment_row = session.get(repo.Experiment, experiment)
        if experiment_row is None:
            typer.echo(f"experiment {experiment!r} not found")
            raise typer.Exit(1)
        outcome = repo.record_outcome(session, experiment_row, kind, payload)
        if status:
            experiment_row.status = status
        if decision:
            experiment_row.decision = decision
        session.commit()
        typer.echo(f"recorded {outcome.id}: {kind} on {experiment}")


@outcomes_app.command("list")
def outcomes_list(
    experiment: Annotated[
        str | None, typer.Option("--experiment", "-e", help="filter by exp_... id")
    ] = None,
) -> None:
    """List recorded outcomes."""
    from sqlalchemy import select

    from evidence_engine.store.models import Outcome

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        stmt = select(Outcome)
        if experiment:
            stmt = stmt.where(Outcome.experiment_id == experiment)
        rows = list(session.execute(stmt).scalars())
        if not rows:
            typer.echo("no outcomes recorded")
            return
        for row in sorted(rows, key=lambda r: r.observed_at):
            typer.echo(f"{row.id}  {row.observed_at:%Y-%m-%d %H:%M}  {row.kind:<22} {row.value}")


@app.command()
def calibration() -> None:
    """Show the closed-loop dataset: feature snapshots joined with outcomes."""
    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        dataset = assemble_calibration_dataset(session)
    typer.echo(f"snapshots: {len(dataset.rows)} · outcomes: {dataset.outcome_count}")
    typer.echo(f"ready_to_fit: {dataset.ready_to_fit}")
    if not dataset.ready_to_fit:
        typer.echo(
            "(calibration stays a stub until enough first-party outcomes exist — "
            "hand-fit weights on tiny data would be ceremony, not calibration)"
        )
    for row in dataset.rows:
        if row["outcomes"]:
            typer.echo(
                f"  {row['idea_id']}: total={row['total']} "
                f"outcomes={[o['kind'] for o in row['outcomes']]}"
            )


@app.command(name="ideas")
def ideas_list(
    vertical: Annotated[str | None, typer.Option("--vertical", "-y")] = None,
    band: Annotated[
        str | None,
        typer.Option("--band", help="paid_validation|interview|collect_more|archive"),
    ] = None,
) -> None:
    """List scored ideas from the store."""
    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        ideas = repo.ideas_for_vertical(session, vertical) if vertical else list(
            session.query(repo.ProductIdea).all()
        )
        if band:
            ideas = [idea for idea in ideas if idea.band == band]
        if not ideas:
            typer.echo("no ideas yet — run `ee pipeline --vertical <slug>` first")
            return
        ideas = sorted(ideas, key=lambda idea: idea.score_total, reverse=True)
        for idea in ideas[:30]:
            failed = [name for name, ok in (idea.gates or {}).items() if not ok]
            typer.echo(
                f"{idea.id}  {idea.score_total:5.1f}  {idea.band:<15} "
                f"{idea.form:<24} gates:{','.join(failed) or 'pass'}"
            )


@app.command()
def bootstrap(
    idea_id: Annotated[str, typer.Option("--idea", "-i", help="idea id (idea_...)")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="output dir")] = None,
) -> None:
    """Scaffold a project folder for an idea (PRD/evidence/pricing/...)."""
    from evidence_engine.bootstrap.generator import scaffold_idea_project

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        idea = repo.get_idea(session, idea_id)
        if idea is None:
            typer.echo(f"idea {idea_id!r} not found")
            raise typer.Exit(1)
        hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
        evidence = repo.evidence_by_ids(
            session, list(hypothesis.evidence_for or []) if hypothesis else []
        )
        rubric = load_rubric(settings)
        spec = draft_experiment_spec(
            idea, hypothesis, rubric.get("experiment_defaults") or {}
        )
        out_dir = out or (settings.ideas_dir / idea.id)
        root = scaffold_idea_project(out_dir, idea, hypothesis, evidence, spec)
    typer.echo(f"scaffolded {root}")


@deletions_app.command("purge")
def deletions_purge() -> None:
    """Delete evidence past retention TTLs (lineage-aware)."""
    settings = Settings.load()
    policy = PolicyRegistry.load(settings.policies_path)
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        events = purge_expired(session, policy)
        session.commit()
    if not events:
        typer.echo("nothing past retention")
        return
    for event in events:
        typer.echo(
            f"{event.id}: {event.source} — {len(event.evidence_ids or [])} evidence, "
            f"{event.affected_claims} claims, {event.affected_segments} segments "
            f"({event.reason})"
        )


@app.command()
def verticals() -> None:
    """List configured seed verticals."""
    settings = Settings.load()
    for slug, vertical in load_verticals(settings).items():
        queries = vertical.get("queries", [])
        typer.echo(f"{slug:<28} {vertical.get('name', '')} ({len(queries)} queries)")


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="show version")] = False,
    verbose: Verbose = False,
    quiet: Quiet = False,
) -> None:
    configure_logging(verbose=verbose, quiet=quiet)
    if version:
        typer.echo(f"evidence-engine {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def main() -> None:  # console_scripts entry point
    app()


if __name__ == "__main__":
    main()
