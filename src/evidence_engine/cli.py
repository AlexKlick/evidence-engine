"""Typer CLI (`evidence-engine` / `ee`)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from evidence_engine import __version__
from evidence_engine.config import Settings, load_learned_weights, load_rubric, load_verticals
from evidence_engine.deletion.service import purge_expired
from evidence_engine.experiments.experiment import draft_experiment_spec
from evidence_engine.experiments.outcomes import OUTCOME_KINDS
from evidence_engine.ideas.review import apply_review
from evidence_engine.logging_setup import configure_logging
from evidence_engine.nlp.pricing import band_from_claims
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
experiments_app = typer.Typer(
    help="Experiment lifecycle: list/start/stop.", no_args_is_help=True
)
app.add_typer(policy_app, name="policy")
app.add_typer(deletions_app, name="deletions")
app.add_typer(outcomes_app, name="outcomes")
app.add_typer(experiments_app, name="experiments")

Verbose = Annotated[bool, typer.Option("--verbose", "-v", help="debug logging")]
Quiet = Annotated[bool, typer.Option("--quiet", "-q", help="warnings only")]


def _pipeline() -> Pipeline:
    return Pipeline()


def _resolve_verticals(vertical: str | None, all_: bool) -> list[str] | None:
    """Validate --vertical vs --all; None means every configured vertical."""
    if all_ and vertical:
        typer.echo("error: pass either --vertical or --all, not both")
        raise typer.Exit(1)
    if not all_ and not vertical:
        typer.echo("error: --vertical is required (or use --all)")
        raise typer.Exit(1)
    return None if all_ else [vertical]


def _echo_batch_failures(ok: int, failed: dict[str, str]) -> None:
    """Final batch line; exit non-zero when any vertical failed (for cron)."""
    typer.echo(f"{ok} verticals ok · {len(failed)} failed")
    for slug, message in failed.items():
        typer.echo(f"  failed: {slug} ({message})")
    if failed:
        raise typer.Exit(1)


def _echo_collect_summary(summary) -> None:
    typer.echo(
        f"collected {summary.records} records over {summary.runs} runs "
        f"for {summary.vertical}"
    )
    for name, entry in summary.by_source.items():
        typer.echo(f"  {name}: {entry.get('status', 'ok')} ({entry.get('records', 0)} records)")


def _echo_pipeline_result(result) -> None:
    typer.echo(f"vertical:        {result.vertical}")
    typer.echo(f"records:         {result.collect.records}")
    typer.echo(f"independent:     {result.originals} (duplicates: {result.duplicates})")
    typer.echo(f"claims:          {result.claims} ({result.extraction_mode})")
    typer.echo(f"clusters:        {result.clusters}")
    typer.echo(f"hypotheses/ideas: {result.hypotheses} / {result.ideas}")
    typer.echo(f"report:          {result.report_path}")


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
    vertical: Annotated[
        str | None, typer.Option("--vertical", "-y", help="vertical slug")
    ] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="run every configured vertical")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=30)] = 10,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="source name (repeatable)"),
    ] = None,
) -> None:
    """Run enabled adapters for a vertical (policy gate enforced)."""
    _resolve_verticals(vertical, all_)
    pipeline = _pipeline()
    sources = list(source) if source else None
    if all_:
        batch = pipeline.collect_all(limit=limit, sources=sources)
        for summary in batch.summaries:
            _echo_collect_summary(summary)
        _echo_batch_failures(len(batch.summaries), batch.failed)
        return
    _echo_collect_summary(pipeline.collect(vertical, limit=limit, sources=sources))


@app.command("pipeline")
def pipeline_run(
    vertical: Annotated[
        str | None, typer.Option("--vertical", "-y", help="vertical slug")
    ] = None,
    all_: Annotated[
        bool, typer.Option("--all", help="run every configured vertical")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=30)] = 8,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="heuristic extraction only")] = False,
) -> None:
    """Collect + process + score + report (full loop)."""
    _resolve_verticals(vertical, all_)
    if all_:
        batch = _pipeline().run_all(limit=limit, use_llm=not no_llm)
        for result in batch.results:
            _echo_pipeline_result(result)
        _echo_batch_failures(len(batch.results), batch.failed)
        return
    _echo_pipeline_result(_pipeline().run(vertical, limit=limit, use_llm=not no_llm))


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
    hypothesis: Annotated[str | None, typer.Option("--hypothesis", "-H", help="hyp_... id")] = None,
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
    queue: Annotated[
        bool,
        typer.Option(
            "--queue", help="list hypotheses failing hard gates or awaiting sign-off"
        ),
    ] = False,
    show: Annotated[
        bool,
        typer.Option(
            "--show", help="read-only evidence pack for -H (decide before editing)"
        ),
    ] = False,
) -> None:
    """Human review: set hypothesis fields, then rescore its ideas (gates update)."""
    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)

    if queue:
        from evidence_engine.ideas.review import review_queue

        rubric = load_rubric(settings)
        with make_session_factory(engine)() as session:
            entries = review_queue(session, rubric=rubric)
        if not entries:
            typer.echo("review queue empty — every hypothesis passes its gates")
            return
        typer.echo(f"{len(entries)} hypotheses needing review:")
        for entry in entries:
            if entry.get("awaiting_review"):
                status = "awaiting review sign-off (paid_validation capped)"
            else:
                status = f"missing={','.join(entry['missing'])}"
            typer.echo(
                f"  {entry['hypothesis_id']}  best={entry['best_score']:5.1f}  "
                f"ideas={entry['ideas']}  {status}"
            )
        return

    if show:
        from evidence_engine.ideas.review import (
            evidence_pack,
            render_evidence_pack,
        )

        update_flags = [buyer, channel, paid_test, compliance, job, pain]
        if not hypothesis:
            typer.echo("error: --show needs --hypothesis")
            raise typer.Exit(1)
        if any(value is not None for value in update_flags):
            typer.echo("error: --show is read-only; drop the update flags")
            raise typer.Exit(1)
        with make_session_factory(engine)() as session:
            try:
                pack = evidence_pack(session, hypothesis, rubric=load_rubric(settings))
            except KeyError as exc:
                typer.echo(f"error: {exc.args[0]}")
                raise typer.Exit(1) from exc
        for line in render_evidence_pack(pack):
            typer.echo(line)
        return

    if not hypothesis:
        typer.echo("error: --hypothesis is required (or use --queue or --show)")
        raise typer.Exit(1)

    rubric = load_rubric(settings)
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
            outcome = apply_review(
                session,
                hypothesis,
                rubric,
                updates,
                weights_override=load_learned_weights(settings),
            )
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
    status: Annotated[
        str | None,
        typer.Option("--status", help="draft->running->stopped (validated)"),
    ] = None,
    decision: Annotated[str | None, typer.Option("--decision", help="advance/kill/...")] = None,
) -> None:
    """Record a first-party outcome against an experiment (closes the loop)."""
    from evidence_engine.experiments.lifecycle import validate_transition

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
        if status:
            try:
                validate_transition(experiment_row.status, status)
            except ValueError as exc:
                typer.echo(f"error: {exc}")
                raise typer.Exit(1) from exc
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


@outcomes_app.command("import")
def outcomes_import(
    csv_path: Annotated[
        Path,
        typer.Option(
            "--csv",
            help="CSV file with header `experiment_id,kind,value`",
        ),
    ],
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="transition the experiment to this status (draft|running|stopped)",
        ),
    ] = None,
    decision: Annotated[
        str | None,
        typer.Option(
            "--decision",
            help="set experiment.decision to this string (free text)",
        ),
    ] = None,
    keep_going: Annotated[
        bool,
        typer.Option(
            "--keep-going",
            help="log all refusals and continue, do not abort at first failure",
        ),
    ] = False,
) -> None:
    """Batch-import outcomes from a CSV (one outcome per row).

    Validation per row, in order: column count == 3; `kind` in
    `OUTCOME_KINDS`; `value` parses as a JSON object; the synthetic
    Outcome passes `outcome_value_carries_text`; the experiment exists;
    if `--status` is set the transition is legal. Atomic by default;
    `--keep-going` collects all refusals before exiting.
    """
    import csv as csv_mod
    import json

    from evidence_engine.experiments.outcomes import OUTCOME_KINDS
    from evidence_engine.ranking.fit import outcome_value_carries_text
    from evidence_engine.store.models import Outcome

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    if not csv_path.is_file():
        typer.echo(f"error: {csv_path} not found")
        raise typer.Exit(1)

    refusals: list[tuple[int, str]] = []
    imported = 0

    def _refuse(row_num: int, message: str) -> None:
        refusals.append((row_num, message))
        typer.echo(f"refuse row {row_num}: {message}", err=True)

    with make_session_factory(engine)() as session, csv_path.open(
        "r", encoding="utf-8"
    ) as handle:
        reader = csv_mod.reader(handle)
        rows = list(reader)

    if not rows:
        typer.echo("error: CSV is empty")
        raise typer.Exit(1)

    header, *data_rows = rows
    expected = ["experiment_id", "kind", "value"]
    if [c.strip() for c in header] != expected:
        typer.echo(
            f"error: header must be {expected}, got {[c.strip() for c in header]}"
        )
        raise typer.Exit(1)

    total = len(data_rows)
    for index, row in enumerate(data_rows, start=1):
        if len(row) != 3:
            _refuse(index, f"expected 3 columns, got {len(row)}")
            if not keep_going:
                break
            continue
        experiment_id, kind, value_text = (cell.strip() for cell in row)
        if kind not in OUTCOME_KINDS:
            _refuse(index, f"unknown kind {kind!r}")
            if not keep_going:
                break
            continue
        if value_text == "":
            payload: dict = {}
        else:
            try:
                parsed = json.loads(value_text)
            except json.JSONDecodeError as exc:
                _refuse(index, f"malformed JSON: {exc.msg}")
                if not keep_going:
                    break
                continue
            if not isinstance(parsed, dict):
                _refuse(index, "value must be a JSON object")
                if not keep_going:
                    break
                continue
            payload = parsed

        synthetic = Outcome(
            id=f"__csv_check_{index:04d}",
            experiment_id=experiment_id,
            kind=kind,
            value=payload,
        )
        snippet = outcome_value_carries_text(synthetic)
        if snippet is not None:
            _refuse(index, f"text leakage: {snippet}")
            if not keep_going:
                break
            continue

        experiment = session.get(repo.Experiment, experiment_id)
        if experiment is None:
            _refuse(index, f"experiment {experiment_id!r} not found")
            if not keep_going:
                break
            continue

        if status is not None:
            from evidence_engine.experiments.lifecycle import validate_transition

            try:
                validate_transition(experiment.status, status)
            except ValueError as exc:
                _refuse(index, exc.args[0])
                if not keep_going:
                    break
                continue
            experiment.status = status

        if decision is not None:
            experiment.decision = decision

        outcome = repo.record_outcome(session, experiment, kind, payload)
        typer.echo(
            f"[{index}/{total}] outcome {outcome.id} kind={kind} on {experiment_id}"
        )
        imported += 1

    session.commit()

    typer.echo(f"imported {imported} · skipped {len(refusals)}")
    if refusals:
        raise typer.Exit(1)



@experiments_app.command("list")
def experiments_list(
    vertical: Annotated[str | None, typer.Option("--vertical", "-y")] = None,
    status: Annotated[
        str | None, typer.Option("--status", "-s", help="draft|running|stopped")
    ] = None,
) -> None:
    """List experiments with their predeclared spend caps."""
    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        rows = repo.list_experiments(session, vertical=vertical, status=status)
        if not rows:
            typer.echo("no experiments match")
            return
        for row in rows:
            idea = repo.get_idea(session, row.idea_id)
            cap = (row.spec or {}).get("maximum_spend")
            cap_text = f"${cap}" if cap is not None else "unset (no price evidence)"
            typer.echo(
                f"{row.id}  {(idea.vertical if idea else '?'):<26} {row.status:<8} "
                f"cap {cap_text}  {row.decision or '—'}  idea:{row.idea_id}"
            )


@experiments_app.command("start")
def experiments_start(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="exp_... id")],
    price: Annotated[
        float | None,
        typer.Option(
            "--price",
            help="operator-reviewed monthly price (overrides the derived band)",
        ),
    ] = None,
) -> None:
    """draft -> running; surfaces the predeclared economics before you spend."""
    from evidence_engine.experiments.experiment import cac_ceiling
    from evidence_engine.experiments.lifecycle import start_experiment

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    defaults = load_rubric(settings).get("experiment_defaults") or {}
    with make_session_factory(engine)() as session:
        row = session.get(repo.Experiment, experiment)
        if row is None:
            typer.echo(f"error: experiment {experiment!r} not found")
            raise typer.Exit(1)
        if price is not None:
            if price <= 0:
                typer.echo("error: --price must be a positive monthly amount")
                raise typer.Exit(1)
            spec = dict(row.spec or {})
            guardrail = dict(spec.get("economics_guardrail") or {})
            margin = float(
                guardrail.get("gross_margin") or defaults.get("gross_margin", 0.85)
            )
            months = int(
                guardrail.get("cac_payback_months")
                or defaults.get("cac_payback_months", 6)
            )
            cap = cac_ceiling(price, margin, months)
            guardrail.update(
                price_monthly=price,
                price_provenance=(
                    f"reviewed:operator {datetime.now(UTC):%Y-%m-%d}"
                ),
                cac_ceiling=cap,
            )
            spec["economics_guardrail"] = guardrail
            spec["maximum_spend"] = cap
            row.spec = spec  # reassign: JSON columns need the new object
        try:
            row = start_experiment(session, experiment)
            session.commit()
        except (KeyError, ValueError) as exc:
            typer.echo(f"error: {exc.args[0]}")
            raise typer.Exit(1) from exc
        spec = row.spec or {}
        guardrail = spec.get("economics_guardrail") or {}
        typer.echo(f"started {row.id} (idea {row.idea_id})")
        typer.echo(f"  primary metric: {spec.get('primary_metric')}")
        typer.echo(
            f"  spend cap: ${spec.get('maximum_spend')} "
            f"(price ${guardrail.get('price_monthly')}/mo × margin "
            f"{guardrail.get('gross_margin')} × "
            f"{guardrail.get('cac_payback_months')}-mo payback)"
        )
        typer.echo(f"  price: {guardrail.get('price_provenance') or 'unknown'}")
        typer.echo(f"  stop condition: {spec.get('stop_condition')}")


@experiments_app.command("stop")
def experiments_stop(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="exp_... id")],
    decision: Annotated[
        str | None, typer.Option("--decision", help="advance|kill|iterate|...")
    ] = None,
) -> None:
    """running -> stopped, with the operator's decision on record."""
    from evidence_engine.experiments.lifecycle import stop_experiment

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        try:
            row = stop_experiment(session, experiment, decision)
            session.commit()
        except (KeyError, ValueError) as exc:
            typer.echo(f"error: {exc.args[0]}")
            raise typer.Exit(1) from exc
        typer.echo(f"stopped {row.id} (decision: {row.decision or '—'})")


@experiments_app.command("decide")
def experiments_decide(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="exp_... id")],
    decision: Annotated[
        str,
        typer.Option(
            "--decision",
            "-d",
            help="advance|kill|iterate (typed via experiments/decisions.py)",
        ),
    ],
    reason: Annotated[
        str | None,
        typer.Option("--reason", "-r", help="free-text reason; capped at 500 chars"),
    ] = None,
    decider: Annotated[
        str,
        typer.Option(
            "--decider",
            help="who decided (default: operator). Recorded on the Outcome.value.",
        ),
    ] = "operator",
    override: Annotated[
        bool,
        typer.Option(
            "--override",
            help="suppress 'advance without prior paid outcome' warning",
        ),
    ] = False,
) -> None:
    """Apply a typed decision to a running experiment, synthesize an Outcome.

    Routes through validate_transition (lifecycle.py:30). The decision is
    written to experiment.decision (typed) AND a row of matching kind
    ('advance'|'kill'|'iterate') is appended to outcome, so calibration
    (ranking/calibration.py:33) finally has something decision-aware to
    join on.
    """
    from evidence_engine.experiments.decisions import DECISIONS, apply_decision

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    if decision not in DECISIONS:
        typer.echo(
            f"error: --decision must be one of {', '.join(DECISIONS)} "
            f"(got {decision!r})"
        )
        raise typer.Exit(1)
    with make_session_factory(engine)() as session:
        try:
            row, outcome, warnings = apply_decision(
                session,
                experiment,
                decision,  # type: ignore[arg-type]
                reason=reason,
                decider=decider,
                override=override,
            )
            session.commit()
        except (KeyError, ValueError) as exc:
            typer.echo(f"error: {exc.args[0]}")
            raise typer.Exit(1) from exc
    typer.echo(f"{decision} {row.id}: status={row.status}")
    if outcome is not None:
        typer.echo(f"  synthesized outcome {outcome.id} (kind={outcome.kind})")
    for warning in warnings:
        typer.echo(f"  warning: {warning}")


@app.command()
def calibration(
    fit: Annotated[
        bool,
        typer.Option(
            "--fit",
            help="Fit logistic on snapshots → outcomes; write data/ranker_weights.json + audit copy.",
        ),
    ] = False,
    show_weights: Annotated[
        bool,
        typer.Option(
            "--show-weights",
            help="Print data/ranker_weights.json as a table.",
        ),
    ] = False,
    clear: Annotated[
        bool,
        typer.Option(
            "--clear",
            help="Delete data/ranker_weights.json (audit copies in reports/ are kept).",
        ),
    ] = False,
    sprint_log: Annotated[
        bool,
        typer.Option(
            "--sprint-log",
            help="Append a timestamped snapshot to gate-logs/sprint.md.",
        ),
    ] = False,
    idea: Annotated[
        str | None,
        typer.Option(
            "--idea",
            help="(with --show-weights) print score diff vs YAML for one idea_id.",
        ),
    ] = None,
) -> None:
    """Show the closed-loop dataset: feature snapshots joined with outcomes.

    Flags (mutually exclusive in spirit; --fit wins if multiple are passed):
      --fit          Fit logistic regression on the joined dataset and write
                     data/ranker_weights.json + reports/ranker-weights-<stamp>.json.
      --show-weights Pretty-print the active artifact, or fall back message.
      --clear        Delete the primary artifact (audit copies stay).
      --sprint-log   Append a timestamped markdown block to gate-logs/sprint.md.
      --idea ID      With --show-weights: filter + score diff vs YAML for one idea.
    """
    settings = Settings.load()

    if idea and not show_weights:
        typer.echo(
            "usage: --idea requires --show-weights (it's a read-only filter on the weights table)"
        )
        raise typer.Exit(1)

    if sprint_log and (fit or show_weights or clear):
        typer.echo(
            "warning: --sprint-log takes precedence over --fit, --show-weights, --clear"
        )

    if fit:
        _calibration_fit(settings)
        return
    if show_weights:
        _calibration_show_weights(settings, idea=idea)
        return
    if clear:
        _calibration_clear(settings)
        return
    if sprint_log:
        _calibration_sprint_log(settings)
        return

    # Default: print the joined dataset (unchanged from prior behavior).
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
                f"decision={row.get('decision') or '—'} "
                f"outcomes={[o['kind'] for o in row['outcomes']]}"
            )


def _calibration_fit(settings: Settings) -> None:
    from evidence_engine.ranking.fit import (
        InsufficientLabelDiversity,
        RankerFitLeakage,
        RankerFitRefused,
        fit_logistic,
        write_weights_artifact,
    )

    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        try:
            dataset = assemble_calibration_dataset(
                session, check_no_text_leakage=True
            )
        except RankerFitLeakage as exc:
            typer.echo(f"error: {exc.args[0]}")
            raise typer.Exit(1) from exc

    if not dataset.ready_to_fit:
        typer.echo(
            f"error: not ready_to_fit (outcome_count={dataset.outcome_count} < "
            f"MIN_OUTCOMES_FOR_FIT=30) — run more decisions first"
        )
        raise typer.Exit(1)

    rows: list[dict] = []
    for row in dataset.rows:
        decision = row.get("decision")
        if decision is None:
            continue
        rows.append(
            {
                "features": row["features"] or {},
                "label": 1 if decision == "advance" else 0,
                "decision": decision,
            }
        )

    try:
        fit_result = fit_logistic(rows)
    except InsufficientLabelDiversity as exc:
        typer.echo(f"error: {exc.args[0]} — fit refused")
        raise typer.Exit(1) from exc
    except RankerFitRefused as exc:
        typer.echo(f"error: {exc.args[0]}")
        raise typer.Exit(1) from exc

    primary = settings.data_dir / "ranker_weights.json"
    audit_path = write_weights_artifact(fit_result, primary, settings.reports_dir)

    counts = fit_result.decision_counts
    advance = counts.get("advance", 0)
    collapse = sum(v for k, v in counts.items() if k != "advance")
    typer.echo(
        f"fitted_at {fit_result.row_count} rows · "
        f"advance {advance} / kill-or-iterate {collapse} · "
        f"loss {fit_result.loss:.4f}"
    )
    typer.echo(f"  primary:  {primary}")
    typer.echo(f"  audit:    {audit_path}")


def _calibration_show_weights(settings: Settings, idea: str | None = None) -> None:
    payload = load_learned_weights(settings)
    if payload is None:
        typer.echo(
            f"no ranker_weights.json at {settings.data_dir / 'ranker_weights.json'} — "
            "score falls back to YAML rubric"
        )
        return

    if idea is not None:
        typer.echo(
            f"filter: idea_id={idea} · weights from {settings.data_dir / 'ranker_weights.json'}"
        )

    typer.echo(
        f"fitted_at {payload.get('fitted_at', '?')} · "
        f"row_count {payload.get('row_count', '?')} · "
        f"loss {payload.get('loss', '?')}"
    )
    typer.echo(
        f"intercept  {payload.get('intercept', 0.0):+.4f}"
    )

    weights = payload.get("weights", {})
    means = payload.get("means", [])
    stds = payload.get("stds", [])
    name_to_index = {name: i for i, name in enumerate(payload.get("features", []))}
    rows = []
    for name, weight in weights.items():
        index = name_to_index.get(name, -1)
        if 0 <= index < len(means):
            mean = means[index]
            std = stds[index] if index < len(stds) else 0.0
        else:
            mean = 0.0
            std = 0.0
        rows.append((name, weight, mean, std))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    for name, weight, mean, std in rows:
        typer.echo(f"  {name:<30}  {weight:+.4f}    z={mean:+.3f}±{std:.3f}")

    dropped = payload.get("dropped_constant_features", [])
    if dropped:
        typer.echo(f"dropped (constant columns): {', '.join(dropped)}")

    if idea is not None:
        _show_weights_idea_diff(settings, idea, weights)


def _show_weights_idea_diff(
    settings: Settings, idea: str, weights: dict
) -> None:
    """Print YAML-baseline vs post-fit `score_idea` totals + delta."""
    from sqlalchemy import select

    from evidence_engine.config import load_rubric
    from evidence_engine.ideas.scoring import score_idea
    from evidence_engine.store.models import ProductIdea, ScoreSnapshot

    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        idea_row = session.get(ProductIdea, idea)
        if idea_row is None:
            typer.echo(f"(idea {idea!r} not in store; skipping score diff)")
            return
        snapshot = (
            session.execute(
                select(ScoreSnapshot).where(ScoreSnapshot.idea_id == idea)
            )
            .scalars()
            .first()
        )
        if snapshot is None or not snapshot.features:
            typer.echo(f"(idea {idea} has no score snapshot; skipping score diff)")
            return
        hypothesis = repo.get_hypothesis(session, idea_row.hypothesis_id)
        if hypothesis is None:
            typer.echo(
                f"(idea's hypothesis not in store; skipping score diff)"
            )
            return

    rubric = load_rubric(settings)
    features = dict(snapshot.features or {})
    baseline = score_idea(features, hypothesis, idea_row.form, rubric)
    override = {
        name: float(w) for name, w in weights.items() if isinstance(w, (int, float))
    }
    learned = score_idea(
        features, hypothesis, idea_row.form, rubric, weights_override=override
    )
    typer.echo(f"score_idea(pre-fit, yaml): {baseline.total}")
    typer.echo(f"score_idea(post-fit, learned): {learned.total}")
    typer.echo(f"delta: {learned.total - baseline.total:+.4f}")


def _calibration_clear(settings: Settings) -> None:
    primary = settings.data_dir / "ranker_weights.json"
    if not primary.is_file():
        typer.echo("already clear")
        return
    primary.unlink()
    typer.echo(f"removed {primary} (audit copy in reports/ kept)")


def _calibration_sprint_log(settings: Settings) -> None:
    """Append a timestamped markdown block to gate-logs/sprint.md.

    Provenance rule: every line names its source. Reuses
    `assemble_calibration_dataset` for snapshot/outcome counts, the
    ranker artifact for fit metadata, and a single Outcome scan for
    the decision distribution + per-vertical running counts.
    """
    import io as _io
    from collections import Counter

    from sqlalchemy import select

    from evidence_engine.experiments.outcomes import OUTCOME_KINDS as _OK
    from evidence_engine.ranking.calibration import assemble_calibration_dataset
    from evidence_engine.store.models import Experiment, Outcome, ProductIdea

    stamp = datetime.now(UTC).isoformat()

    try:
        engine = make_engine(settings.db_url)
        init_db(engine)
        with make_session_factory(engine)() as session:
            dataset = assemble_calibration_dataset(session)
        with make_session_factory(engine)() as session:
            decision_kinds = (
                "advance",
                "kill",
                "iterate",
            )
            outcome_rows = list(
                session.execute(
                    select(Outcome).where(Outcome.kind.in_(decision_kinds))
                ).scalars()
            )
            decision_counts = Counter(r.kind for r in outcome_rows)

            running_per_vertical: dict[str, int] = {}
            for experiment in session.execute(
                select(Experiment).where(Experiment.status == "running")
            ).scalars():
                idea = session.get(ProductIdea, experiment.idea_id)
                if idea is None:
                    continue
                running_per_vertical[idea.vertical] = (
                    running_per_vertical.get(idea.vertical, 0) + 1
                )
    except Exception as exc:  # noqa: BLE001 — surface as a loud sprint-log fail
        typer.echo(f"error: sprint-log read failed: {exc}")
        raise typer.Exit(1) from exc

    payload = load_learned_weights(settings)
    if payload is not None:
        fit_meta = (
            f"fitted_at={payload.get('fitted_at', '?')} · "
            f"loss={payload.get('loss', '?')}"
        )
    else:
        fit_meta = "(no weights fit yet)"

    lines = _io.StringIO()
    lines.write(f"## Sprint snapshot — {stamp}\n")
    lines.write(
        f"- source: `ee calibration` snapshot — "
        f"snapshots={len(dataset.rows)} · "
        f"outcomes={dataset.outcome_count} · "
        f"ready_to_fit={dataset.ready_to_fit}\n"
    )
    lines.write(f"- source: `data/ranker_weights.json` — {fit_meta}\n")
    if running_per_vertical:
        lines.write("- source: `ee calibration` per-vertical running counts:\n")
        for vertical, count in sorted(running_per_vertical.items()):
            lines.write(f"      - {vertical}: {count}\n")
    else:
        lines.write("- source: `ee calibration` per-vertical running counts: (none running)\n")
    lines.write(
        "- source: `ee calibration` decision distribution: "
        f"advance={decision_counts.get('advance', 0)} · "
        f"kill={decision_counts.get('kill', 0)} · "
        f"iterate={decision_counts.get('iterate', 0)}\n"
    )
    lines.write(
        f"- data_dir: {settings.data_dir} · "
        f"reports_dir: {settings.reports_dir} · "
        f"gate_logs: {settings.gate_logs_dir}\n"
    )

    target = settings.gate_logs_dir / "sprint.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        target.write_text("# Sprint log\n\n", encoding="utf-8")
    with target.open("a", encoding="utf-8") as handle:
        handle.write(lines.getvalue())
    typer.echo(f"appended → {target}")


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
        evidence_ids = list(hypothesis.evidence_for or []) if hypothesis else []
        evidence = repo.evidence_by_ids(session, evidence_ids)
        rubric = load_rubric(settings)
        spec = draft_experiment_spec(
            idea,
            hypothesis,
            rubric.get("experiment_defaults") or {},
            price_band=band_from_claims(
                repo.claims_for_evidence(session, evidence_ids)
            ),
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
def competitors(
    vertical: Annotated[str, typer.Option("--vertical", "-y", help="vertical slug")],
) -> None:
    """Competitor pressure map, derived from entitled evidence only."""
    from evidence_engine.ideas.competitors import (
        alternatives_overview,
        render_competitors_markdown,
    )

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        overview = alternatives_overview(session, vertical)
    if not overview.direct_software:
        typer.echo(
            "no incumbents extracted yet — run `ee pipeline` with LLM extraction"
        )
        return
    for line in render_competitors_markdown(overview):
        typer.echo(line)


@app.command()
def landing(
    idea_id: Annotated[str, typer.Option("--idea", "-i", help="idea id (idea_...)")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="output dir")] = None,
    html_export: Annotated[
        bool,
        typer.Option(
            "--html", help="also write static landing-a/b.html + events.json"
        ),
    ] = False,
) -> None:
    """Emit the landing-page draft + experiment plan for an idea."""
    import yaml

    from evidence_engine.experiments.landing import (
        landing_content,
        render_landing_markdown,
    )
    from evidence_engine.experiments.landing_html import render_landing_html
    from evidence_engine.experiments.redisplay_guard import (
        VerbatimRedisplayError,
        assert_no_verbatim_redisplay,
    )

    settings = Settings.load()
    engine = make_engine(settings.db_url)
    init_db(engine)
    with make_session_factory(engine)() as session:
        idea = repo.get_idea(session, idea_id)
        if idea is None:
            typer.echo(f"idea {idea_id!r} not found")
            raise typer.Exit(1)
        hypothesis = repo.get_hypothesis(session, idea.hypothesis_id)
        # persisted Experiment.spec is authoritative: `experiments start
        # --price` commits operator economics there, and the landing must show
        # the RUNNING experiment's numbers, not a fresh draft from claims
        experiment = repo.latest_experiment_for_idea(session, idea.id)
        persisted_guardrail = (
            (experiment.spec or {}).get("economics_guardrail") or {}
            if experiment is not None
            else {}
        )
        # authoritative only when the price is grounded (derived band) or
        # operator-reviewed — a legacy unprovenanced $99 must never render
        if persisted_guardrail.get("price_provenance"):
            spec = dict(experiment.spec)
        else:
            evidence_ids_for_band = (
                list(hypothesis.evidence_for or []) if hypothesis else []
            )
            spec = draft_experiment_spec(
                idea,
                hypothesis,
                load_rubric(settings).get("experiment_defaults") or {},
                price_band=band_from_claims(
                    repo.claims_for_evidence(session, evidence_ids_for_band)
                ),
            )
        content = landing_content(idea, hypothesis, spec)
        markdown = render_landing_markdown(idea, hypothesis, spec)
        # publication gate (ADR-0005): no scraped text may reach the landing
        # artifacts (markdown or html path). Guards evidence_for AND
        # evidence_against lineage, and fails closed when a referenced row is
        # missing (deleted lineage). report/bootstrap outputs remain
        # operator-local and unguarded — follow-up, see review 2026-08-22
        evidence_ids = []
        if hypothesis is not None:
            evidence_ids = list(hypothesis.evidence_for or []) + list(
                hypothesis.evidence_against or []
            )
        evidence_rows = repo.evidence_by_ids(session, evidence_ids)
        texts = {
            "markdown": markdown,
            "html_a": render_landing_html(content, "a"),
            "html_b": render_landing_html(content, "b"),
        }
        try:
            assert_no_verbatim_redisplay(texts, evidence_rows, expected_ids=evidence_ids)
        except VerbatimRedisplayError as exc:
            typer.echo(f"error: {exc}")
            raise typer.Exit(1) from exc
    target = out or (settings.ideas_dir / idea.id)
    target.mkdir(parents=True, exist_ok=True)
    (target / "landing.md").write_text(markdown, encoding="utf-8")
    (target / "experiment-plan.md").write_text(
        "# Experiment plan (predeclared metrics — no retrospective selection)\n\n```yaml\n"
        + yaml.safe_dump(spec, sort_keys=False)
        + "```\n",
        encoding="utf-8",
    )
    typer.echo(f"landing + experiment plan: {target}")
    if html_export:
        from evidence_engine.experiments.landing_html import write_landing_html

        for path in write_landing_html(target, content):
            typer.echo(f"html export: {path}")


@app.command()
def expand(
    vertical: Annotated[str, typer.Option("--vertical", "-y", help="vertical slug")],
    limit: Annotated[int, typer.Option("--limit", "-l", min=1, max=30)] = 10,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="local | minimax (default: config)"),
    ] = None,
) -> None:
    """Expand a vertical's seed queries into an intent-mixed set (seed_expand).

    Input is operator-authored seeds only (no collected evidence leaves the
    host). Output lands in config/expanded_queries/<slug>.yaml — prune it
    before the next collect; that file IS the approval step.
    """
    import os

    from evidence_engine.nlp.llm import LLMClient
    from evidence_engine.nlp.query_expansion import (
        ExpandedQuery,
        expand_queries,
        load_expanded,
        save_expanded,
    )

    settings = Settings.load()
    if provider:
        os.environ["EE_LLM_PROVIDER"] = provider
        settings = Settings.load()
    vertical_data = load_verticals(settings).get(vertical)
    if vertical_data is None:
        typer.echo(f"unknown vertical {vertical!r}")
        raise typer.Exit(1)

    existing = [q.text for q in load_expanded(settings, vertical)]
    client = LLMClient(settings)
    if not client.health():
        typer.echo(
            f"llm provider {settings.llm_provider!r} unavailable — "
            "check `ee doctor`"
        )
        raise typer.Exit(1)

    fresh = expand_queries(
        client,
        vertical,
        seeds=vertical_data.get("queries", []),
        existing=existing,
        limit=limit,
    )
    if not fresh:
        typer.echo("expansion produced no new queries (provider error or all dups)")
        raise typer.Exit(1)

    merged = [
        ExpandedQuery(text=q.text, intent=q.intent)
        for q in load_expanded(settings, vertical)
    ] + [ExpandedQuery(text=q.text, intent=q.intent) for q in fresh]
    path = save_expanded(settings, vertical, merged, client.model_name)
    typer.echo(f"+{len(fresh)} new queries ({len(merged)} total) -> {path}")
    for query in fresh:
        typer.echo(f"  [{query.intent}] {query.text}")


@app.command()
def queries(
    vertical: Annotated[str | None, typer.Option("--vertical", "-y")] = None,
) -> None:
    """Show effective query sets: seeds + expansion sidecar."""
    from evidence_engine.nlp.query_expansion import expanded_path, load_expanded

    settings = Settings.load()
    for slug, vertical_data in load_verticals(settings).items():
        if vertical and slug != vertical:
            continue
        seeds = vertical_data.get("queries", [])
        expanded = load_expanded(settings, slug)
        sidecar = (
            " (sidecar: "
            + str(expanded_path(settings, slug).relative_to(settings.config_dir))
            + ")"
            if expanded
            else ""
        )
        typer.echo(
            f"{slug} — {len(seeds)} seed + {len(expanded)} expanded{sidecar}"
        )
        for query in seeds:
            typer.echo(f"  seed       {query}")
        for query in expanded:
            typer.echo(f"  expanded   [{query.intent}] {query.text}")


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
