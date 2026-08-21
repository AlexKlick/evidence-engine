"""Policy gate: default-deny, purpose rights, lint hygiene."""

from __future__ import annotations

import textwrap
from pathlib import Path

from evidence_engine.policy import PolicyRegistry, Purpose


def test_unknown_source_is_denied(policy: PolicyRegistry) -> None:
    decision = policy.decision("totally_unknown", Purpose.COLLECT)
    assert not decision.allowed
    assert "default-deny" in decision.reason


def test_enabled_source_collect_allowed(policy: PolicyRegistry) -> None:
    assert policy.decision("searxng", Purpose.COLLECT).allowed


def test_searxng_external_inference_denied(policy: PolicyRegistry) -> None:
    decision = policy.decision("searxng", Purpose.EXTERNAL_INFERENCE)
    assert not decision.allowed  # local-only inference posture


def test_reddit_collect_denied_with_contract_reason(policy: PolicyRegistry) -> None:
    decision = policy.decision("reddit", Purpose.COLLECT)
    assert not decision.allowed
    assert "separate agreement" in decision.reason


def test_youtube_aggregation_denied(policy: PolicyRegistry) -> None:
    assert not policy.decision("youtube", Purpose.AGGREGATE).allowed


def test_check_raises_permission_error(policy: PolicyRegistry) -> None:
    try:
        policy.check("reddit", Purpose.COLLECT)
    except PermissionError as exc:
        assert "policy gate denied" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


def test_rights_snapshot_is_row_level(policy: PolicyRegistry) -> None:
    snapshot = policy.rights_snapshot("searxng")
    assert snapshot["store_derived"] is True
    assert snapshot["commercial_use"] is False
    # unknown source -> all-false snapshot, never an exception
    assert PolicyRegistry().rights_snapshot("nope") == {
        key: False for key in snapshot
    }


def test_lint_flags_external_inference_without_commercial_use(tmp_path: Path) -> None:
    config = tmp_path / "policies.yaml"
    config.write_text(
        textwrap.dedent(
            """
            policy_version: 2
            sources:
              risky:
                adapter: searxng
                enabled: true
                rights:
                  collect_enabled: true
                  store_derived: true
                  external_inference: true
                  commercial_use: false
            """
        ),
        encoding="utf-8",
    )
    registry = PolicyRegistry.load(config)
    problems = registry.lint()
    assert any("external_inference" in problem for problem in problems)


def test_repo_registry_lints(policy: PolicyRegistry) -> None:
    problems = policy.lint()
    assert problems == []
