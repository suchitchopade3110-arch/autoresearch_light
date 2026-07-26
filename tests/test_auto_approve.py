import os
import subprocess
import tempfile
from unittest.mock import MagicMock

import pytest

from approval.gate import (
    maybe_auto_approve,
    request_and_await_approval,
    resolve_auto_approve_config,
    should_auto_approve,
)
from approval.store import ApprovalStore
from eval.baseline import BaselineStore
from evolution.scheduler import ConcurrentScheduler
from vcs.git_controller import GitController

MALFORMED_AUTO_APPROVE_CONFIGS = [
    ("approval.auto_approve missing entirely", {"approval": {"enabled": True}}),
    ("approval section missing entirely", {}),
    ("auto_approve is None", {"approval": {"auto_approve": None}}),
    ("auto_approve is a string", {"approval": {"auto_approve": "yes please"}}),
    ("min_improvement_over_baseline missing", {"approval": {"auto_approve": {"require_no_failure_flags": False}}}),
    ("min_improvement_over_baseline is a string", {"approval": {"auto_approve": {"min_improvement_over_baseline": "a lot"}}}),
    ("min_improvement_over_baseline is a bool", {"approval": {"auto_approve": {"min_improvement_over_baseline": True}}}),
    ("min_improvement_over_baseline is negative", {"approval": {"auto_approve": {"min_improvement_over_baseline": -0.1}}}),
]


@pytest.mark.parametrize("label,config", MALFORMED_AUTO_APPROVE_CONFIGS)
def test_malformed_or_missing_auto_approve_config_never_activates(label, config):
    resolved = resolve_auto_approve_config(config)
    assert resolved["min_improvement_over_baseline"] is None, f"failed for case: {label}"
    assert should_auto_approve(config, delta_over_baseline=1.0, has_failure_flags=False) is False, f"failed for case: {label}"


def test_require_no_failure_flags_defaults_to_true_when_omitted():
    resolved = resolve_auto_approve_config({"approval": {"auto_approve": {"min_improvement_over_baseline": 0.01}}})
    assert resolved["require_no_failure_flags"] is True


def test_require_no_failure_flags_malformed_type_fails_safe_to_true():
    resolved = resolve_auto_approve_config(
        {"approval": {"auto_approve": {"min_improvement_over_baseline": 0.01, "require_no_failure_flags": "nah"}}}
    )
    assert resolved["require_no_failure_flags"] is True


AUTO_APPROVE_CONFIG = {"approval": {"auto_approve": {"min_improvement_over_baseline": 0.05}}}


def test_should_auto_approve_true_when_improvement_clears_threshold_and_no_flags():
    assert should_auto_approve(AUTO_APPROVE_CONFIG, delta_over_baseline=0.1, has_failure_flags=False) is True


def test_should_auto_approve_false_when_improvement_below_threshold():
    assert should_auto_approve(AUTO_APPROVE_CONFIG, delta_over_baseline=0.01, has_failure_flags=False) is False


def test_should_auto_approve_false_when_no_baseline_to_compare_against():
    assert should_auto_approve(AUTO_APPROVE_CONFIG, delta_over_baseline=None, has_failure_flags=False) is False


def test_should_auto_approve_false_when_failure_flags_present_by_default():
    assert should_auto_approve(AUTO_APPROVE_CONFIG, delta_over_baseline=0.5, has_failure_flags=True) is False


def test_should_auto_approve_true_with_failure_flags_when_explicitly_permitted():
    permissive_config = {
        "approval": {"auto_approve": {"min_improvement_over_baseline": 0.05, "require_no_failure_flags": False}}
    }
    assert should_auto_approve(permissive_config, delta_over_baseline=0.5, has_failure_flags=True) is True


def test_maybe_auto_approve_records_a_distinct_terminal_state_not_approved():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

        decision = maybe_auto_approve(store, request_id, AUTO_APPROVE_CONFIG, delta_over_baseline=0.5, has_failure_flags=False)

        assert decision == "auto_approved"
        persisted = store.get_request(request_id)
        assert persisted["status"] == "auto_approved"
        assert persisted["status"] != "approved"


def test_maybe_auto_approve_returns_none_and_leaves_request_pending_when_criteria_not_met():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

        decision = maybe_auto_approve(store, request_id, AUTO_APPROVE_CONFIG, delta_over_baseline=0.01, has_failure_flags=False)

        assert decision is None
        assert store.get_request(request_id)["status"] == "pending"


def test_store_decide_accepts_auto_approved_status():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        assert store.decide(request_id, "auto_approved") is True
        assert store.get_request(request_id)["status"] == "auto_approved"


def test_request_and_await_approval_auto_approves_without_ever_sleeping():
    """The core behavioral proof: a qualifying candidate never waits on a human."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))

        def sleep_fn_that_must_never_be_called(seconds):
            raise AssertionError("auto-approval must never poll/sleep waiting for a human decision")

        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9,
            metrics={"delta": 0.5, "score_claim_mismatch": False},
            config={"approval": {"enabled": True, "auto_approve": {"min_improvement_over_baseline": 0.05}}},
            sleep_fn=sleep_fn_that_must_never_be_called,
        )

        assert decision == "auto_approved"
        assert store.list_all()[0]["status"] == "auto_approved"


def test_request_and_await_approval_falls_through_to_human_wait_when_delta_missing():
    """No baseline delta in metrics (e.g. a caller that never sets it) -> unchanged human-gated behavior."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        fake_time = [0.0]

        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9,
            metrics={},
            config={
                "approval": {
                    "enabled": True, "timeout_seconds": 10, "poll_interval_seconds": 3,
                    "auto_approve": {"min_improvement_over_baseline": 0.0},
                }
            },
            sleep_fn=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
            time_fn=lambda: fake_time[0],
        )

        assert decision == "timed_out"


def test_request_and_await_approval_ignores_auto_approve_when_not_configured():
    """Regression: auto_approve absent entirely -> today's always-human-gated behavior, unchanged."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        fake_time = [0.0]

        decision = request_and_await_approval(
            store, "cand-1", "goal", "diff", 0.9,
            metrics={"delta": 100.0, "score_claim_mismatch": False},  # would clearly qualify if auto_approve were configured
            config={"approval": {"enabled": True, "timeout_seconds": 10, "poll_interval_seconds": 3}},
            sleep_fn=lambda s: fake_time.__setitem__(0, fake_time[0] + s),
            time_fn=lambda: fake_time[0],
        )

        assert decision == "timed_out"


def _init_repo(tmp_dir):
    repo_dir = os.path.join(tmp_dir, "repo")
    os.makedirs(repo_dir)
    with open(os.path.join(repo_dir, "candidate_script.py"), "w") as f:
        f.write("\n")
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)
    return repo_dir


def test_scheduler_merges_a_qualifying_candidate_without_a_human_decision(tmp_dir):
    """
    Full evolutionary-mode integration: a candidate whose improvement over
    baseline clears approval.auto_approve.min_improvement_over_baseline
    merges in Phase 2 without ever reaching await_approval_decision.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    store = ApprovalStore(os.path.join(tmp_dir, "approvals.db"))
    baseline_store = BaselineStore(os.path.join(tmp_dir, "state.json"))  # fresh -> baseline 0.0 for every stage

    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 1.0)
    evaluator.last_stage_flags = {"score_claim_mismatch": False}

    candidates = [{"id": "aaa", "diff": "", "goal": "g"}]
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"auto_approve": {"min_improvement_over_baseline": 0.5}}}

    def await_that_must_never_be_called(*a, **kw):
        raise AssertionError("a candidate clearing auto_approve criteria must never reach await_approval_decision")

    import evolution.scheduler as scheduler_module
    original_await = scheduler_module.await_approval_decision
    scheduler_module.await_approval_decision = await_that_must_never_be_called
    try:
        evaluated = scheduler.execute_generation(
            candidates,
            eval_stages=[{"subset_percentage": 100, "threshold": 0.5}],
            git_controller=git_controller,
            sandbox=RecordingSandbox(),
            evaluator=evaluator,
            metrics_calculator=lambda r: {},
            failure_analyzer=lambda r, passed: ("failure", ""),
            approval_store=store,
            approval_config=gate_config,
            truth={"0": 1},
            baseline_store=baseline_store,
        )
    finally:
        scheduler_module.await_approval_decision = original_await

    assert evaluated[0]["approval_decision"] == "auto_approved"
    assert evaluated[0]["success"] is True
