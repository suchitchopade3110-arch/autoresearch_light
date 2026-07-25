import os
import subprocess
from unittest.mock import MagicMock

from approval.store import ApprovalStore
from evolution.scheduler import ConcurrentScheduler
from vcs.git_controller import GitController


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


def test_all_sandbox_evaluation_completes_before_any_approval_request_blocks(tmp_dir):
    """
    Wave 3 acceptance: sandbox execution for a whole generation must
    complete before any approval request blocks compute. With a single
    worker, the old single-phase design would create and await candidate
    1's approval - blocking - before candidate 2's sandbox ever ran; the
    two-phase design runs every candidate's sandbox stages first and only
    then creates and awaits approval requests.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    events = []

    class RecordingSandbox:
        def run_candidate(self, script_path, env_vars=None, out_dir=None):
            events.append(("sandbox", script_path))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "predictions.jsonl"), "w") as f:
                f.write('{"id": "0", "pred": 1}\n')
            return {"exit_code": 0, "stdout": "", "stderr": "", "execution_time": 0.01, "timeout": False}

    class RecordingStore(ApprovalStore):
        def create_request(self, candidate_id, *a, **kw):
            events.append(("approval_request", candidate_id))
            return super().create_request(candidate_id, *a, **kw)

    store = RecordingStore(os.path.join(tmp_dir, "approvals.db"))
    evaluator = MagicMock()
    evaluator.evaluate_stage.return_value = (True, 1.0)

    candidates = [
        {"id": "aaa", "diff": "", "goal": "g"},
        {"id": "bbb", "diff": "", "goal": "g"},
    ]

    # max_workers=1: a single sandbox slot, so the two candidates run their
    # sandbox stages strictly one after another - the ordering assertion
    # below is meaningless with more workers, where sandbox calls could
    # interleave with approval creation for unrelated reasons.
    scheduler = ConcurrentScheduler(max_workers=1)
    gate_config = {"approval": {"enabled": True, "timeout_seconds": 0.05, "poll_interval_seconds": 0.01}}

    scheduler.execute_generation(
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
    )

    sandbox_indices = [i for i, e in enumerate(events) if e[0] == "sandbox"]
    approval_indices = [i for i, e in enumerate(events) if e[0] == "approval_request"]

    assert len(sandbox_indices) == 2
    assert len(approval_indices) == 2
    assert max(sandbox_indices) < min(approval_indices)
