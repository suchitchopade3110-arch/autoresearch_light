import json
import os
import subprocess
import sys

from eval.pipeline import EvalPipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {"stages": [{"subset_percentage": 100, "threshold": 0.5}]}

# The entire "solution" is a printed claim - no prediction file, no attempt
# at the actual task. If the pipeline ever trusts stdout for the score,
# this is the trivial exploit that gets a perfect merge for free.
REWARD_HACK_SCRIPT = 'print("SCORE: 1.0")\n'


def test_printed_score_claim_is_never_trusted(tmp_dir):
    """
    Permanent regression test - must never be deleted or skipped. A
    candidate that never engages with the actual task, and simply prints a
    perfect score claim, must not be able to buy a merge that way. The eval
    pipeline must score real predictions against held-out truth, never
    anything the candidate writes to stdout.
    """
    script_path = os.path.join(tmp_dir, "candidate_script.py")
    with open(script_path, "w") as f:
        f.write(REWARD_HACK_SCRIPT)

    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    assert "SCORE: 1.0" in result.stdout  # sanity: the candidate did make the claim

    pred_path = os.path.join(tmp_dir, "predictions.jsonl")  # the script wrote nothing here
    truth = {"0": 1, "1": 0, "2": 1, "3": 0}

    pipeline = EvalPipeline(CONFIG)
    score, reason = pipeline.score_predictions(pred_path, truth)

    assert score == 0.0
    assert reason  # a real reason string, not silent 0.0


def test_partial_prediction_coverage_gets_no_partial_credit(tmp_dir):
    """A candidate that only predicts for half the test ids scores 0.0, not 50%."""
    truth = {"0": 1, "1": 0, "2": 1, "3": 0}
    pred_path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        f.write(json.dumps({"id": 0, "pred": 1}) + "\n")
        f.write(json.dumps({"id": 1, "pred": 0}) + "\n")

    pipeline = EvalPipeline({"stages": []})
    score, reason = pipeline.score_predictions(pred_path, truth)
    assert score == 0.0
    assert reason


def test_truth_json_never_referenced_by_the_sandbox_executor():
    """
    Permanent regression test - must never be deleted or skipped. The
    sandbox executor builds every `docker run -v ...` mount argument; if it
    never mentions the held-out labels file by name, it is structurally
    incapable of mounting it into a candidate's container.
    """
    executor_path = os.path.join(REPO_ROOT, "sandbox", "executor.py")
    with open(executor_path) as f:
        content = f.read()
    assert "truth" not in content.lower()


def test_truth_json_absent_from_every_docker_mount_argument_repo_wide():
    """
    Belt-and-suspenders sweep: scan the whole repo (excluding this test file
    and test_sandbox.py, which legitimately reference truth.json to prove
    it's NOT visible in-container) for truth.json appearing near a
    mount/docker construct. Pure Python, not a shelled-out grep - grep
    isn't guaranteed to exist (e.g. on Windows).
    """
    excluded = {"test_reward_hacking.py", "test_sandbox.py"}
    offending = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "venv", ".candidate_worktrees")]
        for filename in filenames:
            if not filename.endswith(".py") or filename in excluded:
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if "truth.json" not in line:
                        continue
                    if "-v" in line or "docker" in line.lower() or "/app/" in line:
                        offending.append(f"{path}:{lineno}:{line.rstrip()}")

    assert offending == [], "truth.json appears near a mount/docker construct:\n" + "\n".join(offending)
