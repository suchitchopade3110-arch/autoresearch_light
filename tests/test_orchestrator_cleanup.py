import os
import shutil
import signal
import subprocess
import sys
import tempfile

import pytest

from orchestrator.run import _install_signal_handlers
from vcs.git_controller import GitController


def test_sigterm_after_install_signal_handlers_exits_cleanly_instead_of_a_traceback():
    """
    Wave 3 acceptance: a SIGTERM (e.g. from a container runtime stopping the
    process) must exit promptly with a clear message, not propagate as an
    unhandled signal/traceback - any in-progress candidate is reclaimed by
    the next run's startup cleanup, not by this handler.

    Invokes the installed handler directly rather than actually raising the
    OS signal (e.g. via os.kill) - signal delivery semantics differ across
    platforms, and on Windows, os.kill(pid, SIGTERM) against the current
    process calls TerminateProcess() directly, bypassing any Python-level
    handler entirely (killing the whole test run instead of raising
    SystemExit). Calling the handler is what Python itself does once a
    signal is actually delivered, so this exercises the same logic
    portably.
    """
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    try:
        _install_signal_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGTERM, None)
        assert exc_info.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def _setup_harness(temp_repo):
    for item in ["configs", "orchestrator", "eval", "sandbox", "vcs", "memory", "generation", "approval",
                 "reporting", "observability"]:
        shutil.copytree(item, os.path.join(temp_repo, item))
    shutil.copy("config_schema.py", os.path.join(temp_repo, "config_schema.py"))

    with open(os.path.join(temp_repo, ".gitignore"), "w") as f:
        f.write("chroma_db/\n__pycache__/\n")
    with open(os.path.join(temp_repo, "candidate_script.py"), "w") as f:
        f.write("\n")

    subprocess.run(["git", "init"], cwd=temp_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=temp_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_repo, check=True, capture_output=True)


def test_cleanup_subcommand_reclaims_an_orphaned_candidate_without_needing_docker():
    """
    Wave 3 acceptance: crash recovery must be reachable as a standalone
    command, and must not require a sandbox/dataset/Docker to run - it only
    touches git state and the approval store.
    """
    with tempfile.TemporaryDirectory() as temp_repo:
        _setup_harness(temp_repo)

        # Simulate a run that crashed after creating a candidate worktree
        # and branch but before it could roll back or merge them.
        vcs = GitController(temp_repo)
        branch_name, worktree_path = vcs.create_branch("crashed")
        assert os.path.isdir(worktree_path)

        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo
        cmd = [sys.executable, "-m", "orchestrator.run", "cleanup", "--config", "configs/example.yaml"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_repo, env=env)

        assert result.returncode == 0, result.stderr
        assert "Removed 1 orphan worktree(s) and 1 orphan branch(es)." in result.stdout
        assert "Timed out 0 stale pending approval request(s)." in result.stdout
        assert not os.path.exists(worktree_path)
        assert branch_name not in [h.name for h in vcs.repo.heads]


def test_cleanup_subcommand_rejects_invalid_config_with_readable_error():
    with tempfile.TemporaryDirectory() as temp_repo:
        _setup_harness(temp_repo)

        bad_config_path = os.path.join(temp_repo, "bad.yaml")
        with open(bad_config_path, "w") as f:
            f.write("eval:\n  stages: []\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = temp_repo
        cmd = [sys.executable, "-m", "orchestrator.run", "cleanup", "--config", "bad.yaml"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_repo, env=env)

        assert result.returncode == 1
        assert "eval.stages" in result.stderr
