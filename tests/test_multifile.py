import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml

from config_schema import ConfigError, load_config
from generation.patch_generator import LLMClient, PatchGenerator
from generation.static_check import check_syntax_multi
from sandbox.executor import SandboxExecutor


class _PerFileDiffClient(LLMClient):
    """Stub LLM client returning a different, caller-supplied diff per target_file."""
    def __init__(self, diffs_by_file):
        self.diffs_by_file = diffs_by_file

    def generate_diff(self, prompt, target_file, current_content=""):
        return self.diffs_by_file[target_file]


def _replace_diff(filename, new_content):
    return f"--- a/{filename}\n+++ b/{filename}\n@@ -1 +1 @@\n-\n+{new_content}\n"


def _init_repo_with_files(tmp_dir, filenames):
    repo_dir = os.path.join(tmp_dir, "repo")
    os.makedirs(repo_dir)
    for name in filenames:
        with open(os.path.join(repo_dir, name), "w") as f:
            f.write("\n")
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)
    return repo_dir


def test_multi_file_diff_generation_and_apply(tmp_dir):
    """
    Priority 2 acceptance: a multi-file target_file list generates and
    applies one diff per file, each against that file's own current
    content, via the existing single-file generate_diff/
    validate_and_apply_patch underneath.
    """
    repo_dir = _init_repo_with_files(tmp_dir, ["file_a.py", "file_b.py"])
    client = _PerFileDiffClient({
        "file_a.py": _replace_diff("file_a.py", "print('A')"),
        "file_b.py": _replace_diff("file_b.py", "print('B')"),
    })
    generator = PatchGenerator(client)

    success, combined_diff = generator.generate_and_apply("goal", ["file_a.py", "file_b.py"], cwd=repo_dir)

    assert success is True
    with open(os.path.join(repo_dir, "file_a.py")) as f:
        assert f.read() == "print('A')\n"
    with open(os.path.join(repo_dir, "file_b.py")) as f:
        assert f.read() == "print('B')\n"
    assert "print('A')" in combined_diff
    assert "print('B')" in combined_diff


def test_multi_file_apply_stops_on_first_failing_file_and_leaves_partial_apply(tmp_dir):
    """
    A file whose diff doesn't apply cleanly stops the loop and returns
    False immediately - no custom partial-failure rollback is built here
    (by design; the caller discards the whole worktree via vcs.rollback()
    on any failure), so an earlier file's real, already-applied change is
    left in place. This test documents that contract explicitly.
    """
    repo_dir = _init_repo_with_files(tmp_dir, ["file_a.py", "file_b.py"])
    client = _PerFileDiffClient({
        "file_a.py": _replace_diff("file_a.py", "print('A')"),
        # Context line doesn't match file_b.py's real content ("\n") - git apply --check fails.
        "file_b.py": "--- a/file_b.py\n+++ b/file_b.py\n@@ -1 +1 @@\n-this does not match\n+print('B')\n",
    })
    generator = PatchGenerator(client)

    success, _ = generator.generate_and_apply("goal", ["file_a.py", "file_b.py"], cwd=repo_dir)

    assert success is False
    with open(os.path.join(repo_dir, "file_a.py")) as f:
        assert f.read() == "print('A')\n"
    with open(os.path.join(repo_dir, "file_b.py")) as f:
        assert f.read() == "\n"


def test_check_syntax_multi_passes_when_every_file_is_valid(tmp_dir):
    path_a = os.path.join(tmp_dir, "a.py")
    path_b = os.path.join(tmp_dir, "b.py")
    with open(path_a, "w") as f:
        f.write("x = 1\n")
    with open(path_b, "w") as f:
        f.write("y = 2\n")

    ok, err = check_syntax_multi([path_a, path_b])
    assert ok is True
    assert err == ""


def test_check_syntax_multi_fails_on_first_invalid_file():
    """Consistent with check_syntax's single-error-message contract - stops at the first failure, doesn't aggregate."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        good_path = os.path.join(d, "good.py")
        bad_path = os.path.join(d, "bad.py")
        with open(good_path, "w") as f:
            f.write("x = 1\n")
        with open(bad_path, "w") as f:
            f.write("def broken(:\n")

        ok, err = check_syntax_multi([good_path, bad_path])
        assert ok is False
        assert "bad.py" in err


def test_sandbox_mounts_extra_files_read_only(tmp_dir):
    """
    Priority 2 acceptance: extra_files are mounted individually (never the
    whole worktree directory, which would also expose .git and anything
    else sitting there). Doesn't need a real docker daemon.
    """
    script_path = os.path.join(tmp_dir, "candidate_script.py")
    extra_a = os.path.join(tmp_dir, "helper_a.py")
    extra_b = os.path.join(tmp_dir, "helper_b.py")
    for p in (script_path, extra_a, extra_b):
        with open(p, "w") as f:
            f.write("\n")

    with patch("sandbox.executor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor = SandboxExecutor({'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"})
        executor.run_candidate(script_path, extra_files=[extra_a, extra_b])

        run_call = next(c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "run"])
        cmd = run_call.args[0]

        assert f"{extra_a}:/app/helper_a.py:ro" in cmd
        assert f"{extra_b}:/app/helper_b.py:ro" in cmd


def test_sandbox_run_candidate_with_no_extra_files_is_unchanged(tmp_dir):
    """extra_files defaults to None - the single-file case must produce the exact same mount arguments as before."""
    script_path = os.path.join(tmp_dir, "candidate_script.py")
    with open(script_path, "w") as f:
        f.write("\n")

    with patch("sandbox.executor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        executor = SandboxExecutor({'timeout_seconds': 10, 'cpu_limit': "0.5", 'memory_limit': "256m"})
        executor.run_candidate(script_path)

        run_call = next(c for c in mock_run.call_args_list if c.args[0][:2] == ["docker", "run"])
        cmd = run_call.args[0]

        assert cmd.count("-v") == 1
        assert f"{script_path}:/app/candidate_script.py:ro" in cmd


def test_target_files_without_candidate_script_py_raises_config_error(tmp_dir):
    """
    Regression test: the sandbox's Dockerfile CMD always executes
    candidate_script.py by that exact name - a target.files list omitting
    it would silently generate patches for files the sandbox never runs.
    """
    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump({
            "eval": {"stages": [{"subset_percentage": 100, "threshold": 0.5}]},
            "target": {"files": ["other_file.py"]},
        }, f)

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "candidate_script.py" in str(exc_info.value)


def test_target_files_with_candidate_script_py_loads_fine(tmp_dir):
    config_path = os.path.join(tmp_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump({
            "eval": {"stages": [{"subset_percentage": 100, "threshold": 0.5}]},
            "target": {"files": ["candidate_script.py", "helper.py"]},
        }, f)

    config = load_config(config_path)
    assert config["target"]["files"] == ["candidate_script.py", "helper.py"]
