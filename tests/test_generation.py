import pytest
import os
import tempfile
import subprocess
from generation.static_check import check_syntax
from generation.patch_generator import MockLLMClient, validate_and_apply_patch

def test_static_check_valid():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def foo():\n    return 42\n")
        path = f.name

    try:
        success, msg = check_syntax(path)
        assert success is True
        assert msg == ""
    finally:
        os.remove(path)

def test_static_check_invalid():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def foo():\nreturn 42\n") # Indentation error
        path = f.name

    try:
        success, msg = check_syntax(path)
        assert success is False
        assert "IndentationError" in msg or "SyntaxError" in msg
    finally:
        os.remove(path)

def test_validate_and_apply_patch_valid():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)

        diff = """--- /dev/null
+++ b/test.py
@@ -0,0 +1,2 @@
+def foo():
+    pass
"""
        # Patch application depends on being in the git repo directory
        cwd = os.getcwd()
        os.chdir(d)
        try:
            assert validate_and_apply_patch(diff) is True
            assert os.path.exists("test.py")
        finally:
            os.chdir(cwd)

def test_validate_and_apply_patch_invalid():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)

        diff = """--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,2 +1,2 @@
-def foo():
+def bar():
"""
        cwd = os.getcwd()
        os.chdir(d)
        try:
            assert validate_and_apply_patch(diff) is False
        finally:
            os.chdir(cwd)

def test_validate_and_apply_patch_with_explicit_cwd():
    """The cwd param lets a patch be applied against a directory other than the process's own cwd."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)

        diff = """--- /dev/null
+++ b/test.py
@@ -0,0 +1,2 @@
+def foo():
+    pass
"""
        assert validate_and_apply_patch(diff, cwd=d) is True
        assert os.path.exists(os.path.join(d, "test.py"))
        # process cwd itself is untouched
        assert not os.path.exists("test.py")

def test_multiline_diff_applies_completely():
    """
    Regression test for a Windows-only bug: writing the temp .patch file in
    default text mode translates '\\n' to '\\r\\n' on write, which corrupts
    a unified diff's line-based hunk parsing enough for git apply to
    silently apply only part of a hunk (observed: the last two lines of a
    30-line hunk went missing, with no error and no non-zero exit code).
    Doesn't reproduce the failure on Linux/macOS (default text mode never
    rewrites line endings there), but locks in that every line of a
    multi-line hunk actually lands.
    """
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)
        with open(os.path.join(d, "target.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True)

        lines = [f"line_{i} = {i}" for i in range(30)]
        diff = "--- a/target.py\n+++ b/target.py\n@@ -1 +1,30 @@\n-\n" + "".join(f"+{l}\n" for l in lines)

        assert validate_and_apply_patch(diff, cwd=d) is True
        with open(os.path.join(d, "target.py")) as f:
            result_lines = [l for l in f.read().splitlines() if l.strip()]
        assert result_lines == lines

def test_dry_run_validation_has_no_side_effects_and_is_repeatable():
    """
    Regression test: evolution/population.py validates every candidate's
    diff before scheduling it, without a worktree of its own. That check
    must never mutate the shared checkout - otherwise the first candidate's
    successful validation permanently changes the file on disk, and every
    later candidate (even with the identical diff) fails validation because
    the file no longer matches the diff's expected original content.
    """
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)
        with open(os.path.join(d, "candidate_script.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True)

        diff = MockLLMClient().generate_diff("goal", "candidate_script.py")

        for _ in range(3):
            assert validate_and_apply_patch(diff, cwd=d, dry_run=True) is True

        # file on disk must be untouched by repeated dry runs
        with open(os.path.join(d, "candidate_script.py")) as f:
            assert f.read() == "\n"

def test_evolution_fallback_diff_is_a_valid_patch():
    """Regression test: the fallback diff in EvolutionEngine._generate_candidate must actually apply."""
    import uuid

    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)
        with open(os.path.join(d, "candidate_script.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True)

        fallback_diff = f"--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('Fallback {uuid.uuid4().hex[:4]}')\n"
        assert validate_and_apply_patch(fallback_diff, cwd=d) is True

def test_mock_llm_diff_produces_a_real_predictions_writing_script():
    """
    The mock diff replaces a blank placeholder file with a script that
    reads TRAIN_PATH/TEST_PATH/SUBSET_PERCENTAGE and writes real
    predictions to /app/out/predictions.jsonl - that's what the eval
    pipeline actually scores. It also prints a diagnostic SCORE - but that
    line is never trusted for gating (see eval/pipeline.py and
    test_reward_hacking.py).
    """
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)
        with open(os.path.join(d, "candidate_script.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True)

        diff = MockLLMClient().generate_diff("prompt", "candidate_script.py")
        assert validate_and_apply_patch(diff, cwd=d) is True

        script_path = os.path.join(d, "candidate_script.py")
        success, msg = check_syntax(script_path)
        assert success, msg

        with open(script_path) as f:
            content = f.read()
        assert "TRAIN_PATH" in content
        assert "TEST_PATH" in content
        assert "SUBSET_PERCENTAGE" in content
        assert "predictions.jsonl" in content
        assert "SCORE:" in content  # diagnostic only, see above
        assert "truth" not in content.lower()  # the mock never sees held-out labels
        # Regression: a wrong hunk header line count is what a stricter git
        # silently truncates a patch to, rather than rejecting outright -
        # the entry-point guard was exactly what got cut once, so main()
        # was defined but never called and the script produced no output.
        assert 'if __name__ == "__main__":' in content
        assert content.strip().endswith("main()")

def test_mock_llm_diff_hunk_header_matches_actual_line_count():
    """
    Regression test for the bug above, at its actual root: a unified diff's
    hunk header must declare the true number of added/removed lines. git on
    Linux tolerated a wrong count in this hunk (30 declared vs 33 actual)
    well enough that this went unnoticed for a long time; git on Windows
    truncated the patch to the declared count instead of rejecting it.
    Check the arithmetic directly so this can never silently regress again,
    regardless of how lenient any particular git version happens to be.
    """
    diff = MockLLMClient().generate_diff("prompt", "candidate_script.py")
    lines = diff.splitlines()
    hunk_start = next(i for i, l in enumerate(lines) if l.startswith("@@"))
    header = lines[hunk_start]
    body = lines[hunk_start + 1:]  # excludes the "+++ b/..." file header, which also starts with "+"

    declared_added = int(header.split("+1,")[1].split(" ")[0])
    actual_added = sum(1 for l in body if l.startswith("+"))
    assert declared_added == actual_added, (
        f"hunk header declares {declared_added} added lines but diff actually has {actual_added}"
    )
