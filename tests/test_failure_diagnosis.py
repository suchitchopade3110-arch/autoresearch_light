import contextlib
import os
import subprocess
import tempfile

from evolution.population import EvolutionEngine
from generation.patch_generator import LLMClient, PatchGenerator, validate_and_apply_patch
from generation.prompt_builder import PromptBuilder
from memory.db import ExperimentDB
from memory.failure_analysis import analyze_failure
from orchestrator.metrics import calculate_all_metrics
from vcs.git_controller import GitController


def make_result(stdout="", stderr="", exit_code=0, timeout=False):
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "execution_time": 0.1, "timeout": timeout}


@contextlib.contextmanager
def git_repo():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=d, check=True, capture_output=True)
        yield d


def test_analyze_failure_returns_the_real_stderr_as_traceback_for_runtime_failures():
    category, error_text, traceback = analyze_failure(
        make_result(exit_code=1, stderr="Traceback (most recent call last):\nValueError: boom"), False
    )
    assert category == "runtime"
    assert "ValueError: boom" in error_text
    assert "ValueError: boom" in traceback


def test_analyze_failure_returns_the_real_stderr_as_traceback_for_syntax_failures():
    category, error_text, traceback = analyze_failure(
        make_result(exit_code=1, stderr="  File \"x.py\", line 1\nSyntaxError: invalid syntax"), False
    )
    assert category == "syntax"
    assert traceback == error_text
    assert "SyntaxError" in traceback


def test_analyze_failure_traceback_is_empty_for_metric_regression():
    """No stack trace exists for a candidate that ran fine but scored too low - traceback must stay empty, not fabricate one."""
    category, error_text, traceback = analyze_failure(make_result(exit_code=0), False)
    assert category == "metric-regression"
    assert error_text  # a human-readable label still exists
    assert traceback == ""


def test_analyze_failure_traceback_is_empty_on_success():
    category, error_text, traceback = analyze_failure(make_result(exit_code=0), True)
    assert category == "success"
    assert traceback == ""


def test_analyze_failure_timeout_traceback_is_raw_stderr_not_the_fabricated_default_message():
    """error_text falls back to a friendly 'Execution timed out.' when stderr is empty - traceback must not inherit that fabrication."""
    category, error_text, traceback = analyze_failure(make_result(timeout=True, stderr=""), False)
    assert category == "timeout"
    assert error_text == "Execution timed out."
    assert traceback == ""


def test_validate_and_apply_patch_error_out_captures_the_real_git_diagnostic():
    with git_repo() as d:
        bad_diff = "--- a/nonexistent.py\n+++ b/nonexistent.py\n@@ -1,2 +1,2 @@\n-def foo():\n+def bar():\n"
        error_out = []
        result = validate_and_apply_patch(bad_diff, cwd=d, error_out=error_out)
        assert result is False
        assert len(error_out) == 1
        assert error_out[0]  # git's real stderr, not empty


def test_validate_and_apply_patch_error_out_stays_empty_on_success():
    with git_repo() as d:
        diff = "--- /dev/null\n+++ b/test.py\n@@ -0,0 +1,2 @@\n+def foo():\n+    pass\n"
        error_out = []
        assert validate_and_apply_patch(diff, cwd=d, error_out=error_out) is True
        assert error_out == []


def test_validate_and_apply_patch_omitting_error_out_is_unaffected():
    """Backward compatibility: every pre-existing call site never passes error_out."""
    with git_repo() as d:
        bad_diff = "--- a/nonexistent.py\n+++ b/nonexistent.py\n@@ -1,2 +1,2 @@\n-def foo():\n+def bar():\n"
        assert validate_and_apply_patch(bad_diff, cwd=d) is False


class _FixedDiffClient(LLMClient):
    def __init__(self, diff):
        self.diff = diff

    def generate_diff(self, prompt, target_file, current_content=""):
        return self.diff


def test_patch_generator_captures_last_apply_error_on_failure():
    with git_repo() as d:
        with open(os.path.join(d, "candidate_script.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True, capture_output=True)

        bad_diff = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1,5 +1,5 @@\n-this does not match\n+x\n"
        generator = PatchGenerator(_FixedDiffClient(bad_diff))

        success, _ = generator.generate_and_apply("goal", "candidate_script.py", cwd=d)

        assert success is False
        assert generator.last_apply_error  # real git diagnostic, not empty


def test_patch_generator_clears_last_apply_error_on_a_later_success():
    """A stale error from a previous failed candidate must never leak into a later successful one's record."""
    with git_repo() as d:
        with open(os.path.join(d, "candidate_script.py"), "w") as f:
            f.write("\n")
        subprocess.run(["git", "add", "."], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=d, check=True, capture_output=True)

        bad_diff = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1,5 +1,5 @@\n-this does not match\n+x\n"
        good_diff = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('ok')\n"

        generator = PatchGenerator(_FixedDiffClient(bad_diff))
        generator.generate_and_apply("goal", "candidate_script.py", cwd=d)
        assert generator.last_apply_error

        generator.llm_client = _FixedDiffClient(good_diff)
        success, _ = generator.generate_and_apply("goal", "candidate_script.py", cwd=d)

        assert success is True
        assert generator.last_apply_error == ""


def test_db_round_trips_the_traceback_field_distinct_from_failure_reason(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(
        hypothesis="goal", diff="d", rationale="r", metrics={}, outcome="failure",
        failure_reason="Malformed diff rejected by git apply.",
        traceback="error: patch failed: candidate_script.py:1",
    )

    results = db.retrieve_experiments(query="goal", k=1)
    assert results[0]["traceback"] == "error: patch failed: candidate_script.py:1"
    assert results[0]["failure_reason"] == "Malformed diff rejected by git apply."


def test_db_traceback_defaults_to_empty_string_not_none(tmp_dir):
    """ChromaDB metadata values must be str/int/float/bool - None is invalid, same reason failure_reason already avoids it."""
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="goal", diff="d", rationale="r", metrics={}, outcome="success")

    results = db.retrieve_experiments(query="goal", k=1)
    assert results[0]["traceback"] == ""


def test_prompt_builder_feeds_the_real_traceback_not_just_the_category_label(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(
        hypothesis="improve accuracy", diff="some diff", rationale="r", metrics={}, outcome="failure",
        failure_reason="Malformed diff rejected by git apply.",
        traceback="error: patch failed: candidate_script.py:3\nerror: candidate_script.py: patch does not apply",
    )

    prompt = PromptBuilder(db, {}).build_prompt("improve accuracy")

    assert "Traceback:" in prompt
    assert "patch does not apply" in prompt


def test_prompt_builder_omits_traceback_section_when_none_was_recorded(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(
        hypothesis="improve accuracy", diff="some diff", rationale="r", metrics={}, outcome="failure",
        failure_reason="Candidate failed proxy evaluation thresholds.",
    )

    prompt = PromptBuilder(db, {}).build_prompt("improve accuracy")

    assert "Traceback:" not in prompt


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


class _AlwaysMalformedDiffClient(LLMClient):
    """Always returns a diff whose context lines don't match candidate_script.py's real content."""
    def generate_diff(self, prompt, target_file, current_content=""):
        return "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-this will never match\n+x\n"


def test_evolutionary_mode_exhausted_malformed_generation_records_the_real_diagnostic(tmp_dir):
    """
    Full integration: a candidate that only ever produces a malformed diff
    exhausts its retries - the resulting failure record must carry the
    real git-apply diagnostic from the last attempt, not just the generic
    "exhausted malformed retries" label.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    prompt_builder = PromptBuilder(db, {})
    patch_generator = PatchGenerator(_AlwaysMalformedDiffClient())

    config = {"evolution": {"population_size": 1, "max_generations": 1}, "eval": {"stages": []}}
    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=None,
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=patch_generator,
        prompt_builder=prompt_builder,
        db=db,
    )

    population = engine._generate_population("goal", n=1)

    assert population == []
    assert engine.generation_failed_count == 1

    results = db.retrieve_experiments(query="goal", k=10)
    exhausted = [r for r in results if r.get("outcome") == "failure"]
    assert len(exhausted) == 1
    assert exhausted[0]["traceback"]  # the real git diagnostic, not empty
