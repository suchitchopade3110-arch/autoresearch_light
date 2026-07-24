import os
import subprocess
from unittest.mock import MagicMock

from evolution.population import EvolutionEngine
from generation.patch_generator import PatchGenerator, LLMClient
from generation.prompt_builder import PromptBuilder
from memory.db import ExperimentDB
from orchestrator.metrics import calculate_all_metrics
from memory.failure_analysis import analyze_failure
from vcs.git_controller import GitController

DUPLICATE_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('always the same')\n"


class _AlwaysSameDiffClient(LLMClient):
    """A stub LLM client that always returns the identical diff - guaranteed to be flagged a duplicate once one copy is on record."""
    def generate_diff(self, prompt, target_file, current_content=""):
        return DUPLICATE_DIFF


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


def test_duplicate_exhaustion_produces_a_record_and_never_touches_the_sandbox(tmp_dir):
    """
    Wave 2 acceptance: when every retry for a candidate slot is rejected as
    a duplicate, the engine must record a distinct duplicate_exhausted
    outcome and skip the candidate entirely - never substituting a
    scoreless fallback diff that would otherwise burn a full
    sandbox x stages budget on something that cannot score.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    # Pre-seed the exact diff the stub client will always return, so every
    # attempt is an exact-match duplicate from the very first retry.
    db.store_experiment(hypothesis="goal", diff=DUPLICATE_DIFF, rationale="seed", metrics={}, outcome="success")

    prompt_builder = PromptBuilder(db, {})
    patch_generator = PatchGenerator(_AlwaysSameDiffClient())
    sandbox = MagicMock()

    config = {
        "evolution": {"population_size": 2, "max_generations": 1, "duplicate_threshold": 0.25},
        "eval": {"stages": []},
    }

    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=sandbox,
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=patch_generator,
        prompt_builder=prompt_builder,
        db=db,
    )

    population = engine._generate_population("goal", n=2)

    assert population == []
    assert engine.duplicate_exhausted_count == 2
    sandbox.run_candidate.assert_not_called()

    results = db.retrieve_experiments(query="goal", k=10)
    exhausted = [r for r in results if r.get("outcome") == "duplicate_exhausted"]
    assert len(exhausted) == 2


def test_generate_candidate_creates_and_cleans_up_worktree_on_exhaustion(tmp_dir):
    """
    Wave 2.3: _generate_candidate creates the candidate's worktree before
    generating (so the diff is checked against real current content), and
    must roll it back if generation is ultimately exhausted - it must not
    leak a worktree/branch for a candidate that never gets scheduled.
    """
    repo_dir = _init_repo(tmp_dir)
    git_controller = GitController(repo_path=repo_dir, worktree_root=os.path.join(tmp_dir, "worktrees"))

    db = ExperimentDB(db_path=os.path.join(tmp_dir, "chroma"))
    db.store_experiment(hypothesis="goal", diff=DUPLICATE_DIFF, rationale="seed", metrics={}, outcome="success")

    prompt_builder = PromptBuilder(db, {})
    patch_generator = PatchGenerator(_AlwaysSameDiffClient())

    config = {"evolution": {"duplicate_threshold": 0.25}, "eval": {"stages": []}}
    engine = EvolutionEngine(
        config=config,
        git_controller=git_controller,
        sandbox=MagicMock(),
        evaluator=None,
        metrics_calculator=calculate_all_metrics,
        failure_analyzer=analyze_failure,
        patch_generator=patch_generator,
        prompt_builder=prompt_builder,
        db=db,
    )

    result = engine._generate_candidate("goal")

    assert result is None
    assert os.listdir(os.path.join(tmp_dir, "worktrees")) == []
    assert list(git_controller.repo.heads) == [git_controller.repo.heads[git_controller.original_branch]]
