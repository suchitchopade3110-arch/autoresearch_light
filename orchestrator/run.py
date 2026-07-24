import argparse
import os
import sys
import uuid

import yaml

from vcs.git_controller import GitController
from sandbox.executor import SandboxExecutor
from eval.dataset import generate_split, load_truth
from eval.baseline import BaselineStore
from eval.pipeline import EvalPipeline
from orchestrator.metrics import calculate_all_metrics

# Phase 2 imports
from memory.db import ExperimentDB
from memory.failure_analysis import analyze_failure
from generation.prompt_builder import PromptBuilder
from generation.patch_generator import PatchGenerator, MockLLMClient
from generation.static_check import check_syntax

# Phase 4 imports
from approval.store import ApprovalStore
from approval.gate import request_and_await_approval
from reporting.report_generator import generate_report


def main():
    parser = argparse.ArgumentParser(description="Run the core loop")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--goal", default="Improve the mock candidate script performance", help="The research goal")
    parser.add_argument("--mode", default="sequential", choices=["sequential", "evolutionary"], help="Mode to run the orchestrator in")
    parser.add_argument("--max-iterations", type=int, default=None, help="Sequential mode only: cap on candidates to try")
    parser.add_argument("--target-score", type=float, default=None, help="Sequential mode only: stop early once reached")
    parser.add_argument("--patience", type=int, default=None, help="Sequential mode only: stop after this many iterations with no improvement")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    orch_cfg = config.get('orchestrator', {})
    dataset_cfg = config.get('dataset', {})
    eval_cfg = config.get('eval', {})

    dataset_dir = dataset_cfg.get('path', 'dummy_data')
    dataset_paths = generate_split(
        dataset_dir,
        n=dataset_cfg.get('size', 1000),
        seed=dataset_cfg.get('seed', 42),
        test_frac=dataset_cfg.get('test_frac', 0.25),
    )
    # truth.json is loaded host-side only - it is never mounted into the
    # sandbox (see sandbox/executor.py), so a candidate can never read its
    # own answer key off disk.
    truth = load_truth(dataset_paths['truth'])

    # Initialize components
    vcs = GitController()
    sandbox = SandboxExecutor(config.get('sandbox', {}), dataset_dir=dataset_dir)
    evaluator = EvalPipeline(eval_cfg)
    baseline_store = BaselineStore(eval_cfg.get('state_path', 'state.json'))
    min_improvement = eval_cfg.get('min_improvement', 0.001)

    # Initialize Phase 2 components
    db = ExperimentDB()
    prompt_builder = PromptBuilder(db, config.get('generation', {}))
    llm_client = MockLLMClient()
    patch_generator = PatchGenerator(llm_client)

    # Phase 4: human-approval gate. resolve_approval_config (inside the gate)
    # defaults to "required" for any missing/malformed approval config - see
    # approval/gate.py.
    approval_store = ApprovalStore(config.get('approval', {}).get('db_path', 'approvals.db')
                                    if isinstance(config.get('approval'), dict) else 'approvals.db')

    if args.mode == "evolutionary":
        from evolution.population import EvolutionEngine
        engine = EvolutionEngine(
            config=config,
            git_controller=vcs,
            sandbox=sandbox,
            evaluator=evaluator,
            metrics_calculator=calculate_all_metrics,
            failure_analyzer=analyze_failure,
            patch_generator=patch_generator,
            prompt_builder=prompt_builder,
            db=db,
            approval_store=approval_store,
            truth=truth,
            baseline_store=baseline_store,
        )
        engine.run(args.goal)
        generate_report(db, approval_store)
        sys.exit(0)

    eval_stages = config.get('eval', {}).get('stages', [])

    def run_iteration(candidate_id: str, goal: str):
        print(f"--- Starting iteration for candidate {candidate_id} ---")

        # 1. VCS branching - each candidate gets its own worktree, so this
        # never touches the caller's main checkout.
        branch_name, worktree_path = vcs.create_branch(candidate_id)
        print(f"Created branch {branch_name} (worktree: {worktree_path})")

        script_path = os.path.join(worktree_path, "candidate_script.py")

        # 2. Phase 2 Generation - apply the patch inside the candidate's own
        # worktree (cwd=worktree_path), not the shared main checkout.
        print("Building prompt...")
        prompt = prompt_builder.build_prompt(goal)

        print("Generating and applying patch...")
        apply_success, diff = patch_generator.generate_and_apply(prompt, "candidate_script.py", cwd=worktree_path)

        if not apply_success:
            print("Patch application failed (malformed diff). Rejecting candidate.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated malformed diff",
                metrics={},
                outcome="failure",
                failure_reason="Malformed diff rejected by git apply."
            )
            vcs.rollback(branch_name, worktree_path)
            return False, 0.0

        vcs.commit_patch(worktree_path, f"Add candidate {candidate_id}")

        # 3. Static Analysis Pre-check
        print("Running static analysis...")
        syntax_ok, syntax_err = check_syntax(script_path)
        if not syntax_ok:
            print(f"Static check failed: {syntax_err}")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Prompt generated syntax error",
                metrics={},
                outcome="failure",
                failure_reason=syntax_err
            )
            vcs.rollback(branch_name, worktree_path)
            return False, 0.0

        # 4. Execute in sandbox once per progressive-scaling stage, so each
        # stage's score reflects that stage's own dataset subset. Each run
        # writes predictions.jsonl to out_dir - that, scored against
        # held-out truth, is the only real score; anything the candidate
        # prints is a diagnostic at best (see eval/pipeline.py).
        out_dir = os.path.join(worktree_path, ".eval_out")
        pred_path = os.path.join(out_dir, "predictions.jsonl")

        eval_passed = True
        final_score = 0.0
        last_subset = None
        metrics = {}
        execution_result = None

        for stage in eval_stages:
            subset = stage['subset_percentage']
            threshold = stage['threshold']

            print(f"Running in sandbox (subset={subset}%)...")
            execution_result = sandbox.run_candidate(
                script_path, env_vars={"SUBSET_PERCENTAGE": str(subset)}, out_dir=out_dir
            )

            if execution_result['timeout']:
                print("Execution TIMED OUT")

            metrics = calculate_all_metrics(execution_result)
            print(f"Plugin Metrics: {metrics}")

            stage_success, final_score = evaluator.evaluate_stage(execution_result, subset, threshold, pred_path, truth)
            last_subset = subset
            if not stage_success:
                eval_passed = False
                break

        if eval_passed:
            print("Candidate passed all evaluation stages.")

        # 4b. Baseline gate - clearing every stage's absolute threshold is
        # not enough to merge; the final stage's score must also beat the
        # best score ever actually merged for that stage. Without this, a
        # candidate that regresses relative to what's already in place can
        # still merge as long as it clears the (fixed) threshold.
        baseline_score = baseline_store.get(last_subset) if last_subset is not None else 0.0
        delta = final_score - baseline_score
        below_baseline = False
        if eval_passed and last_subset is not None:
            if not baseline_store.passes(last_subset, final_score, min_improvement):
                below_baseline = True
                eval_passed = False
                print(
                    f"Candidate {candidate_id} scored {final_score:.4f} at {last_subset}%, which does not beat "
                    f"baseline {baseline_score:.4f} + min_improvement {min_improvement}. Rejecting despite "
                    f"clearing the absolute threshold."
                )

        metrics['baseline_score'] = baseline_score
        metrics['delta'] = delta

        # 5. Analyze failure and log to Memory
        if below_baseline:
            category, error_text = "below_baseline", (
                f"score {final_score:.4f} did not beat baseline {baseline_score:.4f} + {min_improvement}"
            )
        else:
            category, error_text = analyze_failure(execution_result, eval_passed)

        merged = False
        if not eval_passed:
            print(f"Candidate {candidate_id} failed ({category}). Rolling back.")
            db.store_experiment(
                hypothesis=goal,
                diff=diff,
                rationale="Generated patch failed",
                metrics=metrics,
                outcome="failure",
                failure_reason=error_text
            )
            vcs.rollback(branch_name, worktree_path)
        else:
            # 6. Human-approval gate - genuinely blocks the merge path.
            # Only "approved" or "skipped" (gate explicitly disabled) may
            # proceed to merge; "rejected" and "timed_out" roll back.
            print(f"Candidate {candidate_id} passed evaluation with score {final_score:.4f}. Awaiting approval...")
            decision = request_and_await_approval(
                approval_store, candidate_id, goal, diff, final_score, metrics, config
            )

            if decision in ("approved", "skipped"):
                print(f"Candidate {candidate_id} approved ({decision}). Merging.")
                db.store_experiment(
                    hypothesis=goal,
                    diff=diff,
                    rationale="Generated patch passed evaluation and approval",
                    metrics=metrics,
                    outcome="success"
                )
                vcs.merge(branch_name, worktree_path)
                merged = True
                if last_subset is not None:
                    baseline_store.update_if_better(last_subset, final_score)
            else:
                print(f"Candidate {candidate_id} was not merged (approval decision: {decision}). Rolling back.")
                db.store_experiment(
                    hypothesis=goal,
                    diff=diff,
                    rationale="Generated patch passed evaluation but was not approved",
                    metrics=metrics,
                    outcome="held",
                    failure_reason=f"approval_decision={decision}"
                )
                vcs.rollback(branch_name, worktree_path)

        print(f"--- Finished iteration for candidate {candidate_id} ---\n")
        return merged, final_score

    max_iterations = args.max_iterations or orch_cfg.get('max_iterations', 1)
    target_score = args.target_score if args.target_score is not None else orch_cfg.get('target_score', 1.0)
    patience = args.patience or orch_cfg.get('patience', max_iterations)

    best_score = 0.0
    any_success = False
    iterations_since_improvement = 0

    for i in range(max_iterations):
        candidate_id = uuid.uuid4().hex[:8]
        success, final_score = run_iteration(candidate_id, args.goal)

        if success:
            any_success = True
        if final_score > best_score:
            best_score = final_score
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        if any_success and best_score >= target_score:
            print(f"Target score {target_score} reached (best={best_score:.4f}). Stopping.")
            break
        if iterations_since_improvement >= patience:
            print(f"No improvement in {patience} iterations. Stopping early.")
            break

    generate_report(db, approval_store)

    if not any_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
