import concurrent.futures
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from approval.gate import request_and_await_approval
from approval.store import ApprovalStore
from generation.patch_generator import validate_and_apply_patch


class ConcurrentScheduler:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        # Each candidate now gets its own git worktree, so patch application,
        # commit, and sandbox execution never share a checkout - only the
        # repo-level branch/merge/rollback calls below still touch the
        # single shared git_controller.repo object and need serializing.
        self.git_lock = threading.Lock()

    def execute_generation(self,
                          candidates: List[Dict[str, Any]],
                          eval_stages: List[Dict[str, Any]],
                          git_controller,
                          sandbox,
                          evaluator,
                          metrics_calculator,
                          failure_analyzer,
                          approval_store: Optional[ApprovalStore] = None,
                          approval_config: Optional[Dict[str, Any]] = None,
                          truth: Optional[Dict[str, int]] = None,
                          baseline_store=None) -> List[Dict[str, Any]]:
        results = []
        # Fail safe: if the caller didn't wire a store, still gate merges
        # rather than silently skipping approval - only an explicit, valid
        # approval.enabled: false in approval_config actually disables it.
        store = approval_store or ApprovalStore()
        gate_config = approval_config or {}
        truth = truth or {}

        def process_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            c_id = candidate['id']
            diff = candidate['diff']
            goal = candidate.get('goal', 'Optimization goal')

            branch_name = None
            worktree_path = None

            try:
                with self.git_lock:
                    branch_name, worktree_path = git_controller.create_branch(c_id)

                script_path = os.path.join(worktree_path, "candidate_script.py")
                if not os.path.exists(script_path):
                    with open(script_path, "w") as f:
                        f.write("\n")

                if diff.strip():
                    if not validate_and_apply_patch(diff, cwd=worktree_path):
                        raise RuntimeError(f"Patch failed to apply for candidate {c_id}")

                git_controller.commit_patch(worktree_path, f"Add candidate {c_id}")

                out_dir = os.path.join(worktree_path, ".eval_out")
                pred_path = os.path.join(out_dir, "predictions.jsonl")

                final_score = 0.0
                all_metrics = {}
                eval_passed = True
                failure_category = "success"
                error_msg = ""
                total_execution_time = 0.0
                last_subset = None

                for stage in eval_stages:
                    subset = stage['subset_percentage']
                    threshold = stage['threshold']

                    env = {"SUBSET_PERCENTAGE": str(subset)}
                    exec_result = sandbox.run_candidate(script_path, env_vars=env, out_dir=out_dir)
                    total_execution_time += exec_result.get('execution_time', 0.0)

                    exec_result['execution_time'] = total_execution_time

                    stage_success, stage_score = evaluator.evaluate_stage(exec_result, subset, threshold, pred_path, truth)
                    last_subset = subset

                    if not stage_success:
                        eval_passed = False
                        final_score = stage_score
                        cat, msg = failure_analyzer(exec_result, False)
                        failure_category = cat
                        error_msg = msg
                        all_metrics = metrics_calculator(exec_result)
                        break

                    final_score = stage_score
                    all_metrics = metrics_calculator(exec_result)

                # Baseline gate - see orchestrator/run.py for the rationale:
                # clearing every stage's absolute threshold isn't enough: the
                # final stage's score must also beat the best score ever
                # actually merged for that stage.
                baseline_score = baseline_store.get(last_subset) if (baseline_store and last_subset is not None) else 0.0
                delta = final_score - baseline_score
                if eval_passed and baseline_store and last_subset is not None:
                    eval_section = gate_config.get('eval')
                    min_improvement = eval_section.get('min_improvement', 0.001) if isinstance(eval_section, dict) else 0.001
                    if not baseline_store.passes(last_subset, final_score, min_improvement):
                        eval_passed = False
                        failure_category = "below_baseline"
                        error_msg = f"score {final_score:.4f} did not beat baseline {baseline_score:.4f} + {min_improvement}"

                all_metrics['baseline_score'] = baseline_score
                all_metrics['delta'] = delta

                merged = False
                approval_decision = None
                if eval_passed:
                    # Each candidate polls its own approval request - this
                    # blocks only this worker thread, so sibling candidates
                    # in the same generation are unaffected while it waits.
                    approval_decision = request_and_await_approval(
                        store, c_id, goal, diff, final_score, all_metrics, gate_config
                    )

                with self.git_lock:
                    if eval_passed and approval_decision in ("approved", "skipped"):
                        git_controller.merge(branch_name, worktree_path)
                        merged = True
                        if baseline_store and last_subset is not None:
                            baseline_store.update_if_better(last_subset, final_score)
                    else:
                        git_controller.rollback(branch_name, worktree_path)
                        if eval_passed:
                            failure_category = "held"
                            error_msg = f"approval_decision={approval_decision}"

                candidate['success'] = merged
                candidate['eval_passed'] = eval_passed
                candidate['approval_decision'] = approval_decision
                candidate['final_score'] = final_score
                candidate['metrics'] = all_metrics
                candidate['failure_category'] = failure_category
                candidate['error_msg'] = error_msg
                candidate['total_execution_time'] = total_execution_time
                return candidate

            except Exception as e:
                with self.git_lock:
                    if branch_name and worktree_path:
                        try:
                            active_branches = [h.name for h in git_controller.repo.heads]
                        except AttributeError:
                            active_branches = git_controller.repo.heads.keys()
                        if branch_name in active_branches:
                            git_controller.rollback(branch_name, worktree_path)
                candidate['success'] = False
                candidate['eval_passed'] = False
                candidate['approval_decision'] = None
                candidate['failure_category'] = "runtime"
                candidate['error_msg'] = str(e)
                candidate['metrics'] = {}
                candidate['total_execution_time'] = 0.0
                return candidate

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_candidate, c) for c in candidates]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return results
