import concurrent.futures
import os
import threading
from typing import Any, Dict, List, Optional

from approval.gate import await_approval_decision, create_approval_request, maybe_auto_approve
from approval.store import ApprovalStore
from generation.patch_generator import validate_and_apply_patch
from observability.logging_config import bind, get_logger
from vcs.git_controller import MergeConflict


class ConcurrentScheduler:
    def __init__(self, max_workers: int, logger=None):
        self.max_workers = max_workers
        # Each candidate now gets its own git worktree, so patch application,
        # commit, and sandbox execution never share a checkout - only the
        # repo-level branch/merge/rollback calls below still touch the
        # single shared git_controller.repo object and need serializing.
        self.git_lock = threading.Lock()
        self.logger = logger or get_logger(__name__)

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
        """
        Two phases, so a human is never a bottleneck on the sandbox pool:

        Phase 1 (bounded by max_workers, no human in the loop): apply each
        candidate's patch and run it through every eval stage. A candidate
        that fails evaluation is rolled back immediately here - it will
        never need approval.

        Phase 2 (unbounded, serial): every candidate that passed
        evaluation gets its approval request created up front, all at
        once, so a reviewer sees the whole generation together instead of
        candidates trickling in one at a time as sandbox slots free up.
        Only then are decisions awaited and candidates merged/rolled back.
        """
        # Fail safe: if the caller didn't wire a store, still gate merges
        # rather than silently skipping approval - only an explicit, valid
        # approval.enabled: false in approval_config actually disables it.
        store = approval_store or ApprovalStore()
        gate_config = approval_config or {}
        truth = truth or {}

        def evaluate_only(candidate: Dict[str, Any]) -> Dict[str, Any]:
            """
            Phase 1 body: apply, commit, evaluate. Never touches approval or
            merge.

            KNOWN LIMITATION: hardcoded to "candidate_script.py", like
            evolution/population.py's candidate generation - evolutionary
            mode does not support target.files multi-file candidates yet.
            """
            c_id = candidate['id']
            diff = candidate['diff']
            candidate_logger = bind(self.logger, candidate_id=c_id)

            branch_name = candidate.get('branch_name')
            worktree_path = candidate.get('worktree_path')

            try:
                if not worktree_path:
                    # No pre-created worktree (e.g. an elite carryover) -
                    # create one now, same as before candidates got their
                    # own worktree at generation time.
                    with self.git_lock:
                        branch_name, worktree_path = git_controller.create_branch(c_id)
                # else: the worktree was already created (and the diff's
                # dry-run already checked) at generation time, against this
                # exact current file content - see evolution/population.py.

                script_path = os.path.join(worktree_path, "candidate_script.py")
                if not os.path.exists(script_path):
                    with open(script_path, "w") as f:
                        f.write("\n")

                if diff.strip():
                    if not validate_and_apply_patch(diff, cwd=worktree_path, logger=candidate_logger):
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
                has_failure_flags = False

                for stage in eval_stages:
                    subset = stage['subset_percentage']
                    threshold = stage['threshold']

                    env = {"SUBSET_PERCENTAGE": str(subset)}
                    exec_result = sandbox.run_candidate(script_path, env_vars=env, out_dir=out_dir)
                    total_execution_time += exec_result.get('execution_time', 0.0)

                    exec_result['execution_time'] = total_execution_time

                    stage_success, stage_score = evaluator.evaluate_stage(
                        exec_result, subset, threshold, pred_path, truth, logger=candidate_logger
                    )
                    # Read immediately after the call - evaluator is a
                    # single shared EvalPipeline instance across every
                    # candidate/stage (see eval/pipeline.py:last_stage_flags).
                    has_failure_flags = has_failure_flags or bool(evaluator.last_stage_flags.get("score_claim_mismatch", False))
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
                # Consumed by approval/gate.py's should_auto_approve as the
                # require_no_failure_flags criterion - see
                # orchestrator/run.py's sequential-path equivalent.
                all_metrics['score_claim_mismatch'] = has_failure_flags
                generation_usage = candidate.get('generation_usage') or {}
                if generation_usage:
                    all_metrics['generation_input_tokens'] = generation_usage.get('input_tokens', 0)
                    all_metrics['generation_output_tokens'] = generation_usage.get('output_tokens', 0)
                    all_metrics['generation_cost_usd'] = generation_usage.get('estimated_cost_usd', 0.0)

                candidate['branch_name'] = branch_name
                candidate['worktree_path'] = worktree_path
                candidate['eval_passed'] = eval_passed
                candidate['final_score'] = final_score
                candidate['metrics'] = all_metrics
                candidate['failure_category'] = failure_category
                candidate['error_msg'] = error_msg
                candidate['total_execution_time'] = total_execution_time
                candidate['last_subset'] = last_subset
                candidate['success'] = False
                candidate['approval_decision'] = None

                if not eval_passed:
                    # Never needs approval - roll back now rather than
                    # carrying a dead candidate into phase 2.
                    with self.git_lock:
                        git_controller.rollback(branch_name, worktree_path)

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
                candidate['final_score'] = 0.0
                candidate['total_execution_time'] = 0.0
                return candidate

        # Phase 1 - bounded by max_workers, no human in the loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(evaluate_only, c) for c in candidates]
            evaluated = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Phase 2 - enqueue the whole generation's approval requests first,
        # so a reviewer sees them all together, then drain serially.
        passed = [c for c in evaluated if c.get('eval_passed')]
        for c in passed:
            request_id = create_approval_request(
                store, c['id'], c.get('goal', 'Optimization goal'), c['diff'], c['final_score'], c['metrics'], gate_config
            )
            c['_approval_request_id'] = request_id
            if request_id is None:
                c['approval_decision'] = 'skipped'
            else:
                c['approval_decision'] = maybe_auto_approve(
                    store, request_id, gate_config,
                    c['metrics'].get('delta'), bool(c['metrics'].get('score_claim_mismatch')),
                )

        for c in passed:
            if c['approval_decision'] is None:
                c['approval_decision'] = await_approval_decision(store, c['_approval_request_id'], gate_config)
            decision = c['approval_decision']

            with self.git_lock:
                if decision in ("approved", "auto_approved", "skipped"):
                    try:
                        git_controller.merge(c['branch_name'], c['worktree_path'])
                        c['success'] = True
                        if baseline_store and c.get('last_subset') is not None:
                            baseline_store.update_if_better(c['last_subset'], c['final_score'])
                    except MergeConflict as e:
                        # Rebasing onto the current base failed - routine
                        # traffic when several candidates merge into the
                        # same base per generation, not a code failure.
                        # Distinct from "failure" so the population's
                        # staleness is visible in reporting.
                        git_controller.rollback(c['branch_name'], c['worktree_path'])
                        c['eval_passed'] = False
                        c['failure_category'] = "conflict"
                        c['error_msg'] = str(e)
                else:
                    git_controller.rollback(c['branch_name'], c['worktree_path'])
                    c['failure_category'] = "held"
                    c['error_msg'] = f"approval_decision={decision}"

            c.pop('_approval_request_id', None)

        return evaluated
