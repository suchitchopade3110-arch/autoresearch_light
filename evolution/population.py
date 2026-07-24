import uuid
import json
import random
from typing import List, Dict, Any
from generation.prompt_builder import PromptBuilder
from generation.patch_generator import PatchGenerator, validate_and_apply_patch
from memory.db import ExperimentDB
from evolution.duplicate_checker import is_duplicate
from evolution.scheduler import ConcurrentScheduler
from evolution.scoring import score_candidates
from evolution.reporting import log_generation_report

class EvolutionEngine:
    def __init__(self, config: Dict[str, Any], git_controller, sandbox, evaluator, metrics_calculator, failure_analyzer, patch_generator: PatchGenerator, prompt_builder: PromptBuilder, db: ExperimentDB, approval_store=None, truth=None, baseline_store=None):
        self.config = config.get('evolution', {})
        self.full_config = config
        self.eval_config = config.get('eval', {})
        self.git_controller = git_controller
        self.sandbox = sandbox
        self.evaluator = evaluator
        self.metrics_calculator = metrics_calculator
        self.failure_analyzer = failure_analyzer
        self.patch_generator = patch_generator
        self.prompt_builder = prompt_builder
        self.db = db
        self.approval_store = approval_store
        self.truth = truth or {}
        self.baseline_store = baseline_store

        self.pop_size = self.config.get('population_size', 5)
        self.max_gens = self.config.get('max_generations', 3)
        self.scheduler = ConcurrentScheduler(self.config.get('max_concurrent_sandboxes', 3))
        # a dedicated instance, not the global random module, so selection is
        # reproducible given the same seed without affecting anything else
        # that happens to use `random` in-process.
        self.rng = random.Random(self.config.get('random_seed', 42))

        self.best_scores = []
        self.duplicate_avoidance_count = 0

    def _generate_candidate(self, goal: str, mutation_context: str = "") -> Dict[str, Any]:
        max_retries = 3
        for _ in range(max_retries):
            prompt = self.prompt_builder.build_prompt(goal)
            if mutation_context:
                prompt += f"\nMutation Instruction: {mutation_context}"

            diff = self.patch_generator.llm_client.generate_diff(prompt, "candidate_script.py")

            # dry_run: only check applicability here, don't mutate the
            # shared checkout - the scheduler applies it for real, once,
            # inside the candidate's own worktree.
            if not validate_and_apply_patch(diff, dry_run=True):
                continue

            dup_threshold = self.config.get('duplicate_threshold', 0.25)
            if not is_duplicate(diff, self.db, dup_threshold, hypothesis=goal):
                return {
                    'id': uuid.uuid4().hex[:8],
                    'diff': diff,
                    'goal': goal
                }
            else:
                self.duplicate_avoidance_count += 1

        return {
            'id': uuid.uuid4().hex[:8],
            'diff': f"--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('Fallback {uuid.uuid4().hex[:4]}')\n",
            'goal': goal
        }

    def _select_parents(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        strategy = self.config.get('selection_strategy', 'tournament')

        sorted_cands = sorted(candidates, key=lambda x: x.get('composite_score', -float('inf')), reverse=True)

        if strategy == 'top-k':
            k = max(1, len(sorted_cands) // 2)
            return sorted_cands[:k]
        elif strategy == 'tournament':
            parents = []
            k = max(1, len(sorted_cands) // 2)
            for _ in range(k):
                tournament = self.rng.sample(sorted_cands, min(3, len(sorted_cands)))
                best = max(tournament, key=lambda x: x.get('composite_score', -float('inf')))
                parents.append(best)
            return parents
        else:
            return sorted_cands[:1]

    def _adaptive_sizing(self) -> bool:
        sizing = self.config.get('adaptive_sizing', {})
        if not sizing.get('enabled', False):
            return False

        conv_gens = sizing.get('convergence_generations', 2)
        if len(self.best_scores) >= conv_gens + 1:
            recent_best = self.best_scores[-conv_gens:]
            prev_best = self.best_scores[-(conv_gens+1)]

            if all(abs(b - prev_best) < 0.01 for b in recent_best):
                self.pop_size = max(sizing.get('min_population', 2), self.pop_size - 1)
                return True
            else:
                self.pop_size = min(sizing.get('max_population', 10), self.pop_size + 1)
        return False

    def run(self, goal: str):
        print(f"Initializing Population of {self.pop_size}...")
        current_generation = [self._generate_candidate(goal) for _ in range(self.pop_size)]

        for gen in range(self.max_gens):
            self.duplicate_avoidance_count = 0
            print(f"\n=== Running Generation {gen + 1}/{self.max_gens} ===")

            evaluated = self.scheduler.execute_generation(
                current_generation,
                self.eval_config.get('stages', []),
                self.git_controller,
                self.sandbox,
                self.evaluator,
                self.metrics_calculator,
                self.failure_analyzer,
                approval_store=self.approval_store,
                approval_config=self.full_config,
                truth=self.truth,
                baseline_store=self.baseline_store,
            )

            scored = score_candidates(evaluated, self.config)

            for c in scored:
                if c['success']:
                    outcome = "success"
                elif c.get('eval_passed'):
                    outcome = "held"  # passed evaluation but not approved for merge
                else:
                    outcome = "failure"

                self.db.store_experiment(
                    hypothesis=goal,
                    diff=c['diff'],
                    rationale=f"Generation {gen} candidate",
                    metrics=c['metrics'],
                    outcome=outcome,
                    failure_reason=c.get('error_msg', None)
                )

            valid_scores = [c['composite_score'] for c in scored if 'composite_score' in c]
            best_score = max(valid_scores) if valid_scores else 0.0
            worst_score = min(valid_scores) if valid_scores else 0.0
            self.best_scores.append(best_score)

            total_time = sum(c.get('total_execution_time', 0.0) for c in scored)

            convergence_signal = self._adaptive_sizing()

            log_generation_report(
                generation=gen + 1,
                scored_candidates=scored,
                best_score=best_score,
                worst_score=worst_score,
                compute_time_spent=total_time,
                convergence_signal=convergence_signal,
                duplicate_avoidance_count=self.duplicate_avoidance_count
            )

            if gen == self.max_gens - 1:
                break

            parents = self._select_parents(scored)

            next_generation = []

            if parents:
                elite = dict(parents[0])
                elite['id'] = uuid.uuid4().hex[:8]
                next_generation.append(elite)

            while len(next_generation) < self.pop_size:
                parent = self.rng.choice(parents) if parents else None
                mutation_context = f"Vary the approach used in diff:\n{parent['diff']}" if parent else ""
                next_generation.append(self._generate_candidate(goal, mutation_context))

            current_generation = next_generation