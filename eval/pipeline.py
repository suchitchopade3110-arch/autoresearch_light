# =============================================================================
# HARD INVARIANT - read this in full before changing anything in this file,
# eval/dataset.py, or sandbox/executor.py:
#
#   1. truth.json (the held-out labels) is NEVER mounted into the sandbox.
#      It is loaded host-side only, via eval/dataset.py:load_truth(), and
#      passed into this module as an in-memory dict. No code path here (or
#      in sandbox/executor.py) may ever construct a docker mount argument
#      referencing truth.json, or make it readable from inside a container.
#      truth.json lives on disk in the SAME directory as train.jsonl/
#      test.jsonl (see eval/dataset.py:generate_split) - the only reason it
#      stays hidden is that sandbox/executor.py mounts individual files
#      (`-v host_path:container_path:ro`), never the whole directory. If
#      that ever changes to a directory-level mount, this invariant breaks
#      silently.
#
#   2. score_predictions() is the ONLY source of truth for a candidate's
#      score. Nothing a candidate writes to stdout/stderr is ever trusted
#      for gating - see _parse_score(), which exists purely to detect a
#      mismatch (a candidate lying about its own performance), never to
#      substitute for a real score.
#
# Both properties are enforced by permanent regression tests that must
# never be deleted, skipped, or weakened:
#   - tests/test_reward_hacking.py::test_printed_score_claim_is_never_trusted
#   - tests/test_reward_hacking.py::test_truth_json_never_referenced_by_the_sandbox_executor
#   - tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide
#   - tests/test_sandbox.py::test_truth_json_unreachable_by_any_path_inside_the_sandbox
#     (walks the ENTIRE container filesystem for a file literally named
#     truth.json - proof that no path construction trick, not just the one
#     obvious path, can ever reach it)
#
# If you're about to mount a whole directory (rather than individual
# files) into the sandbox, or change how/where truth is loaded, stop and
# re-run every test above first.
# =============================================================================

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from observability.logging_config import get_logger

SCORE_PATTERN = re.compile(r"SCORE:\s*([-+]?\d*\.?\d+)")
_module_logger = get_logger(__name__)
SCORE_CLAIM_MISMATCH_THRESHOLD = 0.05


class EvalPipeline:
    """
    Progressive-scaling evaluator. The candidate's own stdout is never
    trusted for the score that gates a merge - score_predictions() scores
    real predictions against held-out truth the candidate never saw. A
    printed "SCORE:" line (if any) is parsed only as a diagnostic, to
    detect a candidate lying about its own performance.
    """
    def __init__(self, config: Dict[str, Any]):
        self.stages = config.get('stages', [])
        self.correlation_log = []
        # Set by evaluate_stage() for the stage just evaluated - callers
        # that need this (see approval/gate.py's auto-approve criteria)
        # must read it immediately after the call, the same way
        # AnthropicClient.last_usage is read immediately after
        # generate_diff (see generation/patch_generator.py), since this is
        # a single shared EvalPipeline instance across every candidate and
        # stage, not per-call state.
        self.last_stage_flags: Dict[str, Any] = {}

    def score_predictions(self, pred_path: str, truth: Dict[str, int]) -> Tuple[float, str]:
        """
        Scores predictions written by the candidate against held-out
        truth. Returns (score, reason) - reason is "" on a clean score,
        otherwise a human-readable explanation of why the score is 0.0.
        Never raises: a missing file, malformed JSON, a wrong id set, or a
        non-0/1 prediction all score 0.0 with a reason instead of an
        exception escaping to the caller.
        """
        if not os.path.exists(pred_path):
            return 0.0, "no predictions written"

        preds: Dict[str, int] = {}
        try:
            with open(pred_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    preds[str(row["id"])] = int(row["pred"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            return 0.0, f"malformed predictions file: {e}"

        if set(preds) != set(truth):
            return 0.0, f"prediction id set mismatch ({len(preds)} vs {len(truth)})"

        if not all(v in (0, 1) for v in preds.values()):
            return 0.0, "predictions must be 0 or 1"

        correct = sum(1 for i, y in truth.items() if preds[i] == y)
        return correct / len(truth), ""

    def evaluate_stage(self, execution_result: Dict[str, Any], subset_percentage: int, threshold: float,
                        pred_path: str, truth: Dict[str, int], logger=None) -> Tuple[bool, float]:
        """
        Evaluates a single stage's real execution result. Returns (success, score).
        logger defaults to a module-level logger with no run/candidate context -
        callers that have it (orchestrator/run.py, evolution/scheduler.py) should
        pass a logger already bound with candidate_id so these records can be
        correlated back to the candidate they're about.
        """
        log = logger or _module_logger
        if execution_result['exit_code'] != 0 or execution_result.get('timeout', False):
            return False, 0.0

        score, reason = self.score_predictions(pred_path, truth)
        if reason:
            log.info(f"Stage {subset_percentage}%: prediction scoring failed: {reason}")

        claimed = self._parse_score(execution_result)
        mismatch = claimed is not None and abs(claimed - score) > SCORE_CLAIM_MISMATCH_THRESHOLD
        if mismatch:
            log.warning(f"score_claim_mismatch: candidate claimed SCORE={claimed:.4f}, real score={score:.4f}")
        self.last_stage_flags = {"score_claim_mismatch": mismatch}

        success = score >= threshold

        log.info(f"Stage {subset_percentage}%: Score={score:.4f}, Threshold={threshold}")
        if not success:
            log.info(f"Candidate failed at {subset_percentage}% subset.")

        return success, score

    @staticmethod
    def _parse_score(execution_result: Dict[str, Any]) -> Optional[float]:
        """Diagnostic only - the candidate's own stdout claim, never trusted for gating."""
        match = SCORE_PATTERN.search(execution_result.get('stdout', ''))
        if not match:
            return None
        return float(match.group(1))
