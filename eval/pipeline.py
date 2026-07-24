import json
import os
import re
from typing import Any, Dict, Optional, Tuple

SCORE_PATTERN = re.compile(r"SCORE:\s*([-+]?\d*\.?\d+)")
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
                        pred_path: str, truth: Dict[str, int]) -> Tuple[bool, float]:
        """Evaluates a single stage's real execution result. Returns (success, score)."""
        if execution_result['exit_code'] != 0 or execution_result.get('timeout', False):
            return False, 0.0

        score, reason = self.score_predictions(pred_path, truth)
        if reason:
            print(f"Stage {subset_percentage}%: prediction scoring failed: {reason}")

        claimed = self._parse_score(execution_result)
        if claimed is not None and abs(claimed - score) > SCORE_CLAIM_MISMATCH_THRESHOLD:
            print(f"WARNING score_claim_mismatch: candidate claimed SCORE={claimed:.4f}, real score={score:.4f}")

        success = score >= threshold

        print(f"Stage {subset_percentage}%: Score={score:.4f}, Threshold={threshold}")
        if not success:
            print(f"Candidate failed at {subset_percentage}% subset.")

        return success, score

    @staticmethod
    def _parse_score(execution_result: Dict[str, Any]) -> Optional[float]:
        """Diagnostic only - the candidate's own stdout claim, never trusted for gating."""
        match = SCORE_PATTERN.search(execution_result.get('stdout', ''))
        if not match:
            return None
        return float(match.group(1))
