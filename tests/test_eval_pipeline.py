import json
import os

from eval.pipeline import EvalPipeline

CONFIG = {"stages": [
    {"subset_percentage": 1, "threshold": 0.5},
    {"subset_percentage": 100, "threshold": 0.8},
]}

TRUTH = {"0": 1, "1": 0, "2": 1, "3": 0}


def make_result(stdout="", exit_code=0, timeout=False):
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "execution_time": 0.1, "timeout": timeout}


def write_predictions(tmp_dir, preds):
    path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(path, "w") as f:
        for row in preds:
            f.write(json.dumps(row) + "\n")
    return path


def test_evaluate_stage_scores_real_predictions_not_stdout(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(), 100, 0.8, pred_path, TRUTH)
    assert success
    assert score == 1.0


def test_evaluate_stage_ignores_a_perfect_stdout_claim(tmp_dir):
    """The exact reward-hacking exploit: a candidate claims SCORE: 1.0 but writes nothing useful."""
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 0}, {"id": 1, "pred": 0}, {"id": 2, "pred": 0}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(stdout="SCORE: 1.0\n"), 100, 0.8, pred_path, TRUTH)
    assert not success
    assert score == 0.5  # 2 of 4 correct, regardless of the claimed 1.0


def test_evaluate_stage_fails_below_threshold(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 0}, {"id": 1, "pred": 0}, {"id": 2, "pred": 0}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(), 1, 0.9, pred_path, TRUTH)
    assert not success
    assert score == 0.5


def test_evaluate_stage_missing_predictions_scores_zero(tmp_dir):
    pipeline = EvalPipeline(CONFIG)
    pred_path = os.path.join(tmp_dir, "never_written.jsonl")
    success, score = pipeline.evaluate_stage(make_result(), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_partial_id_coverage_scores_zero(tmp_dir):
    """A candidate that only predicts for half the test ids gets 0.0, not partial credit."""
    pred_path = write_predictions(tmp_dir, [{"id": 0, "pred": 1}, {"id": 1, "pred": 0}])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(), 100, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_malformed_predictions_scores_zero_not_exception(tmp_dir):
    path = os.path.join(tmp_dir, "predictions.jsonl")
    with open(path, "w") as f:
        f.write("not json at all\n")
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(), 1, 0.5, path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_non_binary_prediction_scores_zero(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 7}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_nonzero_exit_fails(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(exit_code=1), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_evaluate_stage_timeout_fails(tmp_dir):
    pred_path = write_predictions(tmp_dir, [
        {"id": 0, "pred": 1}, {"id": 1, "pred": 0}, {"id": 2, "pred": 1}, {"id": 3, "pred": 0},
    ])
    pipeline = EvalPipeline(CONFIG)
    success, score = pipeline.evaluate_stage(make_result(timeout=True), 1, 0.5, pred_path, TRUTH)
    assert not success
    assert score == 0.0


def test_score_predictions_reports_a_reason_on_every_zero(tmp_dir):
    pipeline = EvalPipeline(CONFIG)

    score, reason = pipeline.score_predictions(os.path.join(tmp_dir, "missing.jsonl"), TRUTH)
    assert score == 0.0 and reason

    mismatched_path = write_predictions(tmp_dir, [{"id": 0, "pred": 1}])
    score, reason = pipeline.score_predictions(mismatched_path, TRUTH)
    assert score == 0.0 and reason
