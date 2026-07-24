import json
import os

from eval.baseline import BaselineStore


def test_get_defaults_to_zero_with_no_history(tmp_dir):
    store = BaselineStore(os.path.join(tmp_dir, "state.json"))
    assert store.get(100) == 0.0


def test_update_if_better_persists_across_instances(tmp_dir):
    path = os.path.join(tmp_dir, "state.json")
    store = BaselineStore(path)
    store.update_if_better(100, 0.85)

    reloaded = BaselineStore(path)
    assert reloaded.get(100) == 0.85


def test_update_if_better_never_lowers_the_baseline(tmp_dir):
    store = BaselineStore(os.path.join(tmp_dir, "state.json"))
    store.update_if_better(100, 0.9)
    store.update_if_better(100, 0.7)
    assert store.get(100) == 0.9


def test_stages_are_tracked_independently(tmp_dir):
    store = BaselineStore(os.path.join(tmp_dir, "state.json"))
    store.update_if_better(1, 0.5)
    store.update_if_better(100, 0.9)
    assert store.get(1) == 0.5
    assert store.get(100) == 0.9


def test_candidate_above_absolute_threshold_but_below_baseline_does_not_pass_gate(tmp_dir):
    """
    Wave 1 acceptance: an absolute threshold alone would let a regressing
    candidate merge as long as it clears the threshold (e.g. 0.81 still
    clears a 0.8 threshold even after a 0.95 candidate already merged).
    The baseline gate must reject that candidate despite it clearing the
    threshold.
    """
    store = BaselineStore(os.path.join(tmp_dir, "state.json"))
    store.update_if_better(100, 0.95)  # best candidate ever merged at this stage

    threshold = 0.8
    regressing_score = 0.81
    assert regressing_score >= threshold  # would pass an absolute-threshold-only gate
    assert store.passes(100, regressing_score) is False  # the baseline gate rejects it anyway

    genuine_improvement = 0.96
    assert store.passes(100, genuine_improvement) is True


def test_min_improvement_requires_more_than_a_tie():
    store = BaselineStore.__new__(BaselineStore)
    store._scores = {"100": 0.9}
    assert store.passes(100, 0.9, min_improvement=0.001) is False
    assert store.passes(100, 0.9005, min_improvement=0.001) is False
    assert store.passes(100, 0.902, min_improvement=0.001) is True


def test_state_file_is_plain_json_with_baseline_scores_key(tmp_dir):
    path = os.path.join(tmp_dir, "state.json")
    store = BaselineStore(path)
    store.update_if_better(100, 0.5)
    with open(path) as f:
        data = json.load(f)
    assert data == {"baseline_scores": {"100": 0.5}}
