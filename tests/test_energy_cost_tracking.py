from evolution.scoring import score_candidates
from memory.db import ExperimentDB
from reporting.report_generator import compute_kpis, render_report_markdown


def test_energy_proxy_is_the_metric_key_not_energy_estimate():
    """
    Priority 5: renamed from energy_estimate so it can never be mistaken
    for a real measurement - execution_time * a constant, nothing more.
    """
    candidates = [{'id': 'a', 'success': True, 'final_score': 0.9, 'metrics': {'execution_time': 2.0}}]

    scored = score_candidates(candidates, {'energy_proxy_watts_constant': 5.0})

    assert scored[0]['metrics']['energy_proxy'] == 10.0
    assert 'energy_estimate' not in scored[0]['metrics']


def test_energy_proxy_defaults_when_no_execution_time_present():
    candidates = [{'id': 'a', 'success': True, 'final_score': 0.9, 'metrics': {}}]

    scored = score_candidates(candidates, {})

    assert scored[0]['metrics']['energy_proxy'] == 100.0


def test_compute_kpis_energy_proxy_is_none_when_no_experiment_has_it(tmp_dir):
    """Sequential-mode-only runs never compute energy_proxy - must report 'no data', not a misleading 0.0."""
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="a", diff="d", rationale="r", metrics={"execution_time": 1.0}, outcome="success")

    kpis = compute_kpis(db, None, evolution_report_path="no_such_file.jsonl")

    assert kpis["total_energy_proxy"] is None


def test_compute_kpis_sums_energy_proxy_across_evolutionary_experiments(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={"energy_proxy": 12.0}, outcome="success")
    db.store_experiment(hypothesis="b", diff="d2", rationale="r", metrics={"energy_proxy": 8.0}, outcome="failure")

    kpis = compute_kpis(db, None, evolution_report_path="no_such_file.jsonl")

    assert kpis["total_energy_proxy"] == 20.0


def test_report_markdown_labels_energy_proxy_as_not_a_real_measurement(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={"energy_proxy": 12.0}, outcome="success")

    kpis = compute_kpis(db, None, evolution_report_path="no_such_file.jsonl")
    markdown = render_report_markdown(kpis)

    assert "Energy proxy total" in markdown
    assert "NOT a real energy/power measurement" in markdown


def test_report_markdown_omits_energy_proxy_line_entirely_when_no_data(tmp_dir):
    db = ExperimentDB(db_path=tmp_dir)
    db.store_experiment(hypothesis="a", diff="d1", rationale="r", metrics={"execution_time": 1.0}, outcome="success")

    kpis = compute_kpis(db, None, evolution_report_path="no_such_file.jsonl")
    markdown = render_report_markdown(kpis)

    assert "Energy proxy" not in markdown


def test_generation_cost_flows_end_to_end_from_last_usage_without_duplication(tmp_dir):
    """
    Priority 5: AnthropicClient.last_usage's estimated_cost_usd (see
    generation/patch_generator.py) is captured into each candidate's
    metrics['generation_cost_usd'] by orchestrator/run.py and
    evolution/scheduler.py, then summed exactly once here - nothing else
    in this module re-aggregates it, so it can never be double-counted.
    """
    db = ExperimentDB(db_path=tmp_dir)
    # Simulates two candidates, each carrying its own real
    # AnthropicClient.last_usage-derived cost, as orchestrator/run.py's
    # metrics['generation_cost_usd'] = generation_usage.get('estimated_cost_usd', 0.0) would set.
    db.store_experiment(
        hypothesis="a", diff="d1", rationale="r",
        metrics={"generation_cost_usd": 0.0123, "generation_input_tokens": 100, "generation_output_tokens": 50},
        outcome="success",
    )
    db.store_experiment(
        hypothesis="b", diff="d2", rationale="r",
        metrics={"generation_cost_usd": 0.0456, "generation_input_tokens": 200, "generation_output_tokens": 80},
        outcome="failure",
    )

    kpis = compute_kpis(db, None, evolution_report_path="no_such_file.jsonl")

    assert kpis["total_generation_cost_usd"] == 0.0123 + 0.0456
    assert kpis["total_generation_input_tokens"] == 300
    assert kpis["total_generation_output_tokens"] == 130

    markdown = render_report_markdown(kpis)
    # Appears exactly once in the report - proof there's no second,
    # independently-computed cost total anywhere in the markdown.
    assert markdown.count("LLM generation cost") == 1
