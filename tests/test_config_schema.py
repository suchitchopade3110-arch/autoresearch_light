import os
import tempfile

import pytest
import yaml

from config_schema import ConfigError, load_config
from orchestrator.run import target_is_harness


def _write_config(tmp_path, data):
    path = os.path.join(tmp_path, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_empty_eval_stages_rejected_with_readable_error(tmp_dir):
    path = _write_config(tmp_dir, {"eval": {"stages": []}})

    with pytest.raises(ConfigError) as exc_info:
        load_config(path)

    assert "eval.stages" in str(exc_info.value)


def test_missing_eval_section_rejected(tmp_dir):
    path = _write_config(tmp_dir, {"dataset": {"size": 10}})

    with pytest.raises(ConfigError):
        load_config(path)


def test_stage_with_out_of_range_threshold_rejected(tmp_dir):
    path = _write_config(tmp_dir, {"eval": {"stages": [{"subset_percentage": 10, "threshold": 1.5}]}})

    with pytest.raises(ConfigError):
        load_config(path)


def test_valid_config_loads_and_returns_plain_dict(tmp_dir):
    path = _write_config(
        tmp_dir,
        {
            "eval": {"stages": [{"subset_percentage": 100, "threshold": 0.5}]},
            "dataset": {"size": 10},
        },
    )

    config = load_config(path)

    assert isinstance(config, dict)
    assert config["eval"]["stages"][0]["threshold"] == 0.5
    assert config["dataset"]["size"] == 10


def test_non_mapping_config_rejected(tmp_dir):
    path = os.path.join(tmp_dir, "config.yaml")
    with open(path, "w") as f:
        f.write("- just\n- a\n- list\n")

    with pytest.raises(ConfigError):
        load_config(path)


def test_target_is_harness_true_when_paths_match(tmp_dir):
    assert target_is_harness(tmp_dir, tmp_dir) is True


def test_target_is_harness_false_for_a_separate_directory(tmp_dir):
    other = os.path.join(tmp_dir, "elsewhere")
    os.makedirs(other)
    assert target_is_harness(other, tmp_dir) is False


def test_target_is_harness_resolves_relative_paths(tmp_dir):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_dir)
        assert target_is_harness(".", tmp_dir) is True
    finally:
        os.chdir(cwd)
