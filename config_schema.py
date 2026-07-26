from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """
    Raised when a config file fails schema validation. The message is
    pydantic's own readable, complete list of every problem found, not
    just the first one - so a malformed config is fixed in one pass
    instead of being re-run once per error.
    """


class EvalStage(BaseModel):
    model_config = ConfigDict(extra="allow")
    subset_percentage: int = Field(gt=0, le=100)
    threshold: float = Field(ge=0.0, le=1.0)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    stages: List[EvalStage]
    min_improvement: float = 0.001
    state_path: str = "state.json"

    @field_validator("stages")
    @classmethod
    def stages_must_not_be_empty(cls, v: List[EvalStage]) -> List[EvalStage]:
        if not v:
            raise ValueError(
                "eval.stages must contain at least one stage. An empty list "
                "means no stage ever runs to fail a candidate, so every "
                "candidate would pass evaluation with a score of 0.0."
            )
        return v


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    # The git repository that candidates are generated into and evaluated
    # against. Defaults to "." for backward compatibility, but that means
    # the harness's own repository unless explicitly pointed elsewhere -
    # see orchestrator/run.py's startup warning.
    repo_path: str = "."
    # Explicit commit-ish to treat as the base for candidate branches and
    # merges. Defaults to None, meaning "whatever branch repo_path's HEAD
    # currently points to" - only needed to pin a base when repo_path may
    # be in a detached HEAD state (see vcs/git_controller.py).
    base_ref: Optional[str] = None
    # The file(s) a candidate patch may touch. Defaults to just the
    # original single file. Must always include "candidate_script.py" -
    # the sandbox's Dockerfile CMD always executes that exact filename, so
    # a file list without it would silently generate patches for files the
    # sandbox never runs.
    files: List[str] = Field(default_factory=lambda: ["candidate_script.py"])

    @field_validator("files")
    @classmethod
    def files_must_include_candidate_script(cls, v: List[str]) -> List[str]:
        if "candidate_script.py" not in v:
            raise ValueError(
                "target.files must include 'candidate_script.py' - the sandbox's "
                "Dockerfile CMD always executes that exact filename, so omitting it "
                "would silently generate patches for files the sandbox never runs."
            )
        return v


class Config(BaseModel):
    """
    Validates the shape of config.yaml at load time, with readable errors
    instead of an AttributeError/KeyError surfacing deep inside a run.
    Every section besides eval/target is still just a free-form dict here
    (extra="allow") - the rest of the codebase keeps consuming config as a
    plain dict post-validation; this only guards the fields that have
    caused real bugs when missing or malformed.
    """
    model_config = ConfigDict(extra="allow")
    eval: EvalConfig
    target: TargetConfig = Field(default_factory=TargetConfig)


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads and validates a config YAML file. Returns the raw dict (not the
    pydantic model) so existing config.get(...) call sites are unaffected -
    validation is a gate, not a rewrite of how config is consumed.

    Raises ConfigError with a complete, readable list of problems if the
    file doesn't parse as a mapping or fails schema validation.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Invalid config at {path}: expected a YAML mapping at the top level, got {type(raw).__name__}")

    try:
        Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config at {path}:\n{e}") from e

    return raw
