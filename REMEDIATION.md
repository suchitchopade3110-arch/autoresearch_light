# Remediation record

Every gap identified across the pre-wave audit and the four remediation
waves, mapped to the commit that closed it and the test that proves it.
Commit hashes are short SHAs on `main`; PR numbers refer to
`suchitchopade3110-arch/autoresearch_lite`.

## Pre-wave audit fixes (PRs #2-#9)

| Gap | Fix | Proving test |
|---|---|---|
| Evolutionary scoring let a fast-failing candidate outrank a real success on execution-speed/energy alone | `27753d3` clamps every failed candidate's composite score below every successful one | `tests/test_scoring.py` |
| Duplicate detection never actually fired - `retrieve_experiments` didn't return a `distance` field, so the threshold check always compared against `inf` | `27753d3` | `tests/test_duplicate_checker.py::test_retrieve_experiments_includes_distance` |
| No human-approval gate before a merge | `20af43c` (PR #5) adds `approval/`, a SQLite-backed store, and a dashboard/API to decide pending candidates | `tests/test_approval_gate.py`, `tests/test_approval_store.py`, `tests/test_api.py` |
| Windows: `git worktree remove` failed with a file-lock `PermissionError` because a `git.Repo` handle was left open | `f745a65` | `tests/test_vcs.py` (Windows-run) |
| Windows: integration tests invoked a bare `python`, which can resolve to a different interpreter than the one running pytest | `566eb98` | `tests/test_integration*.py` |
| Windows: a candidate diff's patch was silently truncated - `.patch` files were written with platform line-ending translation, corrupting `\n` inside hunks | `ed92264`, `d8ed092` (the real root cause: a wrong declared hunk line count) | `tests/test_generation.py::test_mock_llm_diff_hunk_header_matches_actual_line_count` |
| `run_iteration` could crash the whole orchestrator loop by returning `False` instead of a `(bool, float)` tuple on a rejected patch | `2dc1400` | `tests/test_integration.py` |

## Wave 1 - close the reward-hacking gap (PR #10)

| Gap | Fix | Proving test |
|---|---|---|
| The eval pipeline trusted a candidate's own printed `SCORE:` claim to decide pass/fail - a candidate could buy a merge by lying | `0738ff4` replaces stdout-trust with `score_predictions()`, scoring real predictions against held-out `truth.json` the candidate never sees | `tests/test_reward_hacking.py::test_printed_score_claim_is_never_trusted` |
| `truth.json` (the held-out labels) had no guarantee against ever being mounted into the sandbox | `0738ff4` - `eval/dataset.py` keeps it host-side only; `sandbox/executor.py` only ever mounts `train.jsonl`/`test.jsonl` | `tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide` |
| A merge only required clearing a fixed threshold, not beating what was already merged - a regression could still land | `0738ff4` adds `eval/baseline.py`'s `min_improvement` gate | `tests/test_baseline.py` |
| `MockLLMClient` didn't write real predictions, so the reward-hacking fix had nothing genuine to score | `0738ff4` rewrites it to implement an honest baseline and write `predictions.jsonl` | `tests/test_generation.py` |
| The `truth.json`-mount canary test shelled out to `grep`, which doesn't exist on Windows | `0d94809` | `tests/test_reward_hacking.py::test_truth_json_absent_from_every_docker_mount_argument_repo_wide` |

## Wave 2 - make the agent an agent (PR #11)

| Gap | Fix | Proving test |
|---|---|---|
| Every LLM client generated a diff without ever seeing the file's current content - every candidate after the first in a run was dead-on-arrival (diffs against stale state never apply) | `07c66b2` threads `current_content` through `LLMClient.generate_diff` | `tests/test_evolution_population.py` |
| No real, network-backed LLM client existed - only the deterministic mock | `07c66b2` adds `AnthropicClient` (retries on `git apply --check` failure, feeds stderr back into the prompt, tracks token/cost usage) | `tests/test_anthropic_client.py` |
| Duplicate-detection retries silently fell back to a scoreless placeholder diff on exhaustion, burning a full sandbox budget on something that could never score | `07c66b2` returns `None` and records a `duplicate_exhausted`/`failure` outcome instead | `tests/test_evolution_population.py` |
| The sandbox image had no ML libraries, so any real candidate script needing numpy/pandas/scikit-learn would fail immediately | `07c66b2` adds them to `sandbox/requirements-sandbox.txt`, installed before `COPY` for layer caching | `tests/test_sandbox.py::test_sandbox_has_ml_libraries` |

## Wave 3 - resilience (PR #12)

| Gap | Fix | Proving test |
|---|---|---|
| A merge conflict between two candidates could leave the shared checkout mid-merge (`MERGE_HEAD` present, working tree dirty) | `18703cf` - `merge()` rebases inside the candidate's own worktree first, then fast-forwards; raises `MergeConflict` instead of ever touching the shared checkout with a real conflict | `tests/test_vcs.py::test_second_conflicting_merge_raises_and_leaves_the_main_checkout_clean` |
| A human reviewer was a bottleneck on the sandbox pool - the old design awaited each candidate's approval before the next one's sandbox stage could start | `18703cf` - two-phase scheduler: every candidate's sandbox/eval work completes first, then every approval request is created up front | `tests/test_evolution_scheduler.py::test_all_sandbox_evaluation_completes_before_any_approval_request_blocks` |
| An empty `eval.stages` list let every candidate pass evaluation with a score of 0.0 (no stage ever ran to fail it) | `18703cf` - `config_schema.py` rejects it at load time with a readable error | `tests/test_config_schema.py::test_empty_eval_stages_rejected_with_readable_error` |
| The repo under evolution was implicitly the harness's own repository, with no way to separate them, and no warning when that happens | `18703cf` - `target.repo_path`/`target.base_ref` config keys, plus a startup warning | `tests/test_config_schema.py::test_target_is_harness_*` |
| `GitController` crashed via `repo.active_branch.name` if the target repo was in a detached HEAD state | `18703cf` - falls back to the current commit sha, and correctly advances it after a merge | `tests/test_vcs.py::test_detached_head_*` |
| A crashed/killed run left orphaned candidate worktrees/branches and stuck-pending approval requests with nothing to reclaim them | `18703cf` - `cleanup_orphans()`, `timeout_stale_requests()`, a `cleanup` subcommand, and automatic startup cleanup | `tests/test_vcs.py::test_cleanup_orphans_*`, `tests/test_approval_store.py::test_timeout_stale_requests_*`, `tests/test_orchestrator_cleanup.py` |
| `SIGINT`/`SIGTERM` had no handler - a stop signal propagated as an unhandled signal/traceback | `18703cf` | `tests/test_orchestrator_cleanup.py::test_sigterm_after_install_signal_handlers_exits_cleanly_instead_of_a_traceback` |
| Every operational message was a bare `print()` - no way to correlate log lines to a run or candidate | `18703cf` - `observability/logging_config.py`, structured JSON logs with `run_id`/`candidate_id` bound to every record | `tests/test_logging_config.py` |
| `cleanup_orphans()` identified orphan worktrees by comparing filesystem paths, which Windows can report inconsistently (short vs. long form) between git and Python, silently breaking the match | `85be1fd` - identifies orphans by the branch checked out in git's own porcelain listing instead | `tests/test_vcs.py::test_cleanup_orphans_removes_leftover_worktrees_and_branches_from_a_crashed_run` (Windows-run) |
| A raw `sqlite3` connection opened via `with conn:` in a test left a file handle open on Windows, blocking temp-directory cleanup | `85be1fd` | `tests/test_approval_store.py::test_timeout_stale_requests_*` (Windows-run) |
| `os.kill(pid, SIGTERM)` against the current process calls `TerminateProcess()` directly on Windows, bypassing Python's signal handler and killing the whole test run instead of raising `SystemExit` | `e296675` - invokes the installed handler directly instead of raising a real OS signal | `tests/test_orchestrator_cleanup.py::test_sigterm_after_install_signal_handlers_exits_cleanly_instead_of_a_traceback` |

## Wave 4 - hygiene

| Gap | Fix | Proving test |
|---|---|---|
| No LICENSE or packaging metadata | `LICENSE` (MIT), `pyproject.toml` | Build check: `python -m build --sdist` / `setuptools.find_packages()` discovers exactly the 11 real packages |
| No CI | `.github/workflows/tests.yml` runs the full suite (including Docker) on every push/PR to `main` | N/A - CI itself |
| `evolution/` had no `__init__.py`, unlike every other package | `evolution/__init__.py` | Packaging discovery check above |
| The dashboard had no authentication - anyone who could reach it could approve/reject | `api/main.py` - HTTP Basic Auth, fail-safe (required unless `DASHBOARD_AUTH_DISABLED=true`, and rejects everything if enabled with no credentials configured) | `tests/test_api_security.py::test_dashboard_requires_auth_by_default_when_no_credentials_configured`, `test_dashboard_rejects_wrong_credentials`, `test_dashboard_accepts_correct_credentials` |
| The approve/reject endpoints had no CSRF protection - a cross-site auto-submitting form could forge a decision | `api/main.py` - double-submit-cookie CSRF token on both forms | `tests/test_api_security.py::test_approve_without_csrf_token_is_rejected`, `test_approve_with_mismatched_csrf_token_is_rejected`, `test_approve_with_matching_csrf_token_succeeds` |
| The sandbox had no limit on process count or open file descriptors, and its `/tmp` tmpfs was unbounded (RAM-backed) | `sandbox/executor.py` - `--pids-limit`, `--ulimit nofile=`, a size-capped `--tmpfs` | `tests/test_sandbox.py::test_docker_run_command_includes_resource_hardening_flags` |
| `ExperimentDB` had no class docstring, unlike every other core class in the codebase | `memory/db.py` | N/A - documentation only |
| An exact-diff repeat paid for a full embedding + nearest-neighbor search before ever checking if it was a byte-for-byte duplicate | `memory/db.py:has_exact_diff`, wired into `evolution/duplicate_checker.py` | `tests/test_duplicate_checker.py::test_exact_duplicate_short_circuits_before_the_expensive_embedding_query` |
| `energy_estimate`'s placeholder methodology was implemented but not clearly flagged as such | Documented explicitly in README's "What's NOT implemented yet" (no code change - this is a deliberate, already-correct design choice, not a bug) | N/A - documentation |
| README was out of date with Waves 2-4 (target/harness separation, crash recovery, structured logging, dashboard security, sandbox hardening) | README rewritten | N/A - documentation |

## Final verification checklist

- [x] Full suite green (non-Docker): confirmed in this environment (no Docker daemon available here).
- [ ] Full suite green including Docker: requires a machine with Docker running - verify with `pytest -q`.
- [x] `tests/test_reward_hacking.py` (the reward-hacking canary) still passes.
- [x] `git grep truth.json -- '*.py'` shows no occurrence inside a Docker mount argument (`-v ...`) - only host-side loading code.
- [x] A deliberately injected mid-run conflict leaves the repo clean: `tests/test_vcs.py::test_second_conflicting_merge_raises_and_leaves_the_main_checkout_clean` constructs exactly this (two candidates editing the same line) and asserts no `MERGE_HEAD`, no dirty working tree.
- [ ] A real end-to-end evolutionary run with a real LLM client (`generation.client: anthropic`, `ANTHROPIC_API_KEY` set) and the approval gate on: requires a real API key, which was never provided to (or guessed by) this remediation work per its own ground rules - run manually with:
  ```bash
  ANTHROPIC_API_KEY=... python -m orchestrator.run --config configs/example.yaml --mode evolutionary
  ```
  and confirm `reports/latest_report.md` and `evolution_report.jsonl` reflect real generation cost/token usage.
