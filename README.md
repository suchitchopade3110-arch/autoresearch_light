# ML Research Agent

An autonomous ML research agent loop: propose a candidate (as a diff), apply and commit it on its own git branch, run it in a sandbox against progressively larger subsets of a dataset, and merge or roll back based on the result - with RAG-style memory of past attempts, a human-approval gate before any merge, and, optionally, a concurrent evolutionary search over a population of candidates per generation.

## What's implemented
- **Orchestrator (`orchestrator/run.py`):** two modes.
  - `--mode sequential` (default): one candidate at a time; loops up to `orchestrator.max_iterations`, stopping early once `target_score` is reached or `patience` iterations pass with no improvement.
  - `--mode evolutionary`: a population of candidates per generation, evaluated concurrently and evolved via `evolution/population.py`'s `EvolutionEngine` (selection, mutation-by-prompt, Pareto/weighted scoring, adaptive population sizing, all seeded for reproducibility via `evolution.random_seed`).
- **Human-approval gate (`approval/`):** every candidate that passes evaluation is held pending a human decision before it merges - in either mode. The gate defaults to *required* even if the config is missing the `approval` section entirely or has a malformed value in it (see `approval/gate.py:resolve_approval_config`); only an explicit, valid `approval.enabled: false` disables it. A timeout with no decision is recorded as a real, persisted "timed_out" outcome and never merges - it is not treated as approval. Decisions are stored in SQLite (`approvals.db`), so they survive a restart and are visible to both the orchestrator process and the dashboard process.
- **Dashboard + API (`api/main.py`):** a FastAPI app serving a small Tailwind-styled page (auto-refreshing, no separate frontend build) to review and approve/reject pending candidates, plus JSON endpoints (`/api/pending`, `/api/approvals`, `/api/history`, `/api/report`). It reads directly from the same ChromaDB store, approval database, and `evolution_report.jsonl` the orchestrator writes to - there's no separate/forked data store to drift out of sync.
- **Report generator (`reporting/report_generator.py`):** `compute_kpis()` is the single function both the dashboard and the end-of-run report (`reports/latest_report.md`, written automatically when a run finishes) call - so the two surfaces can't independently recompute the same numbers differently. Tracks merge rate, duplicate-avoidance rate, compute cost per improvement, and approval outcomes.
- **Git State Controller (`vcs/git_controller.py`):** every candidate gets its own `git worktree`, so branching, committing, merging, and rolling back a candidate never touches the caller's main checkout (or any uncommitted work in it) - and concurrent candidates in the evolutionary path never share a checkout with each other.
- **Execution Sandbox (`sandbox/executor.py`):** runs each candidate in Docker as a non-root user, with `--network none`, a read-only root filesystem, dropped capabilities, and the existing CPU/memory limits and wall-clock timeout.
- **Real evaluation pipeline (`eval/dataset.py`, `eval/pipeline.py`):** a deterministic synthetic dataset (a noisy linear boundary), split into `train.jsonl`/`test.jsonl` (both mounted read-only into the sandbox) and `truth.json` (the held-out labels - stays on the host, never mounted). A candidate reads `TRAIN_PATH`/`TEST_PATH`/`SUBSET_PERCENTAGE`, and writes real predictions to `/app/out/predictions.jsonl`, which `EvalPipeline.score_predictions()` scores against `truth.json`. A printed `SCORE:` line is parsed only as a diagnostic to flag a mismatch between what the candidate claims and its real score - it is never trusted for gating, so a candidate cannot buy a merge by printing a perfect score claim (see `tests/test_reward_hacking.py`). A merge also requires beating the best-known score for that stage by `eval.min_improvement` (`eval/baseline.py`), not just clearing the stage's absolute threshold.
- **Experiment Memory (RAG) (`memory/db.py`):** a local ChromaDB instance storing hypotheses, diffs, outcomes, metrics, and rationale per experiment (cosine distance, so `evolution/duplicate_checker.py`'s similarity threshold is meaningful).
- **Failure Analysis (`memory/failure_analysis.py`):** categorizes failures (syntax, runtime, timeout, resource-limit, metric-regression).
- **Prompt Builder (`generation/prompt_builder.py`):** retrieves past successes/failures from memory into the next prompt.
- **Patch Generation (`generation/patch_generator.py`):** validates and applies unified diffs; `MockLLMClient` is the placeholder generator - it emits a script that implements the honest solution and writes real predictions (see above), so the eval pipeline has something genuine to gate on even before a real LLM is wired in.
- **Static Analysis Pre-check (`generation/static_check.py`):** rejects malformed/invalid syntax before sandbox execution.
- **Multi-objective scoring (`evolution/scoring.py`):** a candidate's real evaluation score drives selection, and a failed candidate can never outrank a successful one under either scoring strategy regardless of how fast it failed.

## What's NOT implemented yet
- **Real LLM integration.** `MockLLMClient` always returns the same diff regardless of prompt/history - implement the `LLMClient` interface with a real model to get genuinely different candidates per iteration. Because the mock's diff assumes a fixed starting file, concurrent evolutionary candidates whose worktree is created *after* an earlier candidate in the same generation has already merged will fail to apply (the file has moved on) - this is expected with a non-context-aware mock, not a bug, and goes away once a real LLM sees the file's current content in its prompt.
- **Dashboard authentication.** The approval dashboard has no auth - anyone who can reach it can approve or reject. Fine for local/single-user use; add auth before exposing it beyond localhost.
- **Carbon-footprint methodology.** `energy_estimate` is `execution_time * energy_watts_constant` (an arbitrary multiplier, default 10.0), not a real methodology like CodeCarbon or a grid-intensity constant - it's a placeholder signal for relative comparison between candidates, not an absolute measurement.

## Security Disclaimer
The sandbox runs candidates as a non-root user, with `--network none`, a read-only root filesystem, dropped capabilities, and standard Docker `--cpus`/`--memory` limits plus a wall-clock timeout via `subprocess`. This meaningfully raises the bar against a candidate trying to exfiltrate data, persist state, or exceed its resource limits. **It still does NOT provide hardened security against zero-days, container escapes, or a deliberately adversarial kernel exploit.** Do not execute untrusted malware in this sandbox.

## How to run locally

### Prerequisites
- Docker must be installed and running.
- Python 3.9+
- `pip install -r requirements.txt` (GitPython, PyYAML, pytest, chromadb, fastapi, uvicorn, jinja2)

### 1. Start the dashboard (in its own terminal)
```bash
uvicorn api.main:app --reload
```
Open `http://localhost:8000` to see pending approvals and run history. Leave this running - the orchestrator will block waiting for decisions made here.

### 2. Run the orchestrator (sequential mode)
```bash
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance"
# optional: --max-iterations 10 --target-score 0.9 --patience 3
```
Each candidate that passes evaluation shows up on the dashboard; approve or reject it there. No decision within `approval.timeout_seconds` (default 30 minutes) holds it - it will not merge.

### Evolutionary mode
```bash
python -m orchestrator.run --config configs/example.yaml --goal "Improve model performance" --mode evolutionary
```
Every candidate that passes evaluation in every generation gets its own pending approval, resolved independently and concurrently. Writes `evolution_report.jsonl` (one line per generation), stores every candidate's outcome in ChromaDB (`chroma_db/`), and writes `reports/latest_report.md` when the run finishes.

### Running without a human present (e.g. CI, demos)
Set `approval.enabled: false` explicitly in your config. This is an intentional, visible override, not a silent default - the shipped `configs/example.yaml` defaults to `enabled: true` and requires a human decision.

### Swapping in a real LLM

Implement the `LLMClient` interface in `generation/patch_generator.py`:

```python
class MyRealLLMClient(LLMClient):
    def generate_diff(self, prompt: str, target_file: str) -> str:
        # Call your API here and return the string unified diff
        return api.call(prompt)
```

### Config Schema (`configs/example.yaml`)

```yaml
sandbox:
  timeout_seconds: 5
  cpu_limit: "0.5"
  memory_limit: "256m"

dataset:
  path: "dummy_data"  # directory - train.jsonl/test.jsonl/truth.json generated once if missing
  size: 1000
  seed: 42
  test_frac: 0.25

eval:
  stages:                          # progressive scaling
    - subset_percentage: 1
      threshold: 0.5
    - subset_percentage: 5
      threshold: 0.6
    - subset_percentage: 20
      threshold: 0.7
    - subset_percentage: 100
      threshold: 0.8
  min_improvement: 0.001           # a merge must also beat the best-known score for its final stage by this much
  state_path: "state.json"         # persisted best-known score per stage

orchestrator:                      # sequential mode only
  max_iterations: 1
  target_score: 1.0
  patience: 1

evolution:                         # evolutionary mode only
  population_size: 5
  max_generations: 3
  max_concurrent_sandboxes: 3
  duplicate_threshold: 0.25
  selection_strategy: tournament
  scoring_strategy: pareto
  random_seed: 42

generation:
  max_retrieved_failures: 2
  max_retrieved_successes: 2
  prompt_char_budget: 4000

approval:                          # both modes - see Security Disclaimer above
  enabled: true                    # missing/malformed config also defaults to true
  timeout_seconds: 1800
  poll_interval_seconds: 5
  db_path: "approvals.db"
```

## Running Tests

```bash
pytest
```

`test_dataset.py`, `test_eval_pipeline.py`, `test_generation.py`, `test_vcs.py`, `test_memory.py`, `test_scoring.py`, `test_duplicate_checker.py`, `test_approval_store.py`, `test_approval_gate.py`, `test_report_generator.py`, and `test_api.py` are pure Python (`test_memory.py`/`test_report_generator.py` need `chromadb` installed) and need no Docker. `test_sandbox.py` and `test_integration*.py` build and run the Docker sandbox image and require Docker to be running.
