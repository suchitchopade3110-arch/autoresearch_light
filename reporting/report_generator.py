import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from observability.logging_config import get_logger

EVOLUTION_REPORT_PATH = "evolution_report.jsonl"
_module_logger = get_logger(__name__)


def _read_evolution_generations(path: str = EVOLUTION_REPORT_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    generations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                generations.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return generations


def compute_kpis(db, approval_store=None, evolution_report_path: str = EVOLUTION_REPORT_PATH) -> Dict[str, Any]:
    """
    Computes every KPI shown on the dashboard and written to the end-of-run
    report from the same underlying sources - memory/db.py's experiment
    store and, if present, evolution_report.jsonl - so the dashboard and
    the report can never independently recompute (and silently drift from)
    the same numbers. Both call this same function rather than each doing
    their own arithmetic.
    """
    experiments = db.list_all_experiments()
    total = len(experiments)
    success_count = sum(1 for e in experiments if e["outcome"] == "success")
    held_count = sum(1 for e in experiments if e["outcome"] == "held")
    failure_count = sum(1 for e in experiments if e["outcome"] == "failure")
    merge_rate = success_count / total if total else 0.0

    generations = _read_evolution_generations(evolution_report_path)
    duplicates_avoided = sum(g.get("duplicate_avoidance_count", 0) for g in generations)
    candidates_scheduled = sum(g.get("population_size", 0) for g in generations) or total
    duplicate_avoidance_rate = (
        duplicates_avoided / (duplicates_avoided + candidates_scheduled)
        if (duplicates_avoided + candidates_scheduled) else 0.0
    )

    # evolutionary mode logs real compute time per generation directly;
    # sequential mode never writes evolution_report.jsonl, so fall back to
    # summing each stored experiment's own execution_time metric.
    if generations:
        total_compute_seconds = sum(g.get("compute_time_spent", 0.0) for g in generations)
    else:
        total_compute_seconds = sum(e["metrics"].get("execution_time", 0.0) for e in experiments)

    compute_cost_per_improvement_seconds = (
        total_compute_seconds / success_count if success_count else None
    )

    # Only populated by AnthropicClient (see generation/patch_generator.py) -
    # MockLLMClient makes no API calls, so this is 0.0 for every mock run.
    # Summed once, here, over each experiment's own single recorded cost -
    # nothing else in this module or the dashboard re-aggregates it, so
    # this total is never double-counted.
    total_generation_cost_usd = sum(e["metrics"].get("generation_cost_usd", 0.0) for e in experiments)
    total_generation_input_tokens = sum(e["metrics"].get("generation_input_tokens", 0) for e in experiments)
    total_generation_output_tokens = sum(e["metrics"].get("generation_output_tokens", 0) for e in experiments)

    # energy_proxy only exists on evolutionary-mode experiments (see
    # evolution/scoring.py) - sequential mode never computes it. None
    # (not 0.0) when absent, so a sequential-only run reports "no data"
    # rather than a misleading "zero energy used".
    has_energy_proxy_data = any('energy_proxy' in e["metrics"] for e in experiments)
    total_energy_proxy = (
        sum(e["metrics"].get("energy_proxy", 0.0) for e in experiments) if has_energy_proxy_data else None
    )

    approvals = approval_store.list_all() if approval_store else []
    approved = sum(1 for a in approvals if a["status"] == "approved")
    # A distinct terminal state from "approved" (see approval/store.py) -
    # a human never decided these, approval.auto_approve's criteria did.
    auto_approved = sum(1 for a in approvals if a["status"] == "auto_approved")
    rejected = sum(1 for a in approvals if a["status"] == "rejected")
    timed_out = sum(1 for a in approvals if a["status"] == "timed_out")
    pending = sum(1 for a in approvals if a["status"] == "pending")
    decided = approved + auto_approved + rejected + timed_out
    approval_timeout_rate = timed_out / decided if decided else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_experiments": total,
        "success_count": success_count,
        "held_count": held_count,
        "failure_count": failure_count,
        "merge_rate": merge_rate,
        "duplicate_avoidance_rate": duplicate_avoidance_rate,
        "duplicates_avoided": duplicates_avoided,
        "candidates_scheduled": candidates_scheduled,
        "total_compute_seconds": total_compute_seconds,
        "compute_cost_per_improvement_seconds": compute_cost_per_improvement_seconds,
        "total_generation_cost_usd": total_generation_cost_usd,
        "total_generation_input_tokens": total_generation_input_tokens,
        "total_generation_output_tokens": total_generation_output_tokens,
        "total_energy_proxy": total_energy_proxy,
        "approvals": {
            "pending": pending,
            "approved": approved,
            "auto_approved": auto_approved,
            "rejected": rejected,
            "timed_out": timed_out,
            "timeout_rate": approval_timeout_rate,
        },
    }


def render_report_markdown(kpis: Dict[str, Any]) -> str:
    cost_line = (
        f"{kpis['compute_cost_per_improvement_seconds']:.2f}s"
        if kpis["compute_cost_per_improvement_seconds"] is not None
        else "n/a (no merged candidates yet)"
    )
    energy_line = (
        f"- Energy proxy total: {kpis['total_energy_proxy']:.2f} (execution_time x a constant - "
        "NOT a real energy/power measurement, see evolution/scoring.py; evolutionary mode only)\n"
        if kpis.get("total_energy_proxy") is not None
        else ""
    )
    return f"""# Run Report

Generated: {kpis['generated_at']}

## Experiments
- Total experiments: {kpis['total_experiments']}
- Merged (success): {kpis['success_count']}
- Held (passed eval, not approved): {kpis['held_count']}
- Failed: {kpis['failure_count']}
- **Merge rate: {kpis['merge_rate']:.1%}**

## Search efficiency
- Duplicate candidates avoided: {kpis['duplicates_avoided']}
- Candidates scheduled: {kpis['candidates_scheduled']}
- **Duplicate avoidance rate: {kpis['duplicate_avoidance_rate']:.1%}**
- Total compute time: {kpis['total_compute_seconds']:.2f}s
- **Compute cost per improvement: {cost_line}**
- LLM generation cost: ${kpis['total_generation_cost_usd']:.4f} ({kpis['total_generation_input_tokens']} in / {kpis['total_generation_output_tokens']} out tokens) - 0 for mock runs
{energy_line}
## Human approval gate
- Pending: {kpis['approvals']['pending']}
- Approved: {kpis['approvals']['approved']}
- Auto-approved (approval.auto_approve criteria, no human decision): {kpis['approvals']['auto_approved']}
- Rejected: {kpis['approvals']['rejected']}
- Timed out (held, not merged): {kpis['approvals']['timed_out']}
- Timeout rate (of decided): {kpis['approvals']['timeout_rate']:.1%}
"""


def generate_report(db, approval_store=None, output_dir: str = "reports", logger=None) -> str:
    """Writes the end-of-run report to disk and returns its path."""
    kpis = compute_kpis(db, approval_store)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(output_dir, f"report_{timestamp}.md")
    markdown = render_report_markdown(kpis)

    with open(report_path, "w") as f:
        f.write(markdown)
    with open(os.path.join(output_dir, "latest_report.md"), "w") as f:
        f.write(markdown)
    with open(os.path.join(output_dir, "latest_report.json"), "w") as f:
        json.dump(kpis, f, indent=2)

    (logger or _module_logger).info(f"Run report written to {report_path}")
    # The report body itself is deliberately a direct print, not a log
    # record - it's the run's human-facing deliverable output (read as
    # markdown), not an operational trace event.
    print(markdown)
    return report_path
