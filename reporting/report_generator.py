import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

EVOLUTION_REPORT_PATH = "evolution_report.jsonl"


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
    total_generation_cost_usd = sum(e["metrics"].get("generation_cost_usd", 0.0) for e in experiments)
    total_generation_input_tokens = sum(e["metrics"].get("generation_input_tokens", 0) for e in experiments)
    total_generation_output_tokens = sum(e["metrics"].get("generation_output_tokens", 0) for e in experiments)

    approvals = approval_store.list_all() if approval_store else []
    approved = sum(1 for a in approvals if a["status"] == "approved")
    rejected = sum(1 for a in approvals if a["status"] == "rejected")
    timed_out = sum(1 for a in approvals if a["status"] == "timed_out")
    pending = sum(1 for a in approvals if a["status"] == "pending")
    decided = approved + rejected + timed_out
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
        "approvals": {
            "pending": pending,
            "approved": approved,
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

## Human approval gate
- Pending: {kpis['approvals']['pending']}
- Approved: {kpis['approvals']['approved']}
- Rejected: {kpis['approvals']['rejected']}
- Timed out (held, not merged): {kpis['approvals']['timed_out']}
- Timeout rate (of decided): {kpis['approvals']['timeout_rate']:.1%}
"""


def generate_report(db, approval_store=None, output_dir: str = "reports") -> str:
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

    print(f"\n=== Run report written to {report_path} ===")
    print(markdown)
    return report_path
