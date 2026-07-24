import json
from typing import List, Dict, Any

def log_generation_report(
    generation: int,
    scored_candidates: List[Dict[str, Any]],
    best_score: float,
    worst_score: float,
    compute_time_spent: float,
    convergence_signal: bool,
    duplicate_avoidance_count: int,
    duplicate_exhausted_count: int = 0,
    log_file: str = "evolution_report.jsonl"
):
    valid_scores = sorted([c['composite_score'] for c in scored_candidates if 'composite_score' in c])

    best_candidate_id = None
    worst_candidate_id = None
    if scored_candidates:
        best_cand = max(scored_candidates, key=lambda x: x.get('composite_score', -float('inf')))
        worst_cand = min(scored_candidates, key=lambda x: x.get('composite_score', float('inf')))
        best_candidate_id = best_cand.get('id')
        worst_candidate_id = worst_cand.get('id')

    report = {
        "generation": generation,
        "population_size": len(scored_candidates),
        "best_score": best_score,
        "worst_score": worst_score,
        "best_candidate_id": best_candidate_id,
        "worst_candidate_id": worst_candidate_id,
        "score_distribution": valid_scores,
        "convergence_signal": convergence_signal,
        "compute_time_spent": compute_time_spent,
        "duplicate_avoidance_count": duplicate_avoidance_count,
        "duplicate_exhausted_count": duplicate_exhausted_count
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(report) + "\n")

    print(f"Generation Report: {json.dumps(report)}")