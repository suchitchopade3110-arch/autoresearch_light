from typing import List, Dict, Any, Tuple
import math

def dominates(scores_a: Dict[str, float], scores_b: Dict[str, float], maximize: Dict[str, bool]) -> bool:
    at_least_as_good_in_all = True
    strictly_better_in_one = False

    for metric in maximize.keys():
        val_a = scores_a.get(metric, 0.0)
        val_b = scores_b.get(metric, 0.0)

        if maximize[metric]:
            if val_a < val_b:
                at_least_as_good_in_all = False
                break
            if val_a > val_b:
                strictly_better_in_one = True
        else:
            if val_a > val_b:
                at_least_as_good_in_all = False
                break
            if val_a < val_b:
                strictly_better_in_one = True

    return at_least_as_good_in_all and strictly_better_in_one

def calculate_pareto_fronts(candidates: List[Dict[str, Any]], maximize: Dict[str, bool]) -> List[int]:
    n = len(candidates)
    ranks = [0] * n

    for i in range(n):
        domination_count = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(candidates[j]['metrics'], candidates[i]['metrics'], maximize):
                domination_count += 1
        ranks[i] = domination_count

    return ranks

def score_candidates(candidates: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    strategy = config.get('scoring_strategy', 'pareto')
    # NOT a real energy measurement - execution_time * a configurable
    # constant, with no relationship to the candidate's actual CPU/GPU
    # power draw. A real measurement (e.g. via CodeCarbon) would need to
    # run inside the sandboxed container the candidate actually executes
    # in, and CodeCarbon's own hardware power counters (Intel RAPL) are
    # not readable from a container this hardened (--cap-drop ALL,
    # --read-only, no-new-privileges - see sandbox/executor.py) without
    # weakening that isolation, which is a bigger trade-off than this
    # metric is worth. Named energy_proxy, not energy_estimate, so it can
    # never be mistaken for real energy data by anything reading these
    # metrics later (reporting, the dashboard, or a future consumer).
    watts_constant = config.get('energy_proxy_watts_constant', 10.0)

    for c in candidates:
        m = c.get('metrics', {})
        if 'energy_proxy' not in m and 'execution_time' in m:
            m['energy_proxy'] = m['execution_time'] * watts_constant

        # final_score is the candidate's real progressive-eval result, set by
        # the scheduler; without this, validation_accuracy always defaulted
        # to 0.0 for every candidate, so selection ignored correctness
        # entirely and ranked purely on execution_speed/energy_proxy.
        if 'validation_accuracy' not in m and 'final_score' in c:
            m['validation_accuracy'] = c['final_score']

        m.setdefault('validation_accuracy', 0.0)
        m.setdefault('execution_speed', 0.0)
        m.setdefault('memory_usage', 1024.0)
        m.setdefault('energy_proxy', 100.0)

        c['metrics'] = m

    maximize = {
        'validation_accuracy': True,
        'execution_speed': True,
        'memory_usage': False,
        'energy_proxy': False
    }

    if strategy == 'weighted':
        weights = config.get('weights', {
            'validation_accuracy': 1.0,
            'execution_speed': 0.2,
            'memory_usage': 0.1,
            'energy_proxy': 0.1
        })

        for c in candidates:
            m = c['metrics']
            score = (
                m['validation_accuracy'] * weights.get('validation_accuracy', 1.0) +
                m['execution_speed'] * weights.get('execution_speed', 1.0) -
                (math.log(max(m['memory_usage'], 1.0)) * weights.get('memory_usage', 1.0)) -
                (math.log(max(m['energy_proxy'], 1.0)) * weights.get('energy_proxy', 1.0))
            )
            c['composite_score'] = score

    elif strategy == 'pareto':
        ranks = calculate_pareto_fronts(candidates, maximize)
        for i, c in enumerate(candidates):
            c['composite_score'] = -ranks[i]
            c['pareto_rank'] = ranks[i]

    else:
        raise ValueError(f"Unknown scoring strategy: {strategy}")

    # A candidate that failed evaluation must never outrank one that passed,
    # under either strategy - without this, a candidate that crashed almost
    # instantly (very low execution_time -> very high execution_speed / low
    # energy_proxy) can out-score a real success on those axes alone,
    # since execution_speed is unbounded while validation_accuracy is
    # capped at 1.0. Shifting every failure's score down by the same amount
    # preserves relative order within the successes and within the
    # failures, and just guarantees the two groups never cross.
    successful = [c['composite_score'] for c in candidates if c.get('success')]
    failed = [c for c in candidates if not c.get('success')]
    if successful and failed:
        min_success = min(successful)
        max_failure = max(c['composite_score'] for c in failed)
        if max_failure >= min_success:
            shift = (max_failure - min_success) + 1.0
            for c in failed:
                c['composite_score'] -= shift

    return candidates