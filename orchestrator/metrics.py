from typing import Callable, Dict, Any

_METRICS_REGISTRY: Dict[str, Callable] = {}

def register_metric(name: str):
    """
    Decorator to register a new metric calculation function.
    This allows plugin-style metric addition without modifying core loop code.
    """
    def decorator(func: Callable):
        _METRICS_REGISTRY[name] = func
        return func
    return decorator

def calculate_all_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Calculates all registered metrics for a given execution result."""
    metrics = {}
    for name, func in _METRICS_REGISTRY.items():
        metrics[name] = func(result)
    return metrics

# Example plugin metric
@register_metric("execution_speed")
def _execution_speed(result: Dict[str, Any]) -> float:
    time_taken = result.get("execution_time", 0.0)
    if time_taken <= 0:
        return 0.0
    return 1.0 / time_taken

@register_metric("execution_time")
def _execution_time(result: Dict[str, Any]) -> float:
    # Without this, evolution/scoring.py's energy_proxy derivation (which
    # checks for 'execution_time' in the metrics dict) never fires - it
    # always fell through to its hardcoded default regardless of how long a
    # candidate actually ran, and reporting has no real compute-cost signal
    # to work with either.
    return result.get("execution_time", 0.0)
