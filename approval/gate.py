import time
from typing import Any, Dict, Optional

from approval.store import ApprovalStore

DEFAULT_TIMEOUT_SECONDS = 1800  # 30 minutes
DEFAULT_POLL_INTERVAL_SECONDS = 2


def resolve_approval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fail-safe config resolution: a missing 'approval' section, one that
    isn't even a dict, or one with the wrong type for any of its keys must
    never be interpreted as "no gate needed." The gate is required unless
    a config explicitly and validly sets approval.enabled: false.
    """
    section = config.get("approval") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        section = {}

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = True  # malformed type -> fail safe to required

    timeout_seconds = section.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    poll_interval = section.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
    if not isinstance(poll_interval, (int, float)) or isinstance(poll_interval, bool) or poll_interval <= 0:
        poll_interval = DEFAULT_POLL_INTERVAL_SECONDS

    return {"enabled": enabled, "timeout_seconds": timeout_seconds, "poll_interval_seconds": poll_interval}


def resolve_auto_approve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fail-safe like resolve_approval_config: a missing approval.auto_approve
    section, one that isn't a dict, or a malformed/negative
    min_improvement_over_baseline must never be interpreted as "auto-approve
    everything" - auto-approval only ever activates when
    min_improvement_over_baseline resolves to a valid, non-negative number.
    A missing require_no_failure_flags defaults to True (the stricter,
    safer option), not False.
    """
    section = config.get("approval") if isinstance(config, dict) else None
    section = section if isinstance(section, dict) else {}
    auto = section.get("auto_approve")
    auto = auto if isinstance(auto, dict) else {}

    min_improvement = auto.get("min_improvement_over_baseline")
    if not isinstance(min_improvement, (int, float)) or isinstance(min_improvement, bool) or min_improvement < 0:
        min_improvement = None  # unset or malformed -> auto-approval can never activate

    require_no_failure_flags = auto.get("require_no_failure_flags", True)
    if not isinstance(require_no_failure_flags, bool):
        require_no_failure_flags = True  # malformed type -> fail safe to the stricter requirement

    return {
        "min_improvement_over_baseline": min_improvement,
        "require_no_failure_flags": require_no_failure_flags,
    }


def should_auto_approve(config: Dict[str, Any], delta_over_baseline: Optional[float], has_failure_flags: bool) -> bool:
    """
    True only when approval.auto_approve.min_improvement_over_baseline is
    explicitly configured as a valid number AND the candidate's improvement
    over baseline meets or exceeds it, AND (if require_no_failure_flags,
    the default) the candidate has no failure flags. A candidate with no
    baseline to compare against (delta_over_baseline is None) never
    auto-approves - there is nothing to measure the threshold against, so
    it always falls through to the normal human-approval path.
    """
    auto_cfg = resolve_auto_approve_config(config)
    if auto_cfg["min_improvement_over_baseline"] is None:
        return False
    if delta_over_baseline is None or delta_over_baseline < auto_cfg["min_improvement_over_baseline"]:
        return False
    if auto_cfg["require_no_failure_flags"] and has_failure_flags:
        return False
    return True


def maybe_auto_approve(
    store: ApprovalStore,
    request_id: str,
    config: Dict[str, Any],
    delta_over_baseline: Optional[float],
    has_failure_flags: bool,
) -> Optional[str]:
    """
    Checks approval.auto_approve criteria for an already-created request
    and, if cleared, immediately records it as "auto_approved" - a
    distinct terminal state from "approved" (see approval/store.py), so a
    human decision and an automatic one are never conflated in the
    dashboard or reports. Returns "auto_approved" if it decided the
    request, or None if the caller must fall through to the normal
    human-approval wait.
    """
    if not should_auto_approve(config, delta_over_baseline, has_failure_flags):
        return None
    store.decide(request_id, "auto_approved", note="Cleared approval.auto_approve thresholds.")
    return "auto_approved"


def create_approval_request(
    store: ApprovalStore,
    candidate_id: str,
    goal: str,
    diff: str,
    final_score: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[str]:
    """
    Creates a pending approval request and returns its id, or None if the
    gate is disabled via an explicit, valid config - callers must treat a
    None return as "skipped" immediately, never as "approved".

    Split out from request_and_await_approval so a caller managing many
    candidates at once (see evolution/scheduler.py) can create every
    request for a generation up front - so a human reviewer sees the whole
    generation together - before awaiting any of them.
    """
    approval_cfg = resolve_approval_config(config)
    if not approval_cfg["enabled"]:
        return None
    return store.create_request(candidate_id, goal, diff, final_score, metrics)


def await_approval_decision(
    store: ApprovalStore,
    request_id: str,
    config: Dict[str, Any],
    sleep_fn=time.sleep,
    time_fn=time.monotonic,
) -> str:
    """
    Blocks (polling the persisted store) until request_id is decided, or
    the configured timeout elapses.

    Returns one of "approved", "rejected", or "timed_out" - a real,
    persisted decision, never a silent fallback. Callers must only merge
    when the return value is exactly "approved".
    """
    approval_cfg = resolve_approval_config(config)
    deadline = time_fn() + approval_cfg["timeout_seconds"]

    while True:
        try:
            request = store.get_request(request_id)
        except Exception:
            # store unreachable/corrupted mid-wait - fail safe, do not merge
            return "timed_out"

        if request is None or request["status"] != "pending":
            return request["status"] if request else "timed_out"

        if time_fn() >= deadline:
            store.decide(request_id, "timed_out", note="No decision within timeout window.")
            return "timed_out"

        sleep_fn(approval_cfg["poll_interval_seconds"])


def request_and_await_approval(
    store: ApprovalStore,
    candidate_id: str,
    goal: str,
    diff: str,
    final_score: float,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    sleep_fn=time.sleep,
    time_fn=time.monotonic,
) -> str:
    """
    Creates an approval request and blocks (polling the persisted store)
    until a human decides, or the configured timeout elapses.

    Returns one of:
      "approved"  - a human approved it; the caller may merge.
      "rejected"  - a human rejected it; the caller must roll back.
      "timed_out" - no decision arrived in time; the caller must roll back.
                    This is a real, persisted decision, not a silent
                    fallback - it is never treated as approval.
      "skipped"       - the gate is disabled via an explicit, valid config.
      "auto_approved" - approval.auto_approve's criteria were cleared; a
                         human was never asked. See maybe_auto_approve.

    Callers must only merge when the return value is "approved" or
    "auto_approved". A thin wrapper over create_approval_request +
    await_approval_decision for callers handling one candidate at a time
    (see orchestrator/run.py). metrics is expected to carry 'delta' (the
    candidate's improvement over baseline, set by the caller) and
    'score_claim_mismatch' (a failure flag) when auto-approval is in play -
    both simply read as None/False if absent, so callers that never set
    them keep today's always-human-gated behavior unchanged.
    """
    request_id = create_approval_request(store, candidate_id, goal, diff, final_score, metrics, config)
    if request_id is None:
        return "skipped"

    auto_decision = maybe_auto_approve(
        store, request_id, config, metrics.get("delta"), bool(metrics.get("score_claim_mismatch"))
    )
    if auto_decision is not None:
        return auto_decision

    return await_approval_decision(store, request_id, config, sleep_fn=sleep_fn, time_fn=time_fn)
