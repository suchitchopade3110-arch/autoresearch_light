from typing import Dict, Any, Tuple

def analyze_failure(execution_result: Dict[str, Any], evaluation_success: bool) -> Tuple[str, str, str]:
    """
    Analyzes the execution result and evaluation success to categorize the failure.
    Returns (category, error_text, traceback).
    Categories: syntax, runtime, timeout, resource-limit, metric-regression, success

    traceback is the raw diagnostic text (execution_result's stderr) when a
    real one actually exists - distinct from error_text, which for
    non-crash categories (metric-regression, success) is a human-readable
    label with no underlying stack trace to show. Consumers that want the
    real signal for the next prompt (generation/prompt_builder.py) should
    read traceback, not error_text.
    """
    if execution_result.get('timeout', False):
        stderr = execution_result.get('stderr', '')
        return "timeout", stderr or 'Execution timed out.', stderr

    exit_code = execution_result.get('exit_code', 0)
    stderr = execution_result.get('stderr', '')

    if exit_code != 0:
        if exit_code == 137 or "Killed" in stderr:
            return "resource-limit", f"Process killed (likely OOM). Exit code: {exit_code}.\n{stderr}", stderr

        if "SyntaxError:" in stderr or "IndentationError:" in stderr:
            return "syntax", stderr, stderr

        return "runtime", stderr, stderr

    if not evaluation_success:
        return "metric-regression", "Candidate failed proxy evaluation thresholds.", ""

    return "success", "", ""
