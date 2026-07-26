import ast
import os
from typing import List, Tuple

def check_syntax(file_path: str) -> Tuple[bool, str]:
    """
    Checks the syntax of the python file using ast.parse.
    Returns (success, error_message).
    """
    if not os.path.exists(file_path):
        return False, f"File {file_path} does not exist."

    with open(file_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        error_msg = f"SyntaxError in {file_path}:\nLine {e.lineno}: {e.msg}\n{e.text}"
        return False, error_msg


def check_syntax_multi(file_paths: List[str]) -> Tuple[bool, str]:
    """
    Runs check_syntax() across multiple files, returning on the first
    failure - consistent with how a single check_syntax() error message is
    used downstream (stored as one failure_reason string, not a list, so
    aggregating every file's errors would need a wider change there too).
    For a single-element list, behavior and the returned message are
    identical to calling check_syntax() directly.
    """
    for path in file_paths:
        ok, error_msg = check_syntax(path)
        if not ok:
            return False, error_msg
    return True, ""
