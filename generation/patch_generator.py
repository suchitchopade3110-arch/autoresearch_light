from abc import ABC, abstractmethod
import os
import subprocess
import tempfile
from typing import Optional

class LLMClient(ABC):
    @abstractmethod
    def generate_diff(self, prompt: str, target_file: str) -> str:
        """Generates a valid unified diff for the target file based on the prompt."""
        pass

class MockLLMClient(LLMClient):
    """
    A mock LLM client for testing. Returns a deterministic patch that
    replaces the (blank placeholder) target file with a script that
    implements the honest solution to the synthetic task: it reads
    TRAIN_PATH/TEST_PATH (never the held-out labels - it can't, they're
    never mounted) and writes real predictions to /app/out/predictions.jsonl,
    which is what the eval pipeline actually scores. It also prints a
    "SCORE:" line as a diagnostic self-estimate, which the eval pipeline
    parses only to detect a mismatch against the real score - never to
    gate on.
    TODO: Wave 2 - Implement a real LLMClient (e.g. AnthropicClient).
    """
    # The hunk header's added-line count is derived from this list rather
    # than hand-counted, so it can never again silently drift out of sync
    # with the actual body (see test_mock_llm_diff_hunk_header_matches_actual_line_count -
    # a wrong declared count is what a stricter git silently truncates the
    # patch to, rather than rejecting outright).
    _SCRIPT_LINES = [
        "import json",
        "import os",
        "import random",
        "",
        'TRAIN_PATH = os.environ.get("TRAIN_PATH", "/app/data/train.jsonl")',
        'TEST_PATH = os.environ.get("TEST_PATH", "/app/data/test.jsonl")',
        'SUBSET_PERCENTAGE = float(os.environ.get("SUBSET_PERCENTAGE", "100"))',
        'SEED = int(os.environ.get("DATASET_SEED", "42"))',
        'OUT_PATH = "/app/out/predictions.jsonl"',
        "",
        "",
        "def load_jsonl(path):",
        "    with open(path) as f:",
        "        return [json.loads(line) for line in f if line.strip()]",
        "",
        "",
        "def subset(rows, percentage, seed):",
        "    rng = random.Random(seed)",
        "    order = list(range(len(rows)))",
        "    rng.shuffle(order)",
        "    k = max(1, int(len(rows) * percentage / 100))",
        "    return [rows[i] for i in order[:k]]",
        "",
        "",
        "def predict(x1, x2):",
        "    return 1 if (2.0 * x1 - 1.0 * x2) > 0 else 0",
        "",
        "",
        "def main():",
        "    train_rows = subset(load_jsonl(TRAIN_PATH), SUBSET_PERCENTAGE, SEED)",
        "    test_rows = load_jsonl(TEST_PATH)",
        "",
        "    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)",
        '    with open(OUT_PATH, "w") as f:',
        "        for row in test_rows:",
        '            pred = predict(row["x1"], row["x2"])',
        '            f.write(json.dumps({"id": row["id"], "pred": pred}) + "\\n")',
        "",
        '    correct = sum(1 for r in train_rows if predict(r["x1"], r["x2"]) == r["label"])',
        "    train_accuracy = correct / len(train_rows) if train_rows else 0.0",
        '    print(f"SCORE: {train_accuracy:.4f}", flush=True)',
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
    ]

    def generate_diff(self, prompt: str, target_file: str) -> str:
        added = self._SCRIPT_LINES
        header = f"@@ -1 +1,{len(added)} @@"
        body = "".join(f"+{line}\n" for line in added)
        return f"--- a/{target_file}\n+++ b/{target_file}\n{header}\n-\n{body}"

def validate_and_apply_patch(diff_content: str, cwd: Optional[str] = None, dry_run: bool = False) -> bool:
    """
    Validates a patch by attempting to apply it cleanly, then applies it
    unless `dry_run` is set. `cwd` is the git working tree the patch should
    be applied against (a candidate's own worktree) - defaults to the
    caller's current directory if omitted, matching the pre-worktree
    behavior.

    Use dry_run=True for a validity check with no side effects (e.g.
    checking a candidate's diff is even applicable before scheduling it) -
    without it, every successful call mutates `cwd`'s working tree for
    real, so checking the same diff twice would fail the second time.
    Returns True if successful, False otherwise.
    """
    fd, patch_file = tempfile.mkstemp(suffix=".patch")
    try:
        # newline='' disables Python's platform line-ending translation, so
        # the diff's own '\n' bytes are written through unchanged. Without
        # it, the default text mode on Windows rewrites every '\n' to
        # '\r\n' - including inside the patch file's own hunk lines - which
        # confuses git apply's line-based hunk parser and causes it to
        # silently apply only part of the hunk (observed: the last two
        # lines of a 30-line hunk went missing, with no error at all).
        with os.fdopen(fd, "w", newline='') as f:
            f.write(diff_content)

        subprocess.run(["git", "apply", "--check", patch_file], check=True, capture_output=True, cwd=cwd)
        if not dry_run:
            subprocess.run(["git", "apply", patch_file], check=True, capture_output=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Patch validation/application failed: {e.stderr}")
        return False
    finally:
        if os.path.exists(patch_file):
            os.remove(patch_file)

class PatchGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_and_apply(self, prompt: str, target_file: str, cwd: Optional[str] = None) -> bool:
        """
        Generates a patch and attempts to apply it in `cwd` (a candidate's
        own worktree). Returns (success, diff).
        """
        diff = self.llm_client.generate_diff(prompt, target_file)

        print("Generated diff:")
        print(diff)

        return validate_and_apply_patch(diff, cwd=cwd), diff
