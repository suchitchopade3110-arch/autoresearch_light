from abc import ABC, abstractmethod
import os
import subprocess
import tempfile
from typing import Optional, Tuple

class LLMClient(ABC):
    @abstractmethod
    def generate_diff(self, prompt: str, target_file: str, current_content: str) -> str:
        """
        Generates a valid unified diff for target_file based on the prompt
        and the file's actual current content - without current_content, a
        diff is generated against a file state that may have already moved
        on (e.g. a prior candidate in the same generation already merged),
        which is guaranteed to fail to apply.
        """
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

    def generate_diff(self, prompt: str, target_file: str, current_content: str = "") -> str:
        added = self._SCRIPT_LINES
        header = f"@@ -1 +1,{len(added)} @@"
        body = "".join(f"+{line}\n" for line in added)
        return f"--- a/{target_file}\n+++ b/{target_file}\n{header}\n-\n{body}"


def _strip_fences(text: str) -> str:
    """Some models wrap a diff in a markdown code fence despite instructions not to - strip it if present."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    text = text.strip()
    return text + "\n" if text else ""


def _scratch_check_applies(target_file: str, current_content: str, diff: str) -> Tuple[bool, str]:
    """
    Checks whether `diff` applies cleanly against `current_content` by
    materializing both in a throwaway git repo. AnthropicClient has no
    worktree of its own - generate_diff's contract is stateless - so this
    scratch repo is the only way to validate a diff before returning it.
    Returns (applies, stderr) - stderr is "" on success.
    """
    if not diff.strip():
        return False, "empty diff"
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "scratch@example.com"], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Scratch"], cwd=d, check=True, capture_output=True)

        target_path = os.path.join(d, target_file)
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target_path, "w", newline='') as f:
            f.write(current_content)

        subprocess.run(["git", "add", "."], cwd=d, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "scratch"], cwd=d, check=True, capture_output=True)

        fd, patch_file = tempfile.mkstemp(suffix=".patch", dir=d)
        try:
            with os.fdopen(fd, "w", newline='') as f:
                f.write(diff)
            result = subprocess.run(["git", "apply", "--check", patch_file], cwd=d, capture_output=True)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.decode(errors="replace")
        finally:
            if os.path.exists(patch_file):
                os.remove(patch_file)


class AnthropicClient(LLMClient):
    """
    Real LLM-backed candidate generator using the Anthropic Messages API.
    Requires the `anthropic` package and an ANTHROPIC_API_KEY environment
    variable - never read from config, so a committed config file can
    never leak a key.

    On a git-apply-check failure (checked in a throwaway scratch repo,
    since this client has no worktree of its own), retries with the
    actual stderr fed back into the next prompt rather than blindly
    resampling.
    """
    DIFF_SYSTEM_PROMPT = (
        "You generate a single unified diff that replaces the ENTIRE contents "
        "of one target file, given its current full content. Output ONLY a "
        "valid unified diff (--- a/<file>, +++ b/<file>, one or more @@ hunks) "
        "that applies cleanly with `git apply` against the shown current "
        "content - no prose, no explanation, no markdown code fences before "
        "or after the diff. Every hunk header's line counts must exactly "
        "match the number of context/removed/added lines that follow it."
    )

    # Anthropic per-million-token pricing (input, output) in USD, current as
    # of 2026-07 - update if pricing changes. Used only for a rough
    # per-candidate cost estimate logged into metrics/reporting, never for
    # gating a merge decision.
    _PRICING_PER_MTOK = {
        "claude-sonnet-5": (2.00, 10.00),
        "claude-opus-4-8": (5.00, 25.00),
        "claude-haiku-4-5": (1.00, 5.00),
    }

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 4000, max_apply_retries: int = 3):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.max_apply_retries = max_apply_retries
        # Populated after every generate_diff() call - the caller pulls this
        # to attribute cost/tokens to the candidate. Never present on
        # MockLLMClient, so callers must getattr(..., "last_usage", None).
        self.last_usage = {}

    def generate_diff(self, prompt: str, target_file: str, current_content: str = "") -> str:
        feedback = ""
        diff = ""
        total_input = 0
        total_output = 0

        for _ in range(self.max_apply_retries):
            user_content = f"{prompt}\n\n--- CURRENT {target_file} ---\n{current_content}"
            if feedback:
                user_content += f"\n\n--- PREVIOUS ATTEMPT FAILED TO APPLY (git apply --check stderr) ---\n{feedback}"

            # temperature/top_p/top_k are deliberately never sent: Claude
            # Sonnet 5 (and the Opus 4.7/4.8 family) reject non-default
            # sampling parameters outright.
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.DIFF_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            diff = _strip_fences("".join(b.text for b in response.content if b.type == "text"))

            applies, stderr = _scratch_check_applies(target_file, current_content, diff)
            if applies:
                break
            feedback = stderr

        input_rate, output_rate = self._PRICING_PER_MTOK.get(self.model, (0.0, 0.0))
        self.last_usage = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost_usd": round((total_input * input_rate + total_output * output_rate) / 1_000_000, 6),
        }
        return diff

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
        file_path = os.path.join(cwd, target_file) if cwd else target_file
        try:
            with open(file_path) as f:
                current_content = f.read()
        except FileNotFoundError:
            current_content = ""

        diff = self.llm_client.generate_diff(prompt, target_file, current_content)

        print("Generated diff:")
        print(diff)

        return validate_and_apply_patch(diff, cwd=cwd), diff
