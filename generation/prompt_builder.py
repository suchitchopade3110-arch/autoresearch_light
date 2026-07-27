from typing import Dict, Any, List
from memory.db import ExperimentDB

class PromptBuilder:
    def __init__(self, db: ExperimentDB, config: Dict[str, Any]):
        self.db = db
        self.max_failures = config.get("max_retrieved_failures", 2)
        self.max_successes = config.get("max_retrieved_successes", 2)
        self.token_budget = config.get("prompt_char_budget", 4000) # Simple char budget for now

    def build_prompt(self, research_goal: str) -> str:
        """
        Builds a prompt using the research goal and retrieved context.
        """
        prompt = f"System Goal: {research_goal}\n\n"

        # Retrieve past failures
        failures = self.db.retrieve_experiments(
            query=research_goal,
            k=self.max_failures,
            filter_outcome="failure"
        )

        if failures:
            prompt += "--- PAST FAILURES TO AVOID ---\n"
            for f in failures:
                prompt += f"Hypothesis: {f['hypothesis']}\n"
                prompt += f"Failure Reason: {f['failure_reason'][:500]}\n" # Trim long traces
                # The real diagnostic (a git-apply error or the candidate's
                # actual stderr), when one exists - distinct from
                # failure_reason above, which for several failure
                # categories is just a human-readable label with no
                # underlying trace (see memory/failure_analysis.py). Feeds
                # the real signal into the next generation, not just the
                # category it was filed under.
                if f.get('traceback'):
                    prompt += f"Traceback:\n{f['traceback'][:1000]}\n"
                prompt += f"Diff:\n{f['diff']}\n\n"

        # Retrieve past successes
        successes = self.db.retrieve_experiments(
            query=research_goal,
            k=self.max_successes,
            filter_outcome="success"
        )

        if successes:
            prompt += "--- PAST SUCCESSFUL PATTERNS ---\n"
            for s in successes:
                prompt += f"Hypothesis: {s['hypothesis']}\n"
                prompt += f"Metrics: {s['metrics']}\n"
                prompt += f"Diff:\n{s['diff']}\n\n"

        prompt += "--- INSTRUCTIONS ---\n"
        prompt += "Generate a unified diff to advance the goal, avoiding past failures and building on successes. The diff must be ready to apply."

        # Enforce budget roughly
        if len(prompt) > self.token_budget:
            prompt = prompt[:self.token_budget] + "\n...[TRUNCATED]"

        return prompt
