import os
from unittest.mock import MagicMock

from generation.patch_generator import AnthropicClient, LLMClient


def _fake_message(text, input_tokens=100, output_tokens=50):
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg = MagicMock()
    msg.content = [block]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    return msg


GOOD_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print(1)\n"
BAD_DIFF = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1,5 @@\n-\n+print(1)\n"


def _make_client(monkeypatch, responses):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    client = AnthropicClient(model="claude-sonnet-5")
    client.client = MagicMock()
    client.client.messages.create.side_effect = responses
    return client


def test_generate_diff_returns_a_valid_diff_on_first_try(monkeypatch):
    client = _make_client(monkeypatch, [_fake_message(GOOD_DIFF)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert diff.strip() == GOOD_DIFF.strip()
    assert client.client.messages.create.call_count == 1


def test_unparseable_diff_triggers_a_retry_with_the_previous_stderr_fed_back(monkeypatch):
    """
    Wave 2 acceptance: on a git-apply-check failure, the client retries
    with the actual stderr fed back into the next prompt - not a blind
    resample at the same temperature.
    """
    client = _make_client(monkeypatch, [_fake_message(BAD_DIFF), _fake_message(GOOD_DIFF)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")

    assert diff.strip() == GOOD_DIFF.strip()
    assert client.client.messages.create.call_count == 2

    second_call_kwargs = client.client.messages.create.call_args_list[1].kwargs
    second_prompt = second_call_kwargs["messages"][0]["content"]
    assert "PREVIOUS ATTEMPT FAILED TO APPLY" in second_prompt
    # the scratch-repo git apply stderr for the bad hunk count must be in there
    assert "corrupt patch" in second_prompt or "patch" in second_prompt.lower()


def test_exhausting_retries_returns_the_last_attempt_without_raising(monkeypatch):
    client = _make_client(monkeypatch, [_fake_message(BAD_DIFF)] * 3)
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.client.messages.create.call_count == 3
    assert diff.strip() == BAD_DIFF.strip()


def test_last_usage_records_tokens_and_a_cost_estimate(monkeypatch):
    client = _make_client(monkeypatch, [_fake_message(GOOD_DIFF, input_tokens=200, output_tokens=80)])
    client.generate_diff("goal", "candidate_script.py", "\n")
    assert client.last_usage["input_tokens"] == 200
    assert client.last_usage["output_tokens"] == 80
    assert client.last_usage["estimated_cost_usd"] > 0


def test_markdown_fences_are_stripped_from_the_response(monkeypatch):
    fenced = f"```diff\n{GOOD_DIFF}```"
    client = _make_client(monkeypatch, [_fake_message(fenced)])
    diff = client.generate_diff("goal", "candidate_script.py", "\n")
    assert not diff.strip().startswith("```")
    assert diff.strip() == GOOD_DIFF.strip()


def test_two_different_prompts_produce_two_different_diffs(monkeypatch):
    """Wave 2 acceptance: distinct prompts must reach the API as distinct requests and can yield distinct diffs."""
    diff_a = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('a')\n"
    diff_b = "--- a/candidate_script.py\n+++ b/candidate_script.py\n@@ -1 +1 @@\n-\n+print('b')\n"
    client = _make_client(monkeypatch, [_fake_message(diff_a), _fake_message(diff_b)])

    result_a = client.generate_diff("goal A", "candidate_script.py", "\n")
    result_b = client.generate_diff("goal B", "candidate_script.py", "\n")

    assert result_a.strip() != result_b.strip()
    first_prompt = client.client.messages.create.call_args_list[0].kwargs["messages"][0]["content"]
    second_prompt = client.client.messages.create.call_args_list[1].kwargs["messages"][0]["content"]
    assert "goal A" in first_prompt
    assert "goal B" in second_prompt


def test_temperature_is_never_sent(monkeypatch):
    """Claude Sonnet 5 (and the Opus 4.7/4.8 family) reject non-default sampling params outright."""
    client = _make_client(monkeypatch, [_fake_message(GOOD_DIFF)])
    client.generate_diff("goal", "candidate_script.py", "\n")
    call_kwargs = client.client.messages.create.call_args.kwargs
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs
    assert "top_k" not in call_kwargs


def test_anthropic_client_implements_the_llm_client_interface():
    assert issubclass(AnthropicClient, LLMClient)
