import io
import json
import logging

from observability.logging_config import JsonFormatter, bind, get_logger


def _make_logger_with_capture(name):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger, stream


def test_json_formatter_emits_one_parseable_json_object_with_message_preserved():
    logger, stream = _make_logger_with_capture("test.json_formatter")

    logger.info("Candidate passed all evaluation stages.")

    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["message"] == "Candidate passed all evaluation stages."
    assert record["level"] == "INFO"
    assert record["logger"] == "test.json_formatter"
    assert "timestamp" in record
    # Substring checks against raw log output (as the integration tests do
    # against subprocess stdout) must still find the original message text.
    assert "Candidate passed all evaluation stages." in line


def test_bound_context_appears_in_every_record():
    logger, stream = _make_logger_with_capture("test.context")
    adapter = get_logger("test.context", run_id="run-123")

    adapter.info("Starting iteration")

    record = json.loads(stream.getvalue().strip())
    assert record["run_id"] == "run-123"
    assert record["message"] == "Starting iteration"


def test_bind_adds_context_on_top_of_an_existing_adapter():
    logger, stream = _make_logger_with_capture("test.bind")
    run_logger = get_logger("test.bind", run_id="run-123")
    candidate_logger = bind(run_logger, candidate_id="cand-456")

    candidate_logger.warning("Patch failed to apply")

    record = json.loads(stream.getvalue().strip())
    assert record["run_id"] == "run-123"
    assert record["candidate_id"] == "cand-456"
    assert record["level"] == "WARNING"


def test_extra_kwarg_still_merges_alongside_bound_context():
    logger, stream = _make_logger_with_capture("test.extra")
    adapter = get_logger("test.extra", run_id="run-123")

    adapter.info("Stage result", extra={"score": 0.9})

    record = json.loads(stream.getvalue().strip())
    assert record["run_id"] == "run-123"
    assert record["score"] == 0.9
