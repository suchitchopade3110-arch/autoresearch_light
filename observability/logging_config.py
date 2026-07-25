import json
import logging
import sys
from typing import Any, Dict, Union

# Every attribute a stdlib LogRecord carries by default - anything else set
# on a record (via extra=... or a LoggerAdapter's bound context) is candidate
# run/candidate correlation data and belongs in the JSON payload.
_RESERVED_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """
    One JSON object per line: timestamp, level, logger name, the message,
    and any extra context (run_id, candidate_id, ...) - so a record can be
    correlated back to the run/candidate that produced it by filtering a
    field, instead of grepping free-text.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ContextAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges its bound context into every record's extra."""

    def process(self, msg, kwargs):
        merged = {**self.extra, **kwargs.get("extra", {})}
        kwargs["extra"] = merged
        return msg, kwargs


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Idempotent: safe to call from multiple entry points without installing
    duplicate handlers on repeated calls (e.g. across tests in the same
    process). Writes to stdout - the operational log is this program's
    primary output, not a side channel.
    """
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _configured = True


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """
    Returns a LoggerAdapter for `name` with `context` (e.g. run_id,
    candidate_id) merged into every record it emits.
    """
    return _ContextAdapter(logging.getLogger(name), context)


def bind(logger: Union[logging.Logger, logging.LoggerAdapter], **context: Any) -> logging.LoggerAdapter:
    """
    Returns a new adapter carrying `logger`'s existing bound context (if
    any) plus `context` - e.g. bind(run_logger, candidate_id=c_id) to add
    per-candidate correlation on top of a run-level logger.
    """
    if isinstance(logger, logging.LoggerAdapter):
        merged = {**logger.extra, **context}
        return get_logger(logger.logger.name, **merged)
    return get_logger(logger.name, **context)
