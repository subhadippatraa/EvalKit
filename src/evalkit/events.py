"""Structured run events (docs/TARGET-ARCHITECTURE.md §12): what happened, to which unit, how long,
and why it failed -- never *what was said*.

    run.started / run.finished        one per execute
    unit.abandoned                    a unit left pending (budget / replay miss)
    target.finished / evaluator.finished   (DEBUG) one per stage, with its duration
    target.failed / evaluator.failed  a classified failure was recorded
    provider.call                     (DEBUG) one per attempt: outcome, duration, tokens, cost
    provider.retry                    a failed attempt that will be retried
    cache.hit / cache.miss            (DEBUG)
    retry_failed.reopened             units reopened by `--retry-failed`
    budget.exhausted                  a budget stopped calls

Privacy by construction: `emit` accepts only the field names in `FIELDS`, and only scalar values
(strings are scrubbed of secrets and clipped). There is no field for a prompt, an output, a
reference, a rubric or a provider's error text, so no caller can log one by accident; a failure is
identified by its class and kind, not its message. There is no option to turn content on.

Configuration is the ordinary `logging` machinery: nothing is emitted until a handler is installed
(`configure_logging`, or `EVALKIT_LOG_LEVEL` / `EVALKIT_LOG_FORMAT` / `EVALKIT_LOG_FILE` through
`configure_from_env`), or the application attaches its own handler to the `evalkit.events` logger.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from evalkit.redact import scrub

LOGGER = "evalkit.events"
_log = logging.getLogger(LOGGER)
_log.addHandler(logging.NullHandler())  # a library emits nothing until the application asks
_MAX_STR = 256

# every field an event may carry: identifiers, counters, durations, classes -- no content
FIELDS = frozenset(
    {
        "run_id", "case_key", "evaluator", "stage", "attempt", "retry_round", "duration_ms",
        "failure_class", "failure_kind", "error_type", "http_status", "request_id",
        "correlation_id", "provider", "model", "outcome", "input_tokens", "output_tokens",
        "cost_usd", "cache_hit", "cache_mode", "reason", "status", "stop_reason", "cases",
        "units", "reopened_cases", "reopened_evaluators", "calls", "cache_hits", "retries",
        "tokens", "budget", "spent", "limit", "backoff_ms", "elapsed_s", "execution",
    }
)  # fmt: skip
_HANDLER_ATTR = "_evalkit_handler"


def _clean(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return scrub(value)[:_MAX_STR]
    return None


def emit(event: str, level: int = logging.INFO, **fields: Any) -> None:
    if not _log.isEnabledFor(level):
        return
    clean = {}
    for name, value in fields.items():
        if name in FIELDS and (v := _clean(value)) is not None:
            clean[name] = v
    _log.log(level, event, extra={"evalkit_event": event, "evalkit_fields": clean})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        body = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": getattr(record, "evalkit_event", record.getMessage()),
            **getattr(record, "evalkit_fields", {}),
        }
        return json.dumps(body, ensure_ascii=True, allow_nan=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in getattr(record, "evalkit_fields", {}).items())
        ts = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        event = getattr(record, "evalkit_event", record.getMessage())
        return f"{ts} {record.levelname} {event} {fields}".rstrip()


def reset_logging() -> None:
    """Remove the handler `configure_logging` installed."""
    for h in list(_log.handlers):
        if getattr(h, _HANDLER_ATTR, False):
            _log.removeHandler(h)
            h.close()
    _log.propagate = True
    _log.setLevel(logging.NOTSET)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    *,
    file: str | Path | None = None,
    stream: TextIO | None = None,
) -> None:
    """Install (replacing any earlier one) a handler on the event logger: JSON lines or key=value
    text, to `file` (created owner-only), `stream`, or stderr."""
    lvl = logging.getLevelName(str(level).upper())
    if not isinstance(lvl, int):
        raise ValueError(f"unknown log level {level!r}")
    if fmt not in ("json", "text"):
        raise ValueError("log format must be 'json' or 'text'")
    reset_logging()
    if file is not None:
        fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        handler: logging.Handler = logging.StreamHandler(os.fdopen(fd, "a", encoding="utf-8"))
        handler.terminator = "\n"
    else:
        handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    setattr(handler, _HANDLER_ATTR, True)
    _log.addHandler(handler)
    _log.setLevel(lvl)
    _log.propagate = False


def configure_from_env(env: Mapping[str, str] | None = None, *, stream: TextIO | None = None):
    """Configure from EVALKIT_LOG_LEVEL / EVALKIT_LOG_FORMAT / EVALKIT_LOG_FILE. Returns whether
    logging was switched on (it stays off unless a level or file is given)."""
    env = os.environ if env is None else env
    level, file = env.get("EVALKIT_LOG_LEVEL"), env.get("EVALKIT_LOG_FILE")
    if not level and not file:
        return False
    configure_logging(
        level or "INFO", env.get("EVALKIT_LOG_FORMAT") or "json", file=file or None, stream=stream
    )
    return True
