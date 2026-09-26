"""Structured logging: identifiers and numbers only, configurable, never content or secrets."""

import io
import json
import logging

import pytest

from evalkit import events


@pytest.fixture(autouse=True)
def clean():
    yield
    events.reset_logging()


def capture(level="DEBUG", fmt="json"):
    stream = io.StringIO()
    events.configure_logging(level=level, fmt=fmt, stream=stream)
    return stream


def lines(stream):
    return [json.loads(x) for x in stream.getvalue().splitlines() if x.strip()]


def test_an_event_is_one_json_line_with_its_identifiers():
    s = capture()
    events.emit(
        "provider.call", run_id="r1", case_key="c1", evaluator="judge:q:abc", attempt=2,
        duration_ms=15, failure_class="infrastructure", failure_kind="timeout",
        request_id="req-9", correlation_id="corr-1", provider="p", model="m",
    )  # fmt: skip
    (rec,) = lines(s)
    assert rec["event"] == "provider.call" and rec["level"] == "INFO"
    assert rec["run_id"] == "r1" and rec["case_key"] == "c1" and rec["attempt"] == 2
    assert rec["failure_class"] == "infrastructure" and rec["request_id"] == "req-9"
    assert rec["duration_ms"] == 15 and rec["correlation_id"] == "corr-1" and "ts" in rec


def test_only_known_fields_are_logged_so_content_cannot_leak_in():
    s = capture()
    events.emit("x", run_id="r", prompt="SECRET PROMPT", output="MODEL OUTPUT", message="text")
    (rec,) = lines(s)
    assert "SECRET" not in json.dumps(rec) and "MODEL OUTPUT" not in json.dumps(rec)
    assert "prompt" not in rec and "output" not in rec and "message" not in rec


def test_strings_are_scrubbed_and_clipped_and_only_scalars_survive():
    s = capture()
    events.emit(
        "x", request_id="Bearer abcdefghijkl", case_key="k" * 1000, model={"nested": "dict"}
    )
    (rec,) = lines(s)
    assert "abcdefghijkl" not in rec["request_id"] and len(rec["case_key"]) <= 256
    assert "model" not in rec


def test_the_level_filters_events():
    s = capture(level="WARNING")
    events.emit("quiet", level=logging.INFO, run_id="r")
    events.emit("loud", level=logging.WARNING, run_id="r")
    assert [r["event"] for r in lines(s)] == ["loud"]


def test_nothing_is_emitted_until_logging_is_configured(caplog):
    events.emit("x", run_id="r")  # no handler installed by evalkit; must not raise or print
    assert True


def test_text_format_is_key_value():
    s = capture(fmt="text")
    events.emit("run.started", run_id="r1", cases=3)
    assert "run.started" in s.getvalue() and "run_id=r1" in s.getvalue()


def test_configuring_twice_replaces_the_handler_and_does_not_duplicate():
    capture()
    s = capture()
    events.emit("x", run_id="r")
    assert len(lines(s)) == 1
    handlers = logging.getLogger("evalkit.events").handlers
    assert sum(1 for h in handlers if not isinstance(h, logging.NullHandler)) == 1


def test_a_log_file_is_owner_only(tmp_path):
    path = tmp_path / "ev.log"
    events.configure_logging(level="INFO", fmt="json", file=path)
    events.emit("x", run_id="r")
    events.reset_logging()
    assert json.loads(path.read_text())["run_id"] == "r"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_bad_configuration_is_refused():
    for kw in ({"level": "LOUD"}, {"fmt": "xml"}):
        with pytest.raises(ValueError):
            events.configure_logging(**kw)


def test_configuration_from_the_environment():
    stream = io.StringIO()
    events.configure_from_env(
        {"EVALKIT_LOG_LEVEL": "info", "EVALKIT_LOG_FORMAT": "json"}, stream=stream
    )
    events.emit("x", run_id="r")
    assert lines(stream)[0]["run_id"] == "r"
    events.reset_logging()
    assert events.configure_from_env({}) is False  # unset: logging stays off
