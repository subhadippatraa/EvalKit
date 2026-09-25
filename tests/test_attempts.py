"""F-2: every judge call is recorded; a retry-success never looks like a first-try success."""

import hashlib
import json

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import Evaluator, JudgeError, JudgeOutputError, JudgeTimeoutError
from evalkit.limits import MAX_ERROR_CHARS, MAX_EVIDENCE_BYTES


def run(store, judge, **kwargs):
    return Evaluator(judge, store).evaluate("Explain DI.", "DI is ...", criteria=CRITERIA, **kwargs)


def stored_attempts(store):
    (row,) = store.list()
    return row.attempts


def test_a_first_try_success_records_one_ok_attempt_without_raw_output(store):
    result = run(store, FakeJudge(judged(correctness=5, clarity=4)))
    (a,) = result.attempts
    assert (a.n, a.outcome, a.error, a.error_type, a.raw, a.raw_truncated) == (
        1, "ok", None, None, None, False,
    )  # fmt: skip
    assert a.raw_sha256 and a.duration_ms >= 0 and a.started_at is not None
    assert stored_attempts(store) == result.attempts  # persisted and round-trips exactly


def test_rejected_output_then_successful_retry_is_visible_as_two_attempts(store):
    rejected = judged(correctness=9, clarity=3)  # 9 is outside the 1-5 scale
    judge = FakeJudge(rejected, judged(correctness=5, clarity=5))
    result = run(store, judge)

    assert result.status == "ok" and len(judge.calls) == 2
    first, second = result.attempts
    assert (first.n, first.outcome, first.error_type) == (1, "invalid_output", "JudgeOutputError")
    assert "outside scale" in first.error
    assert json.loads(first.raw) == rejected  # the rejected payload is kept, verbatim
    assert first.raw_sha256 == hashlib.sha256(first.raw.encode()).hexdigest()
    assert (second.n, second.outcome, second.raw) == (2, "ok", None)
    assert first.started_at <= second.started_at
    assert stored_attempts(store) == [first, second]  # not collapsed into "succeeded"


def test_two_rejected_outputs_are_both_kept_on_the_error_row(store):
    judge = FakeJudge(judged(correctness=4), judged(correctness=4, clarity=0))
    with pytest.raises(JudgeOutputError) as info:
        run(store, judge)
    (row,) = store.list()
    assert row.id == info.value.evaluation_id and row.status == "error"
    a1, a2 = row.attempts
    assert (a1.outcome, a2.outcome) == ("invalid_output", "invalid_output")
    assert "missing" in a1.error and "outside scale" in a2.error
    assert json.loads(a1.raw) == judged(correctness=4)
    assert json.loads(a2.raw) == judged(correctness=4, clarity=0)


def test_provider_raw_evidence_and_kind_are_recorded(store):
    partial = {"stopReason": "guardrail_intervened", "message": {"content": [{"text": "blocked"}]}}
    err = JudgeOutputError("judge response was blocked", kind="refused", raw=partial)
    judge = FakeJudge(err, JudgeOutputError("again", kind="refused", raw=partial))
    with pytest.raises(JudgeOutputError):
        run(store, judge)
    (a1, a2) = stored_attempts(store)
    assert a1.kind == "refused" and json.loads(a1.raw) == partial


@pytest.mark.parametrize(
    ("exc", "outcome", "type_name"),
    [
        (JudgeError("Bedrock ThrottlingException: slow down"), "provider_error", "JudgeError"),
        (JudgeTimeoutError("timed out"), "timeout", "JudgeTimeoutError"),
    ],
)
def test_non_retried_failures_record_exactly_one_attempt(store, exc, outcome, type_name):
    judge = FakeJudge(exc, judged(correctness=5, clarity=5))
    with pytest.raises(JudgeError):
        run(store, judge)
    (a,) = stored_attempts(store)
    assert (a.n, a.outcome, a.error_type, a.raw) == (1, outcome, type_name, None)
    assert len(judge.calls) == 1


def test_an_unexpected_exception_from_a_custom_judge_is_recorded(store):
    with pytest.raises(JudgeError, match="RuntimeError: bug"):
        run(store, FakeJudge(RuntimeError("bug")))
    (a,) = stored_attempts(store)
    assert a.outcome == "unexpected_error" and "RuntimeError: bug" in a.error


def test_a_provider_error_during_the_retry_is_the_second_attempt(store):
    judge = FakeJudge(judged(correctness=0, clarity=0), JudgeTimeoutError("timed out"))
    with pytest.raises(JudgeTimeoutError):
        run(store, judge)
    assert [a.outcome for a in stored_attempts(store)] == ["invalid_output", "timeout"]


def test_raw_output_is_truncated_to_the_documented_limit_but_hashed_in_full(store):
    huge = {
        "correctness": {"reasoning": "x" * 300_000, "score": 99},
        "clarity": {"reasoning": "y", "score": 1},
    }
    with pytest.raises(JudgeOutputError):
        run(store, FakeJudge(huge, huge))
    a = stored_attempts(store)[0]
    full = json.dumps(huge, ensure_ascii=False, default=repr)
    assert a.raw_truncated is True
    assert len(a.raw.encode()) <= MAX_EVIDENCE_BYTES
    assert full.startswith(a.raw)
    assert a.raw_sha256 == hashlib.sha256(full.encode()).hexdigest()


def test_multibyte_evidence_is_cut_on_a_character_boundary(store):
    huge = {
        "correctness": {"reasoning": "日" * 100_000, "score": 99},
        "clarity": {"reasoning": "y", "score": 1},
    }
    with pytest.raises(JudgeOutputError):
        run(store, FakeJudge(huge, huge))
    a = stored_attempts(store)[0]
    assert a.raw_truncated and len(a.raw.encode()) <= MAX_EVIDENCE_BYTES
    a.raw.encode("utf-8")  # valid text, no half-character


def test_long_error_messages_are_clipped(store):
    with pytest.raises(JudgeError):
        run(store, FakeJudge(JudgeError("boom " * 5000)))
    (row,) = store.list()
    assert len(row.error) <= MAX_ERROR_CHARS and len(row.attempts[0].error) <= MAX_ERROR_CHARS


def test_provider_error_text_is_scrubbed_before_it_is_stored_or_raised(store):
    leaky = (
        "AccessDeniedException: User: arn:aws:sts::123456789012:assumed-role/eval/session is not "
        "authorized; Authorization: Bearer abcdef1234567890 api_key=sk-abcdefghijklmnopqrstuvwx"
    )
    with pytest.raises(JudgeError) as info:
        run(store, FakeJudge(JudgeError(leaky)))
    (row,) = store.list()
    for text in (str(info.value), row.error, row.attempts[0].error):
        assert "123456789012" not in text and "abcdef1234567890" not in text
        assert "sk-abcdefghij" not in text and "<redacted>" in text


def test_unpaired_surrogates_in_provider_messages_do_not_break_persistence(store):
    with pytest.raises(JudgeError):
        run(store, FakeJudge(JudgeError("bad \ud800 text")))
    (row,) = store.list()  # would have raised UnicodeEncodeError inside store.save
    assert row.status == "error"


def test_attempts_appear_in_the_serialized_result(store):
    result = run(
        store, FakeJudge(judged(correctness=9, clarity=1), judged(correctness=5, clarity=5))
    )
    dumped = result.model_dump(mode="json")
    assert [a["outcome"] for a in dumped["attempts"]] == ["invalid_output", "ok"]


def test_sdk_retries_stay_disabled_and_the_judge_is_called_at_most_twice(store):
    """EvalKit owns retry behavior: at most one retry, and only for malformed output."""
    judge = FakeJudge(*(judged(correctness=0, clarity=0) for _ in range(5)))
    with pytest.raises(JudgeOutputError):
        run(store, judge)
    assert len(judge.calls) == 2
    # the SDK-level switches are asserted against real clients in test_bedrock*.py:
    # test_real_client_has_sdk_retries_disabled_*
