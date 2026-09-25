"""The failure taxonomy (design section 7): input | target | evaluator | infrastructure."""

import pytest
from pydantic import ValidationError

from evalkit import EvalFailure, Failure, FailureClass
from evalkit.failures import (
    CASE_RESULT_CLASSES,
    EVALUATOR_RESULT_CLASSES,
    KINDS,
    check_kind,
)

# design section 7.2, transcribed independently of the implementation
EXPECTED = {
    "input": {"invalid_case", "oversize", "missing_field", "duplicate_key", "bad_config"},
    "target": {"exception", "timeout", "contract_violation", "blocked", "empty_output"},
    "evaluator": {"invalid_output", "truncated", "refused", "bad_request", "internal_error"},
    "infrastructure": {
        "rate_limited", "provider_unavailable", "timeout", "connection", "auth",
        "quota_exhausted", "storage", "budget_exceeded", "deadline_exceeded", "cancelled",
    },
}  # fmt: skip
RETRYABLE = {
    ("target", "timeout"), ("evaluator", "invalid_output"), ("infrastructure", "rate_limited"),
    ("infrastructure", "provider_unavailable"), ("infrastructure", "timeout"),
    ("infrastructure", "connection"),
}  # fmt: skip
SYSTEMIC = {
    ("evaluator", "bad_request"), ("infrastructure", "auth"), ("infrastructure", "quota_exhausted"),
}  # fmt: skip


def test_the_four_classes_are_exactly_those_of_the_design():
    assert {c.value for c in FailureClass} == set(EXPECTED)
    assert {c.value: set(kinds) for c, kinds in KINDS.items()} == EXPECTED


@pytest.mark.parametrize("cls,kind", [(c, k) for c, ks in EXPECTED.items() for k in sorted(ks)])
def test_every_kind_has_the_design_retryability_and_systemic_flags(cls, kind):
    f = Failure(failure_class=cls, kind=kind, message="m")
    assert f.retryable is ((cls, kind) in RETRYABLE)
    assert f.systemic is ((cls, kind) in SYSTEMIC)


def test_retryable_can_be_overridden_when_a_retry_budget_is_spent():
    f = Failure(failure_class="infrastructure", kind="rate_limited", message="m", retryable=False)
    assert f.retryable is False
    # ...and an explicit False is not mistaken for "omitted"
    assert Failure.model_validate_json(f.model_dump_json()).retryable is False


@pytest.mark.parametrize(
    "cls,kind", [("target", "auth"), ("infrastructure", "exception"), ("input", "timeout")]
)
def test_a_kind_from_another_class_is_refused(cls, kind):
    with pytest.raises(ValidationError, match="unknown failure kind"):
        Failure(failure_class=cls, kind=kind, message="m")


def test_the_same_kind_name_in_two_classes_stays_two_failures():
    """`timeout` exists for both target and infrastructure; the class is what attributes it."""
    target = Failure(failure_class="target", kind="timeout", message="m")
    infra = Failure(failure_class="infrastructure", kind="timeout", message="m")
    assert target != infra


def test_unknown_class_and_empty_message_are_refused():
    with pytest.raises(ValidationError):
        Failure(failure_class="model", kind="timeout", message="m")
    with pytest.raises(ValidationError):
        Failure(failure_class="target", kind="timeout", message="")


def test_messages_are_scrubbed_of_secrets_and_clipped():
    f = Failure(
        failure_class="infrastructure",
        kind="auth",
        message="denied for arn:aws:iam::123456789012:role/x key sk-abcdefghijklmnopqrstuvwx",
    )
    assert "123456789012" not in f.message and "sk-abcdef" not in f.message
    long = Failure(failure_class="target", kind="exception", message="x" * 10_000)
    assert len(long.message) == 2000 and long.message.endswith("…")


def test_a_message_with_a_lone_surrogate_becomes_encodable():
    f = Failure(failure_class="target", kind="exception", message="bad \ud800 char")
    f.message.encode("utf-8")


def test_a_failure_is_immutable():
    f = Failure(failure_class="target", kind="exception", message="m")
    with pytest.raises(ValidationError):
        f.kind = "timeout"


def test_the_store_accepts_each_class_only_where_it_belongs():
    assert {c.value for c in CASE_RESULT_CLASSES} == {"input", "target", "infrastructure"}
    assert {c.value for c in EVALUATOR_RESULT_CLASSES} == {"input", "evaluator", "infrastructure"}
    # a target failure is never an evaluator result, an evaluator failure never a case result
    assert FailureClass.TARGET not in EVALUATOR_RESULT_CLASSES
    assert FailureClass.EVALUATOR not in CASE_RESULT_CLASSES


def test_eval_failure_is_a_plain_raisable_exception_carrying_the_record():
    try:
        raise EvalFailure(
            "infrastructure", "rate_limited", "429 slow down", http_status=429,
            request_id="req-1", provider="bedrock", retry_after_s=2.5,
        )  # fmt: skip
    except EvalFailure as e:
        assert isinstance(e, Exception) and str(e) == "429 slow down"
        assert (e.failure_class, e.kind, e.retryable, e.systemic) == (
            FailureClass.INFRA, "rate_limited", True, False
        )  # fmt: skip
        assert (e.http_status, e.request_id, e.provider, e.retry_after_s) == (
            429, "req-1", "bedrock", 2.5
        )  # fmt: skip
        assert e.failure == Failure(
            failure_class="infrastructure", kind="rate_limited", message="429 slow down"
        )


def test_eval_failure_refuses_an_unknown_kind_at_the_raise_site():
    with pytest.raises(ValidationError):
        EvalFailure("target", "made_up", "m")


def test_check_kind_reports_the_known_kinds():
    with pytest.raises(ValueError, match="known: .*exception"):
        check_kind(FailureClass.TARGET, "nope")
