"""The call layer: retry classification, backoff, deadlines, cancellation, guard, limiter."""

import random
import threading

import pytest
from hypothesis import given
from hypothesis import strategies as st

from evalkit import EvalFailure, FailureClass
from evalkit.calls import (
    Budget,
    CallRunner,
    CancelToken,
    RateLimiter,
    RetryPolicy,
    RunGuard,
)
from evalkit.llm import LLMResponse, Usage


class Clock:
    """A fake monotonic clock that only moves when the (fake) sleep moves it."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class SleepToken(CancelToken):
    """Records the waits instead of sleeping; optionally 'cancels' at the Nth wait."""

    def __init__(self, clock, cancel_at=None):
        super().__init__()
        self.clock, self.waits, self.cancel_at = clock, [], cancel_at

    def wait(self, seconds):
        self.waits.append(seconds)
        self.clock.t += seconds
        if self.cancel_at is not None and len(self.waits) >= self.cancel_at:
            self.cancel("test")
        return self.cancelled


def runner(policy=None, *, cancel_at=None, guard=None, limiter=None):
    clock = Clock()
    token = SleepToken(clock, cancel_at)
    r = CallRunner(
        policy or RetryPolicy(),
        token=token,
        guard=guard or RunGuard(monotonic=clock),
        limiter=limiter,
        rng=random.Random(7),
        monotonic=clock,
    )
    return r, token, clock


def fail(cls, kind, msg="x", **kw):
    return EvalFailure(cls, kind, msg, **kw)


def flaky(*outcomes):
    """A send() that plays back outcomes: exceptions are raised, anything else returned."""
    calls = []

    def send(feedback):
        calls.append(feedback)
        o = outcomes[len(calls) - 1]
        if isinstance(o, BaseException):
            raise o
        return o

    send.calls = calls
    return send


# --- policy ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"max_attempts": 0}, {"max_attempts": True}, {"base_s": 0}, {"base_s": float("inf")},
        {"cap_s": 0.5, "base_s": 1.0}, {"timeout_s": -1}, {"unit_deadline_s": 0},
        {"validation_retries": 4}, {"validation_retries": -1}, {"base_s": True},
    ],
)  # fmt: skip
def test_invalid_policies_are_refused(bad):
    with pytest.raises(ValueError):
        RetryPolicy(**bad)


def test_policy_from_mapping_refuses_unknown_settings():
    assert RetryPolicy.from_mapping({"max_attempts": 2}).max_attempts == 2
    with pytest.raises(ValueError, match="unknown retry setting"):
        RetryPolicy.from_mapping({"max_attemps": 2})


def test_the_unit_deadline_defaults_to_three_timeouts():
    assert RetryPolicy(timeout_s=10).deadline_s == 30
    assert RetryPolicy(timeout_s=10, unit_deadline_s=5).deadline_s == 5


@given(n=st.integers(0, 12), ra=st.one_of(st.none(), st.floats(0, 1000)), seed=st.integers())
def test_backoff_is_full_jitter_bounded_by_the_cap_and_never_undercuts_retry_after(n, ra, seed):
    p = RetryPolicy(base_s=1.0, cap_s=30.0)
    wait = p.backoff_s(n, ra, random.Random(seed))
    ceiling = min(30.0, 2.0**n)
    assert 0 <= wait <= max(ceiling, min(ra or 0, 30.0)) + 1e-9
    if ra is not None:
        assert wait >= min(ra, 30.0) - 1e-9


@pytest.mark.parametrize(
    "f,expected",
    [
        (fail("infrastructure", "rate_limited"), 3),
        (fail("infrastructure", "provider_unavailable"), 3),
        (fail("infrastructure", "connection"), 3),
        (fail("infrastructure", "timeout"), 2),
        (fail("target", "timeout"), 1),
        (fail("target", "exception"), 0),
        (fail("target", "blocked"), 0),
        (fail("evaluator", "truncated"), 0),
        (fail("evaluator", "refused"), 0),
        (fail("evaluator", "invalid_output"), 0),  # its own path: one retry *with feedback*
        (fail("infrastructure", "auth"), 0),
        (fail("infrastructure", "quota_exhausted"), 0),
        (fail("input", "oversize"), 0),
        (fail("infrastructure", "rate_limited", retryable=False), 0),  # explicit override
    ],
)
def test_retries_follow_the_design_table(f, expected):
    assert RetryPolicy().retries_for(f) == expected


def test_a_systemic_failure_is_never_retried_even_if_it_claims_to_be_retryable():
    f = fail("infrastructure", "auth", retryable=True)
    assert f.retryable and f.systemic and RetryPolicy().retries_for(f) == 0


def test_max_attempts_caps_every_kind():
    assert RetryPolicy(max_attempts=2).retries_for(fail("infrastructure", "rate_limited")) == 1
    assert RetryPolicy(max_attempts=1).retries_for(fail("infrastructure", "rate_limited")) == 0


# --- retries, attempts, evidence -------------------------------------------------------------


def test_a_first_try_success_records_one_ok_attempt():
    r, token, _ = runner()
    unit = r.unit()
    resp = LLMResponse(payload={"a": 1}, usage=Usage(10, 4), request_id="rid")
    from evalkit.calls import describe_llm_response

    value = unit.run(lambda fb: resp, describe=describe_llm_response, provider="p", model="m")
    assert value is resp and token.waits == []
    (a,) = unit.attempts
    assert (a.n, a.outcome, a.provider, a.model, a.raw) == (1, "ok", "p", "m", None)
    assert (a.input_tokens, a.output_tokens, a.request_id) == (10, 4, "rid")
    assert a.raw_sha256  # the payload is hashed, never kept


def test_a_retry_success_is_two_attempts_with_the_failure_kept_as_evidence():
    r, token, _ = runner()
    unit = r.unit()
    boom = fail(
        "infrastructure", "rate_limited", "429", retry_after_s=2.0, http_status=429, raw={"e": 1}
    )
    assert unit.run(flaky(boom, "fine")) == "fine"
    first, second = unit.attempts
    assert (first.n, first.outcome, first.error_kind, first.http_status) == (
        1,
        "failed",
        "rate_limited",
        429,
    )
    assert first.raw and (second.n, second.outcome, second.raw) == (2, "ok", None)
    assert len(token.waits) == 1 and token.waits[0] >= 2.0  # Retry-After honoured


def test_retry_exhaustion_raises_the_last_failure_after_the_capped_number_of_attempts():
    r, token, _ = runner(RetryPolicy(max_attempts=4))
    unit = r.unit()
    send = flaky(*[fail("infrastructure", "provider_unavailable", f"503 #{i}") for i in range(10)])
    with pytest.raises(EvalFailure, match="503 #3"):
        unit.run(send)
    assert len(send.calls) == 4 and len(unit.attempts) == 4 and len(token.waits) == 3
    assert [a.n for a in unit.attempts] == [1, 2, 3, 4]
    assert all(a.outcome == "failed" for a in unit.attempts)


def test_backoff_waits_grow_within_their_jitter_ceilings():
    r, token, _ = runner(RetryPolicy(base_s=1.0, cap_s=30.0, max_attempts=4))
    send = flaky(*[fail("infrastructure", "connection") for _ in range(4)])
    with pytest.raises(EvalFailure):
        r.unit().run(send)
    assert len(token.waits) == 3
    for i, w in enumerate(token.waits):
        assert 0 <= w <= min(30.0, 2.0**i)


@pytest.mark.parametrize(
    "f",
    [
        fail("target", "exception"),
        fail("target", "blocked"),
        fail("evaluator", "truncated"),
        fail("evaluator", "refused"),
        fail("input", "oversize"),
    ],
)
def test_non_retryable_failures_are_attempted_once(f):
    r, token, _ = runner()
    send = flaky(f)
    unit = r.unit()
    with pytest.raises(EvalFailure):
        unit.run(send)
    assert len(send.calls) == 1 and token.waits == [] and len(unit.attempts) == 1


def test_a_target_timeout_is_retried_once():
    r, _, _ = runner()
    send = flaky(fail("target", "timeout"), fail("target", "timeout"), "never reached")
    with pytest.raises(EvalFailure):
        r.unit().run(send)
    assert len(send.calls) == 2


def test_an_invalid_output_is_retried_once_with_the_validation_error_fed_back():
    r, token, _ = runner()
    unit = r.unit()
    bad = fail("evaluator", "invalid_output", "score 9 is outside scale 1-5", raw={"score": 9})
    send = flaky(bad, "second")
    assert unit.run(send, feedback_retry=True) == "second"
    assert send.calls == [None, "score 9 is outside scale 1-5"]  # the retry is not a repeat
    assert token.waits == []  # no backoff: nothing to wait for
    assert [a.outcome for a in unit.attempts] == ["failed", "ok"]
    assert unit.attempts[0].raw and unit.attempts[0].error_class is FailureClass.EVALUATOR


def test_invalid_output_twice_fails_and_keeps_both_rejected_outputs():
    r, _, _ = runner()
    unit = r.unit()
    send = flaky(
        fail("evaluator", "invalid_output", "one", raw="A"),
        fail("evaluator", "invalid_output", "two", raw="B"),
        "never",
    )
    with pytest.raises(EvalFailure, match="two"):
        unit.run(send, feedback_retry=True)
    assert len(send.calls) == 2 and [a.raw for a in unit.attempts] == ['"A"', '"B"']


def test_without_feedback_retry_an_invalid_output_is_final():
    r, _, _ = runner()
    send = flaky(fail("evaluator", "invalid_output", "x"), "unused")
    with pytest.raises(EvalFailure):
        r.unit().run(send)
    assert len(send.calls) == 1


def test_validate_failures_count_as_failed_attempts_and_can_succeed_on_retry():
    r, _, _ = runner()
    unit = r.unit()

    def validate(resp):
        if resp == "bad":
            raise fail("evaluator", "invalid_output", "nope", raw=resp)
        return resp.upper()

    assert unit.run(flaky("bad", "good"), validate=validate, feedback_retry=True) == "GOOD"
    assert [a.outcome for a in unit.attempts] == ["failed", "ok"]


def test_a_transport_retry_after_a_validation_retry_uses_its_own_budget():
    r, _, _ = runner()
    send = flaky(
        fail("evaluator", "invalid_output", "x"),
        fail("infrastructure", "rate_limited"),
        "ok",
    )
    assert r.unit().run(send, feedback_retry=True) == "ok"


def test_attempt_numbers_continue_across_calls_within_a_unit():
    r, _, _ = runner()
    unit = r.unit(first_n=3)
    unit.run(flaky("a"))
    unit.run(flaky(fail("infrastructure", "connection"), "b"))
    assert [a.n for a in unit.attempts] == [3, 4, 5]


def test_error_type_is_the_wrapped_exception_class():
    r, _, _ = runner()
    unit = r.unit()
    with pytest.raises(EvalFailure):
        unit.run(flaky(fail("target", "exception", "boom", exc_type="ZeroDivisionError")))
    assert unit.attempts[0].error_type == "ZeroDivisionError"


# --- systemic failures, deadline, cancellation -----------------------------------------------


@pytest.mark.parametrize(
    "f,reason",
    [
        (fail("infrastructure", "auth"), "infrastructure.auth"),
        (fail("infrastructure", "quota_exhausted"), "infrastructure.quota_exhausted"),
        (fail("evaluator", "bad_request"), "evaluator.bad_request"),
        (fail("input", "bad_config"), "input.bad_config"),
    ],
)
def test_a_systemic_failure_is_not_retried_and_stops_the_run_once_it_repeats(f, reason):
    guard = RunGuard(systemic_threshold=1)  # (the default of 3 is tested in test_systemic_failures)
    r, _, _ = runner(guard=guard)
    send = flaky(f)
    with pytest.raises(EvalFailure):
        r.unit().run(send)
    assert len(send.calls) == 1
    assert guard.abort == ("failed", reason)


def test_the_unit_deadline_ends_retrying_with_deadline_exceeded():
    r, token, clock = runner(RetryPolicy(timeout_s=1.0, unit_deadline_s=0.5, base_s=1.0, cap_s=1.0))
    unit = r.unit()
    with pytest.raises(EvalFailure) as info:
        unit.run(flaky(*[fail("infrastructure", "rate_limited", "429", retry_after_s=10)] * 3))
    f = info.value
    assert (f.failure_class, f.kind) == (FailureClass.INFRA, "deadline_exceeded")
    assert "429" in str(f) and len(unit.attempts) == 1


def test_cancellation_during_backoff_ends_the_unit_with_its_last_real_failure():
    r, token, _ = runner(cancel_at=1)
    unit = r.unit()
    send = flaky(fail("infrastructure", "rate_limited", "429 slow"), "would succeed")
    with pytest.raises(EvalFailure, match="429 slow") as info:
        unit.run(send)
    assert info.value.kind == "rate_limited"  # not a fake "cancelled" outcome
    assert len(send.calls) == 1 and len(unit.attempts) == 1 and token.cancelled


def test_cancellation_does_not_interrupt_a_call_in_flight():
    r, token, _ = runner()
    token.cancel("stop")
    assert r.unit().run(flaky("done")) == "done"  # dispatch decides; a started call finishes


def test_the_cancel_token_wait_is_interruptible():
    token = CancelToken()
    threading.Timer(0.05, token.cancel, args=("now",)).start()
    assert token.wait(5.0) is True and token.reason == "now"
    assert CancelToken().wait(0.001) is False
    token.cancel("again")
    assert token.reason == "now"  # the first reason wins


# --- guard: breaker and budget ---------------------------------------------------------------


def infra_attempt(n=1, ok=False):
    from datetime import UTC, datetime

    from evalkit import RunAttempt

    now = datetime.now(UTC)
    if ok:
        return RunAttempt.succeeded(n, now, 1, input_tokens=5, output_tokens=5)
    return RunAttempt.failed(n, now, 1, fail("infrastructure", "provider_unavailable", "503"))


def test_the_breaker_trips_after_consecutive_infrastructure_failures_only():
    g = RunGuard(breaker_threshold=3)
    g.record(infra_attempt())
    g.record(infra_attempt())
    g.record(infra_attempt(ok=True))  # a success resets the count
    g.record(infra_attempt())
    g.record(infra_attempt())
    assert g.abort is None
    g.record(infra_attempt())
    assert g.abort == ("partial", "provider_outage")


def test_target_and_evaluator_failures_do_not_trip_the_breaker():
    from datetime import UTC, datetime

    from evalkit import RunAttempt

    g = RunGuard(breaker_threshold=2)
    for _ in range(10):
        g.record(RunAttempt.failed(1, datetime.now(UTC), 1, fail("target", "exception", "x")))
    assert g.abort is None


def test_a_systemic_abort_is_not_downgraded_by_a_later_breaker_trip():
    from datetime import UTC, datetime

    from evalkit import RunAttempt

    g = RunGuard(breaker_threshold=1, systemic_threshold=1)
    g.record(RunAttempt.failed(1, datetime.now(UTC), 1, fail("infrastructure", "auth")))
    g.record(infra_attempt())
    assert g.abort == ("failed", "infrastructure.auth")


def test_budget_stops_on_tokens_calls_and_duration():
    g = RunGuard(Budget(max_tokens=20))  # each ok attempt spends 10 tokens
    assert g.budget_stop() is None
    g.record(infra_attempt(ok=True))
    assert g.budget_stop() is None
    g.record(infra_attempt(ok=True))
    assert g.budget_stop() == "budget"  # reaching the limit counts as reached
    g2 = RunGuard(Budget(max_calls=2))
    g2.record(infra_attempt(ok=True))
    assert g2.budget_stop() is None
    g2.record(infra_attempt(ok=True))
    assert g2.budget_stop() == "budget"
    clock = Clock()
    g3 = RunGuard(Budget(max_duration_s=5), monotonic=clock)
    assert g3.budget_stop() is None
    clock.t = 5
    assert g3.budget_stop() == "deadline"
    assert RunGuard().budget_stop() is None  # no budget: never


@pytest.mark.parametrize(
    "bad", [{"max_tokens": 0}, {"max_calls": -1}, {"max_duration_s": 0}, {"max_tokens": True}]
)
def test_invalid_budgets_are_refused(bad):
    with pytest.raises(ValueError):
        Budget(**bad)
    with pytest.raises(ValueError, match="unknown budget"):
        Budget.from_mapping({"max_dollars": 1})


def test_guard_counts_are_thread_safe():
    g = RunGuard()
    threads = [
        threading.Thread(target=lambda: [g.record(infra_attempt(ok=True)) for _ in range(200)])
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert (g.usage.calls, g.usage.tokens) == (1600, 16000)


# --- rate limiter ----------------------------------------------------------------------------


def test_the_limiter_spaces_call_starts():
    clock = Clock()
    limiter = RateLimiter(2.0, monotonic=clock)
    token = SleepToken(clock)
    waits = [limiter.acquire(token) for _ in range(4)]
    assert waits == [0.0, 0.5, 0.5, 0.5]  # slots at t = 0, .5, 1, 1.5; the clock moves as we wait


def test_limiter_rejects_bad_rates_and_is_used_by_the_runner():
    for bad in (0, -1, float("inf"), True):
        with pytest.raises(ValueError):
            RateLimiter(bad)
    clock = Clock()
    token = SleepToken(clock)
    r = CallRunner(
        RetryPolicy(), token=token, limiter=RateLimiter(1.0, monotonic=clock), monotonic=clock
    )
    unit = r.unit()
    unit.run(flaky("a"))
    unit.run(flaky("b"))
    assert token.waits == [pytest.approx(1.0)]
