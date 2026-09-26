"""The call layer (docs/TARGET-ARCHITECTURE.md §8): one place where every outbound call (a
target's or an evaluator's) gets timeout handling, retry classification, backoff, rate limiting,
a fail-fast guard, budget accounting, and an `Attempt` record.

This is the P1 skeleton of the design's `CallRunner`. Deliberately *not* here (P2): AIMD adaptive
concurrency, per-provider limiter tables, the response cache, USD pricing/reservation budgets,
leases and multi-process workers.

Semantics worth knowing:

* Retrying is decided by `(failure class, kind)`: transport-type infrastructure failures retry with
  full-jitter exponential backoff (honouring `Retry-After`, capped); `evaluator.invalid_output`
  retries once *with the validation error fed back* (an identical retry at temperature 0 just
  reproduces the failure); systemic failures and everything else never retry.
* Every attempt -- success or failure, first or retry -- is recorded, with the failed call's
  evidence (P0 rules).
* Cancellation is cooperative and checked between units and during backoff sleeps. A call already
  in flight is never abandoned mid-request; a unit whose backoff is interrupted ends with the last
  *real* failure, so every recorded result is honest evidence (nothing is "cancelled" into a fake
  outcome).
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

from evalkit.failures import EvalFailure, FailureClass, check_kind
from evalkit.llm import LLMResponse
from evalkit.runs import RunAttempt

T = TypeVar("T")


class CancelToken:
    """Cooperative cancellation. `wait` is an interruptible sleep."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`; True if cancelled meanwhile."""
        return self._event.wait(max(0.0, seconds))


# ---- policy ---------------------------------------------------------------------------------

# how many times each retryable kind may be retried (design 7.4); anything absent is not retried
_DEFAULT_RETRIES: dict[tuple[str, str], int] = {
    ("infrastructure", "rate_limited"): 3,
    ("infrastructure", "provider_unavailable"): 3,
    ("infrastructure", "connection"): 3,
    ("infrastructure", "timeout"): 2,
    ("target", "timeout"): 1,
}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4  # transport attempts per call, including the first
    base_s: float = 1.0
    cap_s: float = 30.0
    timeout_s: float = 60.0  # per request (passed to clients / callable wrapper)
    unit_deadline_s: float | None = None  # None: 3 x timeout_s, retries and backoff included
    validation_retries: int = 1  # retries of an unusable (invalid) judge output, with feedback

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an integer >= 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be an integer >= 1")
        for name in ("base_s", "cap_s", "timeout_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v < float("inf"):
                raise ValueError(f"{name} must be a positive finite number")
        if self.cap_s < self.base_s:
            raise ValueError("cap_s must be >= base_s")
        if self.unit_deadline_s is not None and not 0 < self.unit_deadline_s < float("inf"):
            raise ValueError("unit_deadline_s must be positive and finite")
        v = self.validation_retries
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 3:
            raise ValueError("validation_retries must be an integer in 0..3")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> RetryPolicy:
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown retry setting(s): {sorted(unknown)}")
        return cls(**m)

    @property
    def deadline_s(self) -> float:
        return self.unit_deadline_s if self.unit_deadline_s is not None else 3 * self.timeout_s

    def retries_for(self, f: EvalFailure) -> int:
        """Retries allowed after `f`, from the kind's cap; 0 when the failure says not to."""
        if not f.retryable or f.systemic:
            return 0
        cap = _DEFAULT_RETRIES.get((f.failure_class.value, f.kind), 0)
        return min(cap, self.max_attempts - 1)

    def backoff_s(self, retry_no: int, retry_after: float | None, rng: random.Random) -> float:
        """Full jitter: uniform(0, min(cap, base * 2**retry_no)); `Retry-After` is honoured (never
        undercut) but capped."""
        ceiling = min(self.cap_s, self.base_s * 2**retry_no)
        wait = rng.uniform(0, ceiling)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.cap_s))
        return wait


class RateLimiter:
    """Minimum spacing between call starts, shared by all workers: `max_per_s` calls per second."""

    def __init__(
        self,
        max_per_s: float,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if isinstance(max_per_s, bool) or not 0 < max_per_s < float("inf"):
            raise ValueError("max_per_s must be a positive finite number")
        self._interval = 1.0 / max_per_s
        self._monotonic = monotonic
        self._next = 0.0
        self._lock = threading.Lock()

    def acquire(self, token: CancelToken) -> float:
        """Wait for a slot; returns the seconds waited (0 if none)."""
        with self._lock:
            now = self._monotonic()
            slot = max(now, self._next)
            self._next = slot + self._interval
        wait = slot - now
        if wait > 0:
            token.wait(wait)
        return max(0.0, wait)


@dataclass(frozen=True)
class Budget:
    """Hard stops that need no price table. Checked before each unit is dispatched, so in-flight
    work can overshoot: the guarantee is `spend <= budget + window x max_unit_spend`."""

    max_tokens: int | None = None
    max_calls: int | None = None
    max_duration_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_calls"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} must be a positive integer")
        d = self.max_duration_s
        if d is not None and (isinstance(d, bool) or not 0 < d < float("inf")):
            raise ValueError("max_duration_s must be positive and finite")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Budget:
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown budget setting(s): {sorted(unknown)}")
        return cls(**m)


@dataclass
class SpendTotals:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RunGuard:
    """Shared, thread-safe state that can stop a run: a systemic failure that keeps repeating (auth,
    a rejected request, quota: every case would fail the same way), a tripped breaker
    (`breaker_threshold` consecutive infrastructure failures with no success between), or an
    exhausted budget. It only *decides*; the scheduler acts on it.

    A systemic failure is only *suspected* the first time: one rejected request (an over-long
    input, say) among thousands is that case's problem. The run stops when `systemic_threshold`
    failures of the same systemic kind follow each other from the same provider and model with no
    success of that provider and model in between (design 8.4). A success elsewhere (a callable
    target that keeps working) does not mask a broken judge."""

    def __init__(
        self,
        budget: Budget | None = None,
        breaker_threshold: int = 10,
        monotonic: Callable[[], float] = time.monotonic,
        systemic_threshold: int = 3,
    ):
        for name, value in (
            ("breaker_threshold", breaker_threshold),
            ("systemic_threshold", systemic_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        self.budget = budget or Budget()
        self.breaker_threshold = breaker_threshold
        self.systemic_threshold = systemic_threshold
        self._monotonic = monotonic
        self._start = monotonic()
        self._lock = threading.Lock()
        self.usage = SpendTotals()
        self._consecutive_infra = 0
        self._systemic: dict[tuple[str | None, str | None], tuple[str, int]] = {}
        self._success_seq: dict[tuple[str | None, str | None], int] = {}
        self._seq = 0
        self._abort: tuple[str, str] | None = None  # (run status, stop_reason)
        self.abort_detail: str | None = None  # the message of the failure that tripped the abort

    def record(self, attempt: RunAttempt) -> None:
        with self._lock:
            u = self.usage
            u.calls += 1
            u.input_tokens += attempt.input_tokens or 0
            u.output_tokens += attempt.output_tokens or 0
            key = (attempt.provider, attempt.model)
            if attempt.outcome == "ok":
                self._consecutive_infra = 0
                self._systemic.pop(key, None)
                self._seq += 1
                self._success_seq[key] = self._seq
                return
            u.failed_calls += 1
            if attempt.error_class is FailureClass.INFRA:
                self._consecutive_infra += 1
                if self._consecutive_infra >= self.breaker_threshold and self._abort is None:
                    self._abort = ("partial", "provider_outage")
            else:
                self._consecutive_infra = 0
            if attempt.error_class is not None and attempt.error_kind is not None:
                if check_kind(attempt.error_class, attempt.error_kind).systemic:
                    kind = f"{attempt.error_class.value}.{attempt.error_kind}"
                    seen_kind, n = self._systemic.get(key, (kind, 0))
                    n = n + 1 if seen_kind == kind else 1
                    self._systemic[key] = (kind, n)
                    if n >= self.systemic_threshold and (
                        self._abort is None or self._abort[0] != "failed"
                    ):
                        self._abort = ("failed", kind)
                        self.abort_detail = attempt.error

    def success_seq(self, provider: str | None, model: str | None) -> int:
        """A counter that grows with every success of this provider and model: a failure recorded
        at count `k` is *proven* case-specific once the count exceeds `k`."""
        with self._lock:
            return self._success_seq.get((provider, model), 0)

    @property
    def abort(self) -> tuple[str, str] | None:
        with self._lock:
            return self._abort

    def budget_stop(self) -> str | None:
        """`"budget"` / `"deadline"` once a limit is reached, else None."""
        b = self.budget
        with self._lock:
            if b.max_tokens is not None and self.usage.tokens >= b.max_tokens:
                return "budget"
            if b.max_calls is not None and self.usage.calls >= b.max_calls:
                return "budget"
        if b.max_duration_s is not None and self._monotonic() - self._start >= b.max_duration_s:
            return "deadline"
        return None


# ---- the call runner ------------------------------------------------------------------------


def describe_llm_response(resp: LLMResponse) -> dict[str, Any]:
    """Attempt fields a successful `LLMResponse` contributes (usage, request id)."""
    return {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "request_id": resp.request_id,
        "evidence": resp.payload if resp.payload is not None else resp.text,
    }


@dataclass
class CallRunner:
    """Policy shared by every unit of a run. `unit()` gives one unit's view."""

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    token: CancelToken = field(default_factory=CancelToken)
    guard: RunGuard = field(default_factory=RunGuard)
    limiter: RateLimiter | None = None
    rng: random.Random = field(default_factory=random.Random)
    monotonic: Callable[[], float] = time.monotonic
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)  # noqa: E731

    def unit(self, first_n: int = 1) -> UnitCalls:
        return UnitCalls(self, first_n)


class UnitCalls:
    """The calls of one unit of work (one target call, or one evaluator's calls). Collects every
    attempt in order, numbered from `first_n`, so the owner can record all of them."""

    def __init__(self, runner: CallRunner, first_n: int = 1):
        self._runner = runner
        self.attempts: list[RunAttempt] = []
        self._first_n = first_n
        self._deadline_at = runner.monotonic() + runner.policy.deadline_s

    @property
    def timeout_s(self) -> float:
        return self._runner.policy.timeout_s

    def run(
        self,
        send: Callable[[str | None], T],
        *,
        validate: Callable[[T], Any] | None = None,
        describe: Callable[[T], Mapping[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        feedback_retry: bool = False,
    ) -> Any:
        """Make one logical call. `send(feedback)` performs one request (feedback is None on the
        first try, and the previous validation error on a feedback retry). `validate(response)`
        turns the response into the value returned, or raises `EvalFailure`. Raises the final
        `EvalFailure` if the call cannot succeed; the attempts are on `self.attempts` either way."""
        r = self._runner
        policy = r.policy
        transport_retries = 0
        validation_retries = 0
        feedback: str | None = None
        while True:
            if r.limiter is not None:
                r.limiter.acquire(r.token)
            started_at, t0 = r.clock(), r.monotonic()
            try:
                resp = send(feedback)
                value = validate(resp) if validate is not None else resp
            except EvalFailure as f:
                self._record_failure(f, started_at, t0, provider, model)
                if (
                    f.failure_class is FailureClass.EVALUATOR
                    and f.kind == "invalid_output"
                    and feedback_retry
                    and validation_retries < policy.validation_retries
                ):
                    validation_retries += 1
                    feedback = str(f)  # the request changes, so the retry is not a repeat
                    continue
                if transport_retries >= policy.retries_for(f):
                    raise
                wait = policy.backoff_s(transport_retries, f.retry_after_s, r.rng)
                if r.monotonic() + wait > self._deadline_at:
                    raise EvalFailure(
                        FailureClass.INFRA,
                        "deadline_exceeded",
                        f"unit deadline ({policy.deadline_s:g}s) reached; last error: {f}",
                        provider=f.provider,
                        http_status=f.http_status,
                        request_id=f.request_id,
                    ) from f
                transport_retries += 1
                if r.token.wait(wait):  # cancelled during backoff: end with the real failure
                    raise
                continue
            attempt_fields = dict(describe(resp)) if describe is not None else {}
            evidence = attempt_fields.pop("evidence", None)  # hashed, never kept, when successful
            self._record(
                RunAttempt.succeeded(
                    self._next_n(),
                    started_at,
                    self._ms(t0),
                    evidence=evidence,
                    provider=provider,
                    model=model,
                    **attempt_fields,
                )
            )
            return value

    def _ms(self, t0: float) -> int:
        return max(0, round((self._runner.monotonic() - t0) * 1000))

    def _next_n(self) -> int:
        return self._first_n + len(self.attempts)

    def _record(self, attempt: RunAttempt) -> None:
        self.attempts.append(attempt)
        self._runner.guard.record(attempt)

    def _record_failure(
        self,
        f: EvalFailure,
        started_at: datetime,
        t0: float,
        provider: str | None,
        model: str | None,
    ) -> None:
        self._record(
            RunAttempt.failed(
                self._next_n(),
                started_at,
                self._ms(t0),
                f,
                evidence=f.raw,
                provider=provider or f.provider,
                model=model,
            )
        )
