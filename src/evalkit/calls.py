"""The call layer (docs/TARGET-ARCHITECTURE.md §8): one place where every outbound call (a
target's or an evaluator's) gets timeout handling, retry classification, backoff, rate limiting,
a fail-fast guard, budget accounting, and an `Attempt` record.

P2.1 added to it: the response cache (`evalkit.cache`), token/cost capture with a versioned price
table (`evalkit.pricing`), per-call budget *reservation* (a call is made only if its worst case
fits what is left, so concurrent workers cannot overspend), and structured events
(`evalkit.events`). Deliberately *not* here: AIMD adaptive concurrency, per-provider limiter
tables, leases and multi-process workers.

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

import json
import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

from evalkit import events
from evalkit.cache import ResponseCache, cache_key
from evalkit.failures import EvalFailure, FailureClass, check_kind
from evalkit.llm import LLMRequest, LLMResponse
from evalkit.pricing import NO_PRICING, PriceTable
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
    """Hard stops. Token / USD / call budgets are enforced *per call*: a call is admitted only if
    its worst case fits what is left (see `RunGuard.reserve`), so they cannot be overspent by more
    than the estimate's error. The duration budget is checked when a unit is dispatched. Token and
    USD budgets are per *run*: what earlier executions of the run spent counts, so a resume cannot
    spend the budget again. `max_calls` and `max_duration_s` are per execution (P1 behaviour)."""

    max_tokens: int | None = None
    max_calls: int | None = None
    max_duration_s: float | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_calls"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_duration_s", "max_cost_usd"):
            d = getattr(self, name)
            if d is not None and (
                isinstance(d, bool) or not isinstance(d, int | float) or not 0 < d < float("inf")
            ):
                raise ValueError(f"{name} must be positive and finite")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Budget:
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown budget setting(s): {sorted(unknown)}")
        return cls(**m)


@dataclass
class SpendTotals:
    calls: int = 0  # provider (and callable) calls; a cache hit is not one
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0  # ESTIMATED, priced calls only
    unpriced_calls: int = 0  # calls whose cost is unknown (no price, or no usage reported)
    # spend a budget assumed because the provider reported no usage (the call's worst case)
    assumed_tokens: int = 0
    assumed_cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def budget_tokens(self) -> int:
        return self.tokens + self.assumed_tokens

    @property
    def budget_cost_usd(self) -> float:
        return self.cost_usd + self.assumed_cost_usd


class UnitAbandoned(Exception):
    """A call was not made and its unit must stay pending: the budget has no room for it
    (`reason="budget"`), or replay mode found no cached response (`reason="cache_miss"`)."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class CallEstimate:
    tokens: int | None  # worst case (input upper bound + output cap); None = unknowable
    cost_usd: float | None  # its worst-case price; None = unpriced


# a token is at least one byte for byte-level tokenizers, so the request's UTF-8 size bounds its
# input tokens; the constant covers message framing. An estimate, not a promise (documented).
INPUT_OVERHEAD_TOKENS = 64


def estimate_call(
    req: LLMRequest, pricing: PriceTable, provider: str | None, model: str | None
) -> CallEstimate:
    size = len(req.system.encode("utf-8", "replace")) + len(req.user.encode("utf-8", "replace"))
    if req.tool is not None:
        size += len(json.dumps(asdict(req.tool), default=str).encode("utf-8", "replace"))
    tokens_in = size + INPUT_OVERHEAD_TOKENS
    return CallEstimate(
        tokens_in + req.max_tokens, pricing.upper_bound(provider, model, tokens_in, req.max_tokens)
    )


@dataclass
class Reservation:
    tokens: int = 0
    cost_usd: float = 0.0


class RunGuard:
    """Shared, thread-safe state that can stop a run: a systemic failure that keeps repeating (auth,
    a rejected request, quota: every case would fail the same way), a tripped breaker
    (`breaker_threshold` consecutive infrastructure failures with no success between), or an
    exhausted budget. It only *decides*; the scheduler acts on it.

    A systemic failure is only *suspected* the first time: one rejected request (an over-long
    input, say) among thousands is that case's problem. The run stops when `systemic_threshold`
    failures of the same systemic kind follow each other from the same provider and model with no
    success of that provider and model in between (design 8.4). A success elsewhere (a callable
    target that keeps working) does not mask a broken judge.

    Budgets. `reserve` admits a call only if `spent + reserved-by-calls-in-flight + this call's
    worst case` is within every limit, atomically under the guard's lock; `record` then replaces
    the reservation with what the call really used (or, when the provider reported no usage, keeps
    the worst case: unknown spend is assumed, not ignored). `prior` is what earlier executions of
    the run already spent."""

    def __init__(
        self,
        budget: Budget | None = None,
        breaker_threshold: int = 10,
        monotonic: Callable[[], float] = time.monotonic,
        systemic_threshold: int = 3,
        prior: SpendTotals | None = None,
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
        self.usage = SpendTotals()  # this execution
        self.prior = prior or SpendTotals()  # earlier executions of the run
        self._reserved = Reservation()
        self._reserved_calls = 0
        self._consecutive_infra = 0
        self._systemic: dict[tuple[str | None, str | None], tuple[str, int]] = {}
        self._success_seq: dict[tuple[str | None, str | None], int] = {}
        self._seq = 0
        self._abort: tuple[str, str] | None = None  # (run status, stop_reason)
        self.abort_detail: str | None = None  # the message of the failure that tripped the abort
        self.blocked: tuple[str, str] | None = None  # (stop_reason, detail): a call was refused

    # -- budget reservation ------------------------------------------------------------------

    def _used(self) -> tuple[int, float, int]:
        u, p = self.usage, self.prior
        return (
            u.budget_tokens + p.budget_tokens,
            u.budget_cost_usd + p.budget_cost_usd,
            u.calls,  # max_calls is per execution, as it always was (docs/DECISIONS.md, DR-17)
        )

    def reserve(self, est: CallEstimate | None) -> Reservation:
        """Admit one call or raise `UnitAbandoned`. Cheap and atomic."""
        b = self.budget
        with self._lock:
            tokens, cost, calls = self._used()
            r = self._reserved
            why = None
            if b.max_calls is not None and calls + self._reserved_calls + 1 > b.max_calls:
                why = (
                    f"max_calls {b.max_calls} reached "
                    f"({calls} made, {self._reserved_calls} in flight)"
                )
            elif b.max_tokens is not None:
                need = est.tokens if est is not None and est.tokens is not None else None
                total = tokens + r.tokens
                if (need is None and total >= b.max_tokens) or (
                    need is not None and total + need > b.max_tokens
                ):
                    why = (
                        f"max_tokens {b.max_tokens}: {tokens} spent + {r.tokens} in flight"
                        f" + up to {'?' if need is None else need} for this call"
                    )
            if why is None and b.max_cost_usd is not None:
                need_c = est.cost_usd if est is not None else None
                total_c = cost + r.cost_usd
                if (need_c is None and total_c >= b.max_cost_usd) or (
                    need_c is not None and total_c + need_c > b.max_cost_usd
                ):
                    why = (
                        f"max_cost_usd {b.max_cost_usd:g}: ${cost:.6f} spent + ${r.cost_usd:.6f}"
                        f" in flight + up to ${need_c or 0:.6f} for this call"
                    )
            if why is not None:
                if self.blocked is None:
                    self.blocked = ("budget", why)
                raise UnitAbandoned("budget", why)
            res = Reservation(
                est.tokens if est is not None and est.tokens is not None else 0,
                est.cost_usd if est is not None and est.cost_usd is not None else 0.0,
            )
            r.tokens += res.tokens
            r.cost_usd += res.cost_usd
            self._reserved_calls += 1
            return res

    def release(self, res: Reservation | None) -> None:
        """Give a reservation back without recording a call (the call never happened)."""
        if res is None:
            return
        with self._lock:
            self._release(res)

    def _release(self, res: Reservation) -> None:
        self._reserved.tokens -= res.tokens
        self._reserved.cost_usd -= res.cost_usd
        self._reserved_calls -= 1

    def block(self, reason: str, detail: str) -> None:
        with self._lock:
            if self.blocked is None:
                self.blocked = (reason, detail)

    # -- recording ---------------------------------------------------------------------------

    def record(self, attempt: RunAttempt, reservation: Reservation | None = None) -> None:
        with self._lock:
            if reservation is not None:
                self._release(reservation)
            u = self.usage
            if attempt.cache_hit:
                u.cache_hits += 1
                return
            u.calls += 1
            u.input_tokens += attempt.input_tokens or 0
            u.output_tokens += attempt.output_tokens or 0
            if attempt.cost_usd is not None:
                u.cost_usd += attempt.cost_usd
            elif attempt.outcome == "ok" or attempt.input_tokens is not None:
                u.unpriced_calls += 1
            if (
                reservation is not None
                and attempt.outcome == "ok"
                and attempt.input_tokens is None
                and attempt.output_tokens is None
            ):
                u.assumed_tokens += reservation.tokens  # no usage reported: assume the worst
                u.assumed_cost_usd += reservation.cost_usd
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
        """`"budget"` / `"deadline"` / `"cache_miss"` once a limit is reached or a call was
        refused, else None."""
        b = self.budget
        with self._lock:
            if self.blocked is not None:
                return self.blocked[0]
            tokens, cost, calls = self._used()
            if b.max_tokens is not None and tokens >= b.max_tokens:
                return "budget"
            if b.max_calls is not None and calls >= b.max_calls:
                return "budget"
            if b.max_cost_usd is not None and cost >= b.max_cost_usd:
                return "budget"
        if b.max_duration_s is not None and self._monotonic() - self._start >= b.max_duration_s:
            return "deadline"
        return None

    @property
    def budget_detail(self) -> str | None:
        with self._lock:
            return None if self.blocked is None else self.blocked[1]


# ---- the call runner ------------------------------------------------------------------------


def describe_llm_response(resp: LLMResponse) -> dict[str, Any]:
    """Attempt fields a successful `LLMResponse` contributes (usage, request id)."""
    return {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "request_id": resp.request_id,
        "evidence": resp.payload if resp.payload is not None else resp.text,
    }


_USAGE_FIELDS = ("input_tokens", "output_tokens", "request_id")


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
    cache: ResponseCache | None = None
    pricing: PriceTable = NO_PRICING

    def unit(
        self,
        first_n: int = 1,
        *,
        scope: str = "",
        retry_round: int = 0,
        ctx: Mapping[str, Any] | None = None,
    ) -> UnitCalls:
        """`scope` names what the calls are for (an evaluator key, a target identity): it is part
        of every cache key. `ctx` (run_id, case_key, evaluator, correlation_id, ...) goes on every
        event the unit emits."""
        return UnitCalls(self, first_n, scope=scope, retry_round=retry_round, ctx=ctx)


class UnitCalls:
    """The calls of one unit of work (one target call, or one evaluator's calls). Collects every
    attempt in order, numbered from `first_n`, so the owner can record all of them."""

    def __init__(
        self,
        runner: CallRunner,
        first_n: int = 1,
        *,
        scope: str = "",
        retry_round: int = 0,
        ctx: Mapping[str, Any] | None = None,
    ):
        self._runner = runner
        self.attempts: list[RunAttempt] = []
        self._first_n = first_n
        self._scope = scope
        self._retry_round = retry_round
        self._ctx = dict(ctx or {})
        self._deadline_at = runner.monotonic() + runner.policy.deadline_s

    @property
    def timeout_s(self) -> float:
        return self._runner.policy.timeout_s

    def _emit(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        events.emit(event, level, **(self._ctx | fields))

    def run(
        self,
        send: Callable[[str | None], T],
        *,
        validate: Callable[[T], Any] | None = None,
        describe: Callable[[T], Mapping[str, Any]] | None = None,
        provider: str | None = None,
        model: str | None = None,
        feedback_retry: bool = False,
        request: Callable[[str | None], LLMRequest] | None = None,
        endpoint: str | None = None,
    ) -> Any:
        """Make one logical call. `send(feedback)` performs one request (feedback is None on the
        first try, and the previous validation error on a feedback retry). `validate(response)`
        turns the response into the value returned, or raises `EvalFailure`. Raises the final
        `EvalFailure` if the call cannot succeed; the attempts are on `self.attempts` either way.

        `request(feedback)` describes the LLM request `send` will make. With it the call is
        eligible for the response cache and its worst-case size is reserved against the budget
        before it is sent; without it (a callable target) neither applies beyond the totals so
        far. Raises `UnitAbandoned` -- nothing sent, unit stays pending -- when the budget has no
        room or replay finds no cached response; if attempts were already made in this call the
        budget refusal ends it as `infrastructure.budget_exceeded` instead, keeping the evidence."""
        r = self._runner
        policy = r.policy
        cache = r.cache
        transport_retries = 0
        validation_retries = 0
        feedback: str | None = None
        made = 0  # attempts of this logical call
        last: EvalFailure | None = None
        while True:
            if r.limiter is not None:
                r.limiter.acquire(r.token)
            req = request(feedback) if request is not None else None
            key = None
            if req is not None and cache is not None and cache.eligible(req):
                key = cache_key(
                    req, provider=provider, model=model, endpoint=endpoint, scope=self._scope
                )
            started_at, t0 = r.clock(), r.monotonic()
            resp: Any = None
            reservation: Reservation | None = None
            hit = False
            try:
                with cache.key_lock(key) if (cache is not None and key) else nullcontext():
                    value: Any = None
                    cached = cache.get(key) if (cache is not None and key) else None
                    if cached is not None:
                        try:
                            value = validate(cached) if validate is not None else cached
                            resp, hit = cached, True
                        except EvalFailure:  # stale under today's validation: refetch
                            cache.store.cache_delete(key)
                    if not hit:
                        if key is not None:
                            self._emit("cache.miss", logging.DEBUG, cache_mode=cache.policy.mode)
                            if cache.replay:
                                detail = "replay mode: no cached response for this request"
                                r.guard.block("cache_miss", detail)
                                raise UnitAbandoned("cache_miss", detail)
                        est = (
                            estimate_call(req, r.pricing, provider, model)
                            if req is not None
                            else None
                        )
                        reservation = r.guard.reserve(est)
                        started_at, t0 = r.clock(), r.monotonic()
                        resp = send(feedback)
                        value = validate(resp) if validate is not None else resp
                        if key is not None and cache is not None:
                            cache.put(key, provider, model, resp)
            except UnitAbandoned as ua:
                if not made:
                    self._emit("unit.abandoned", logging.WARNING, reason=ua.reason)
                    raise
                if ua.reason == "cache_miss" and last is not None:
                    raise last from None
                self._emit("budget.exhausted", logging.WARNING, reason=ua.reason)
                raise EvalFailure(
                    FailureClass.INFRA,
                    "budget_exceeded",
                    f"budget exhausted before another attempt: {ua.detail}"
                    + (f"; last error: {last}" if last is not None else ""),
                    provider=None if last is None else last.provider,
                    request_id=None if last is None else last.request_id,
                ) from ua
            except EvalFailure as f:
                usage = self._usage_of(resp, describe)
                self._record_failure(f, started_at, t0, provider, model, key, reservation, usage)
                made += 1
                last = f
                if (
                    f.failure_class is FailureClass.EVALUATOR
                    and f.kind == "invalid_output"
                    and feedback_retry
                    and validation_retries < policy.validation_retries
                ):
                    validation_retries += 1
                    feedback = str(f)  # the request changes, so the retry is not a repeat
                    self._emit(
                        "provider.retry", logging.INFO, **self._failure_fields(f), backoff_ms=0
                    )
                    continue
                if transport_retries >= policy.retries_for(f):
                    self._emit("provider.failed", logging.WARNING, **self._failure_fields(f))
                    raise
                wait = policy.backoff_s(transport_retries, f.retry_after_s, r.rng)
                if r.monotonic() + wait > self._deadline_at:
                    self._emit("provider.failed", logging.WARNING, **self._failure_fields(f))
                    raise EvalFailure(
                        FailureClass.INFRA,
                        "deadline_exceeded",
                        f"unit deadline ({policy.deadline_s:g}s) reached; last error: {f}",
                        provider=f.provider,
                        http_status=f.http_status,
                        request_id=f.request_id,
                    ) from f
                transport_retries += 1
                self._emit(
                    "provider.retry", logging.INFO, **self._failure_fields(f),
                    backoff_ms=round(wait * 1000),
                )  # fmt: skip
                if r.token.wait(wait):  # cancelled during backoff: end with the real failure
                    raise
                continue
            except BaseException:
                r.guard.release(reservation)
                raise
            self._record_success(
                resp, hit, started_at, t0, provider, model, key, reservation, describe
            )
            return value

    # -- recording ---------------------------------------------------------------------------

    @staticmethod
    def _failure_fields(f: EvalFailure) -> dict[str, Any]:
        return {
            "failure_class": f.failure_class.value,
            "failure_kind": f.kind,
            "http_status": f.http_status,
            "request_id": f.request_id,
            "error_type": f.exc_type,
        }

    @staticmethod
    def _usage_of(resp: Any, describe: Callable[[Any], Mapping[str, Any]] | None) -> dict[str, Any]:
        """Usage of a response that arrived but failed validation: it was still billed."""
        if resp is None or describe is None:
            return {}
        try:
            return {k: v for k, v in describe(resp).items() if k in _USAGE_FIELDS}
        except Exception:  # noqa: BLE001 - a describe bug must not hide the real failure
            return {}

    def _cost(
        self, provider: str | None, model: str | None, usage: Mapping[str, Any]
    ) -> dict[str, Any]:
        c = self._runner.pricing.cost(
            provider, model, usage.get("input_tokens"), usage.get("output_tokens")
        )
        return {"cost_usd": c.usd, "price_version": c.version}

    def _ms(self, t0: float) -> int:
        return max(0, round((self._runner.monotonic() - t0) * 1000))

    def _next_n(self) -> int:
        return self._first_n + len(self.attempts)

    def _record(self, attempt: RunAttempt, reservation: Reservation | None) -> None:
        self.attempts.append(attempt)
        self._runner.guard.record(attempt, reservation)
        self._emit(
            "provider.call",
            logging.DEBUG,
            attempt=attempt.n,
            outcome=attempt.outcome,
            duration_ms=attempt.duration_ms,
            provider=attempt.provider,
            model=attempt.model,
            request_id=attempt.request_id,
            input_tokens=attempt.input_tokens,
            output_tokens=attempt.output_tokens,
            cost_usd=attempt.cost_usd,
            cache_hit=attempt.cache_hit,
            retry_round=attempt.retry_round,
            failure_class=None if attempt.error_class is None else attempt.error_class.value,
            failure_kind=attempt.error_kind,
            http_status=attempt.http_status,
        )

    def _record_success(
        self, resp, hit, started_at, t0, provider, model, key, reservation, describe
    ) -> None:
        if hit:
            self._emit("cache.hit", logging.DEBUG, cache_mode=self._runner.cache.policy.mode)
            self._record(
                RunAttempt.succeeded(
                    self._next_n(), started_at, self._ms(t0), provider=provider, model=model,
                    cache_hit=True, cache_key=key, cost_usd=0.0, retry_round=self._retry_round,
                ),
                None,
            )  # fmt: skip
            return
        fields = dict(describe(resp)) if describe is not None else {}
        evidence = fields.pop("evidence", None)  # hashed, never kept, when successful
        self._record(
            RunAttempt.succeeded(
                self._next_n(), started_at, self._ms(t0), evidence=evidence, provider=provider,
                model=model, cache_key=key, retry_round=self._retry_round,
                **fields, **self._cost(provider, model, fields),
            ),
            reservation,
        )  # fmt: skip

    def _record_failure(self, f, started_at, t0, provider, model, key, reservation, usage) -> None:
        self._record(
            RunAttempt.failed(
                self._next_n(), started_at, self._ms(t0), f, evidence=f.raw,
                provider=provider or f.provider, model=model, cache_key=key,
                retry_round=self._retry_round, **usage, **self._cost(provider, model, usage),
            ),
            reservation,
        )  # fmt: skip
