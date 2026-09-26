"""Failure taxonomy (docs/TARGET-ARCHITECTURE.md §7): who is responsible when something failed.

    input           the case or configuration is unusable, regardless of any model
    target          the system under test failed to produce an output
    evaluator       a healthy provider returned something the evaluator cannot use
    infrastructure  we failed to get a complete response at all (throttle, 5xx, auth, storage, ...)

`(failure_class, kind)` is data, not a string: it is stored on the result, so "the AI system's
problems" versus "my rubric misbehaved" versus "rerun after the outage" is a query. Which classes
may appear where is fixed by the store: a case result (target stage) is input | target |
infrastructure; an evaluator result is input | evaluator | infrastructure. A target failure is
therefore never an evaluator result and never a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalkit.evidence import clip
from evalkit.limits import MAX_ERROR_CHARS
from evalkit.redact import scrub


class FailureClass(StrEnum):
    INPUT = "input"
    TARGET = "target"
    EVALUATOR = "evaluator"
    INFRA = "infrastructure"


@dataclass(frozen=True)
class KindInfo:
    retryable: bool  # worth trying again (the default; a recorded failure may override it)
    systemic: bool = False  # will fail every case: the run should stop rather than continue


_C = FailureClass
# §7.2. `retryable` says a retry is *allowed* (with its own small cap, decided by the retry policy).
KINDS: dict[FailureClass, dict[str, KindInfo]] = {
    _C.INPUT: {
        **{
            k: KindInfo(False)
            for k in ("invalid_case", "oversize", "missing_field", "duplicate_key")
        },
        # a configuration the provider rejects (unknown model id, bad parameters) fails every case
        "bad_config": KindInfo(False, systemic=True),
    },
    _C.TARGET: {
        "exception": KindInfo(False),
        "timeout": KindInfo(True),
        "contract_violation": KindInfo(False),
        "blocked": KindInfo(False),
        "empty_output": KindInfo(False),
    },
    _C.EVALUATOR: {
        "invalid_output": KindInfo(True),
        "truncated": KindInfo(False),
        "refused": KindInfo(False),
        "bad_request": KindInfo(False, systemic=True),
        "internal_error": KindInfo(False),
    },
    _C.INFRA: {
        "rate_limited": KindInfo(True),
        "provider_unavailable": KindInfo(True),
        "timeout": KindInfo(True),
        "connection": KindInfo(True),
        "auth": KindInfo(False, systemic=True),
        "quota_exhausted": KindInfo(False, systemic=True),
        "storage": KindInfo(False),
        "budget_exceeded": KindInfo(False),
        "deadline_exceeded": KindInfo(False),
        "cancelled": KindInfo(False),
        # a bug in EvalKit itself while handling a unit: not the target's, the evaluator's or the
        # provider's fault, and never silently turned into a score
        "internal_error": KindInfo(False),
    },
}

# Where each class may be recorded. Enforced again by CHECK constraints in the database.
CASE_RESULT_CLASSES = frozenset({_C.INPUT, _C.TARGET, _C.INFRA})
EVALUATOR_RESULT_CLASSES = frozenset({_C.INPUT, _C.EVALUATOR, _C.INFRA})


# `--retry-failed` (resume with a second try for failed units). A failure is eligible when it says
# it is retryable *and* another try can plausibly differ: an `evaluator.invalid_output` is retried
# once in the unit with the validation error fed back, and at temperature 0 a later identical try
# just reproduces it, so it is permanent. A deadline or a budget that cut a unit short is not the
# unit's fault: it is retried though its kind is not "retryable" in the in-unit sense. Systemic
# kinds (auth, bad config, quota) fail every case the same way and are fixed, not retried.
RETRY_NEVER = frozenset({(_C.EVALUATOR, "invalid_output")})
RETRY_ALSO = frozenset({(_C.INFRA, "deadline_exceeded"), (_C.INFRA, "budget_exceeded")})


def retry_eligible(failure_class: FailureClass, kind: str, retryable: bool) -> bool:
    key = (failure_class, kind)
    if key in RETRY_ALSO:
        return True
    if key in RETRY_NEVER or check_kind(failure_class, kind).systemic:
        return False
    return bool(retryable)


def check_kind(failure_class: FailureClass, kind: str) -> KindInfo:
    try:
        return KINDS[failure_class][kind]
    except KeyError:
        known = ", ".join(sorted(KINDS[failure_class]))
        raise ValueError(
            f"unknown failure kind {kind!r} for class {failure_class.value!r} (known: {known})"
        ) from None


class Failure(BaseModel):
    """A recorded failure: class and kind, whether a retry is allowed, and a scrubbed message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_class: FailureClass
    kind: str
    message: str = Field(min_length=1)
    retryable: bool = None  # type: ignore[assignment]  # omitted: the default for this kind

    @field_validator("message", mode="before")
    @classmethod
    def _scrubbed(cls, v: Any) -> Any:
        # diagnostics, not data: make provider text encodable (P0 behaviour), then scrub and clip
        return clip(scrub(v), MAX_ERROR_CHARS) if isinstance(v, str) else v

    @model_validator(mode="before")
    @classmethod
    def _default_retryable(cls, data: object) -> object:
        """Resolve an omitted `retryable` to the kind's default (unknown kinds fail just below)."""
        if isinstance(data, dict) and data.get("retryable") is None:
            try:
                info = check_kind(FailureClass(data.get("failure_class")), data.get("kind", ""))
            except (ValueError, TypeError):
                return data
            return data | {"retryable": info.retryable}
        return data

    @model_validator(mode="after")
    def _kind(self) -> Failure:
        check_kind(self.failure_class, self.kind)
        return self

    @property
    def systemic(self) -> bool:
        return check_kind(self.failure_class, self.kind).systemic


class EvalFailure(Exception):
    """Raised by targets and evaluators to report a classified failure.

    A plain Exception subclass (frozen dataclasses break raise/traceback handling). The message is
    scrubbed of secrets. `failure` is the record that gets stored.
    """

    def __init__(
        self,
        failure_class: FailureClass,
        kind: str,
        message: str,
        *,
        retryable: bool | None = None,
        provider: str | None = None,
        http_status: int | None = None,
        request_id: str | None = None,
        retry_after_s: float | None = None,
        subkind: str | None = None,
        raw: Any = None,
        exc_type: str | None = None,
    ):
        extra = {} if retryable is None else {"retryable": retryable}
        self.failure = Failure(failure_class=failure_class, kind=kind, message=message, **extra)
        super().__init__(self.failure.message)
        self.provider = provider
        self.http_status = http_status
        self.request_id = request_id
        self.retry_after_s = retry_after_s
        self.subkind = subkind  # provider-detected sub-case (truncated, no_tool_call, ...)
        self.raw = raw  # rejected / partial provider output, kept as attempt evidence
        self.exc_type = exc_type  # class name of the exception this wraps (target code raised it)

    @property
    def failure_class(self) -> FailureClass:
        return self.failure.failure_class

    @property
    def kind(self) -> str:
        return self.failure.kind

    @property
    def retryable(self) -> bool:
        return self.failure.retryable

    @property
    def systemic(self) -> bool:
        return self.failure.systemic
