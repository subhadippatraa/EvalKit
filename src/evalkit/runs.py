"""Runs and their results (docs/TARGET-ARCHITECTURE.md §3-§4, §7): the persistent domain model.

    Run              one frozen RunConfig over exactly one sealed DatasetVersion
     └ CaseResult    the target stage of one case: an output, or a classified failure
        ├ Attempt    every external call made for it (target)
        └ EvaluatorResult   one evaluator's outcome for that case: metrics and a verdict, or a
           │                classified failure, or "not applicable" / "skipped"
           └ Attempt        every external call made for it (evaluator)

This module is data and rules only: the models, their coherence checks, hashing, and the
`RunService` facade. It does not *execute* anything (no targets, no evaluators, no scheduler);
whatever executes runs later records what happened through `RunService`. Persistence is in
`run_store.py`; the database enforces the same rules again (CHECKs and triggers, migration 4).

Rules that matter:

* A run's config, dataset version and identity are frozen at creation. `identity_hash` names what
  is *measured* (dataset content, target, evaluators, scoring version); `exec_hash` names how it
  was *executed* (policy: concurrency, retries, ...), and is excluded from comparison identity.
* Results are write-once evidence. A case has at most one result per run, and only if it belongs to
  the run's dataset version.
* A target failure is a case-result failure. It is never an evaluator result with a score:
  evaluators of a failed case result are `skipped`, carrying no metric and no verdict.
* Evaluator and infrastructure failures are recorded as such, on the evaluator result, and stay
  distinguishable from target failures (`FailureClass`).
* Every attempt is kept, failed ones with their evidence (P0 guarantees, `evalkit.evidence`).
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from evalkit.datasets import DatasetService, _check_json
from evalkit.errors import RunError
from evalkit.evidence import capture_evidence, clip
from evalkit.failures import (
    CASE_RESULT_CLASSES,
    EVALUATOR_RESULT_CLASSES,
    EvalFailure,
    Failure,
    FailureClass,
    check_kind,
)
from evalkit.hashing import stable_hash
from evalkit.limits import (
    MAX_ATTEMPTS,
    MAX_CONFIG_BYTES,
    MAX_ERROR_CHARS,
    MAX_EVIDENCE_BYTES,
    MAX_METRICS,
    MAX_RUN_EVALUATORS,
    MAX_TAG_CHARS,
    Limits,
)
from evalkit.models import format_validation_error
from evalkit.redact import scrub

# Bumped whenever normalization, aggregation or verdict logic changes what a score means, so that
# results produced under different logic are never silently compared (design §3.3).
SCORING_VERSION = 1

RUN_IDENTITY_DOMAIN = "evalkit-run-identity-v1"
RUN_EXEC_DOMAIN = "evalkit-run-exec-v1"
EVALUATOR_DOMAIN = "evalkit-evaluator-v1"

RunStatus = Literal["created", "running", "succeeded", "partial", "failed", "cancelled"]
TERMINAL_STATUSES = frozenset({"succeeded", "partial", "failed", "cancelled"})
# §4.2. partial / cancelled / failed keep every result and can be resumed.
TRANSITIONS: dict[str, frozenset[str]] = {
    "created": frozenset({"running"}),
    "running": frozenset({"succeeded", "partial", "cancelled", "failed"}),
    "partial": frozenset({"running"}),
    "cancelled": frozenset({"running"}),
    "failed": frozenset({"running"}),
    "succeeded": frozenset(),
}
_STOP_REASON_STATUSES = frozenset({"partial", "failed", "cancelled"})  # these say why they stopped

_KIND_RE = r"^[a-z][a-z0-9_]{0,31}$"
_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"  # no ':' -- it separates the parts of a key
_METRIC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}$")  # score, recall@5, criterion.x
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_LABEL = 128


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


def _json_object(value: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Plain, finite, encodable JSON, in the exact form that will be stored and hashed."""
    _check_json(dict(value), where)
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where} is not JSON-serializable: {e}") from e


def _text(value: str | None, where: str) -> str | None:
    if value is not None:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError(f"{where} is not valid Unicode text (unpaired surrogate)") from e
    return value


# -- configuration -----------------------------------------------------------------------------


class TargetSpec(BaseModel):
    """Declares what is evaluated. It is a *record*: nothing here calls a target.

    `identity` is whatever makes two targets of the same kind different (model id, prompt hash,
    a declared pipeline version); it is hashed into the run's identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["precomputed", "reuse", "callable", "model"] = "precomputed"
    identity: dict[str, Any] = Field(default_factory=dict)

    @field_validator("identity")
    @classmethod
    def _plain(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_object(v, "target identity")


class EvaluatorSpec(BaseModel):
    """Declares one evaluator of a run. `key` covers everything that can change a score, so
    results with different keys are never silently compared (§3.3): `kind:name:<12 hex>`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: str = Field(pattern=_KIND_RE)  # llm_judge, exact_match, retrieval, ...
    name: str = Field(pattern=_NAME_RE)  # distinguishes two evaluators of one kind
    version: int = Field(default=1, ge=1)  # of the evaluator's own scoring logic
    params: dict[str, Any] = Field(default_factory=dict)  # rubric, judge model, k, pattern, ...

    @field_validator("params")
    @classmethod
    def _plain(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_object(v, "evaluator params")

    @property
    def key(self) -> str:
        digest = stable_hash(
            EVALUATOR_DOMAIN,
            {"kind": self.kind, "name": self.name, "version": self.version, "params": self.params},
        )
        return f"{self.kind}:{self.name}:{digest[:12]}"


class RunConfig(BaseModel):
    """The frozen specification of a run: *what* is measured and *how* it is executed.

    Frozen means the object cannot be reassigned, and once a run is created its persisted JSON
    cannot be changed (the database refuses). `policy` is deliberately opaque here: concurrency,
    retry, budget and gate settings belong to the execution engine, which will type them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    target: TargetSpec = Field(default_factory=TargetSpec)
    evaluators: tuple[EvaluatorSpec, ...] = ()
    policy: dict[str, Any] = Field(default_factory=dict)
    scoring_version: int = Field(default=SCORING_VERSION, ge=1)

    @field_validator("evaluators", mode="before")
    @classmethod
    def _tuple(cls, v: Any) -> Any:
        return tuple(v) if isinstance(v, list) else v  # accept a list; store an immutable tuple

    @field_validator("policy")
    @classmethod
    def _plain(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_object(v, "policy")

    @model_validator(mode="after")
    def _evaluators_distinct(self) -> RunConfig:
        if len(self.evaluators) > MAX_RUN_EVALUATORS:
            raise ValueError(f"a run has at most {MAX_RUN_EVALUATORS} evaluators")
        keys = [e.key for e in self.evaluators]
        if len(set(keys)) != len(keys):
            raise ValueError("two evaluators have the same key (identical kind, name and config)")
        return self

    @property
    def evaluator_keys(self) -> list[str]:
        return sorted(e.key for e in self.evaluators)

    @property
    def exec_hash(self) -> str:
        """Hash of the execution policy only: speed and cost, never scores."""
        return stable_hash(RUN_EXEC_DOMAIN, self.policy)

    def identity_hash(self, dataset_content_hash: str) -> str:
        """Hash of everything that affects scores. Content-addressed: it does not depend on the
        dataset's name or version number, the run's id, or the evaluators' order, so a re-run of
        the same measurement has the same identity (and a different execution policy does not
        change it)."""
        return stable_hash(
            RUN_IDENTITY_DOMAIN,
            {
                "dataset": dataset_content_hash,
                "target": {"kind": self.target.kind, "identity": self.target.identity},
                "evaluators": self.evaluator_keys,
                "scoring_version": self.scoring_version,
            },
        )


# -- attempts ----------------------------------------------------------------------------------


class RunAttempt(BaseModel):
    """One external call (a target call or an evaluator call), successful or not.

    Every call is recorded, so a result that succeeded on its retry is distinguishable from one that
    succeeded first time. Failed calls keep the rejected / partial output as evidence, truncated to
    MAX_EVIDENCE_BYTES with the hash of the full text; successful calls keep only the hash. Build
    them with `RunAttempt.succeeded` / `RunAttempt.failed` to get that handling for free; a directly
    constructed attempt is held to the same rules.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(default_factory=_new_id)
    n: int = Field(ge=1)  # 1-based, contiguous per owner
    outcome: Literal["ok", "failed"]
    started_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    provider: str | None = Field(default=None, max_length=_MAX_LABEL)
    model: str | None = Field(default=None, max_length=_MAX_LABEL)
    error_class: FailureClass | None = Field(default=None, strict=False)  # "target" is fine
    error_kind: str | None = None
    error_type: str | None = Field(default=None, max_length=_MAX_LABEL)  # exception class name
    error: str | None = None  # scrubbed and clipped
    http_status: int | None = Field(default=None, ge=100, le=599)
    request_id: str | None = Field(default=None, max_length=256)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    raw: str | None = None  # rejected / partial output as text (failed attempts only)
    raw_sha256: str | None = None  # of the full, untruncated text
    raw_truncated: bool = False

    @field_validator("error", mode="before")
    @classmethod
    def _scrubbed(cls, v: Any) -> Any:
        return clip(scrub(v), MAX_ERROR_CHARS) if isinstance(v, str) else v

    @model_validator(mode="after")
    def _coherent(self) -> RunAttempt:
        _text(self.raw, "raw")
        if self.raw_sha256 is not None and not _SHA_RE.match(self.raw_sha256):
            raise ValueError("raw_sha256 must be 64 lowercase hex characters")
        if self.raw is not None:
            if self.raw_sha256 is None:
                raise ValueError("kept evidence needs the hash of the full text")
            if len(self.raw.encode("utf-8")) > MAX_EVIDENCE_BYTES:
                raise ValueError(
                    f"evidence is longer than {MAX_EVIDENCE_BYTES} bytes; use RunAttempt.failed()"
                )
        if self.raw_truncated and self.raw is None:
            raise ValueError("raw_truncated is set but no evidence is kept")
        if self.outcome == "ok":
            if self.error_class or self.error_kind or self.error or self.error_type or self.raw:
                raise ValueError("a successful attempt carries no error and no kept evidence")
        else:
            if self.error_class is None or self.error_kind is None:
                raise ValueError("a failed attempt needs error_class and error_kind")
            check_kind(self.error_class, self.error_kind)
        return self

    @classmethod
    def succeeded(
        cls, n: int, started_at: datetime, duration_ms: int, *, evidence: Any = None, **fields: Any
    ) -> RunAttempt:
        _, sha, _ = capture_evidence(evidence, keep_text=False)
        return cls(
            n=n,
            outcome="ok",
            started_at=started_at,
            duration_ms=duration_ms,
            raw_sha256=sha,
            **fields,
        )

    @classmethod
    def failed(
        cls,
        n: int,
        started_at: datetime,
        duration_ms: int,
        failure: Failure | EvalFailure,
        *,
        evidence: Any = None,
        **fields: Any,
    ) -> RunAttempt:
        """A failed call. `failure` may be the `EvalFailure` that was raised; its provider,
        status and request id are picked up unless given explicitly."""
        if isinstance(failure, EvalFailure):
            fields.setdefault("provider", failure.provider)
            fields.setdefault("http_status", failure.http_status)
            fields.setdefault("request_id", failure.request_id)
            fields.setdefault("error_type", failure.exc_type or type(failure).__name__)
            failure = failure.failure
        text, sha, truncated = capture_evidence(evidence, keep_text=True)
        return cls(
            n=n,
            outcome="failed",
            started_at=started_at,
            duration_ms=duration_ms,
            error_class=failure.failure_class,
            error_kind=failure.kind,
            error=failure.message,
            raw=text,
            raw_sha256=sha,
            raw_truncated=truncated,
            **fields,
        )


# -- case results ------------------------------------------------------------------------------


def _check_failure_classes(failure: Failure | None, allowed: frozenset[FailureClass], where: str):
    if failure is not None and failure.failure_class not in allowed:
        names = ", ".join(sorted(c.value for c in allowed))
        raise ValueError(
            f"a {where} failure is one of: {names}; got {failure.failure_class.value!r}"
        )


def _check_times(started_at: datetime | None, finished_at: datetime | None) -> None:
    if started_at is not None and finished_at is not None and finished_at < started_at:
        raise ValueError("finished_at is before started_at")


class CaseOutcome(BaseModel):
    """What happened at the target stage for one case: an output, or a failure -- never both, and
    never neither (a case still to be processed is a `pending` result, see `RunService.plan`).

    `output` may be the empty string (a target really can return nothing; whether that is a failure
    is policy, `target.empty_output`). It is stored losslessly.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    output: str | None = None
    retrieved: list[str] | None = None  # doc ids the target's retriever returned, in rank order
    failure: Failure | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime = Field(default_factory=_now)
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> CaseOutcome:
        _text(self.output, "output")
        if (self.output is None) == (self.failure is None):
            raise ValueError("a case outcome has an output or a failure, exactly one of them")
        if self.failure is not None and self.retrieved is not None:
            raise ValueError("a failed case outcome carries no retrieved documents")
        _check_failure_classes(self.failure, CASE_RESULT_CLASSES, "case")
        _check_times(self.started_at, self.finished_at)
        return self

    @classmethod
    def complete(cls, output: str, **fields: Any) -> CaseOutcome:
        return cls(output=output, **fields)

    @classmethod
    def fail(cls, failure: Failure | EvalFailure, **fields: Any) -> CaseOutcome:
        failure = failure.failure if isinstance(failure, EvalFailure) else failure
        return cls(failure=failure, **fields)


class CaseResult(BaseModel):
    """The stored target-stage result of one case in one run."""

    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    case_id: str
    case_key: str
    status: Literal["pending", "complete", "failed"]
    output: str | None = None
    retrieved: list[str] | None = None
    failure: Failure | None = None
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    attempts: list[RunAttempt] = Field(default_factory=list)  # target calls; empty if not loaded

    @model_validator(mode="after")
    def _coherent(self) -> CaseResult:
        pending = self.status == "pending"
        if pending:
            if any(
                v is not None for v in (self.output, self.retrieved, self.failure, self.finished_at)
            ):
                raise ValueError("a pending case result has no outcome yet")
            return self
        if self.finished_at is None:
            raise ValueError("a terminal case result has finished_at")
        if (self.status == "complete") != (self.output is not None):
            raise ValueError("a complete case result has an output, and a failed one has none")
        if (self.status == "failed") != (self.failure is not None):
            raise ValueError("a failed case result has a failure, and a complete one has none")
        if self.failure is not None and self.retrieved is not None:
            raise ValueError("a failed case result carries no retrieved documents")
        _check_failure_classes(self.failure, CASE_RESULT_CLASSES, "case")
        _check_times(self.started_at, self.finished_at)
        return self


# -- evaluator results -------------------------------------------------------------------------

MetricValue = StrictFloat | StrictInt  # a bool is not a number, and NaN/Infinity are refused below


class EvaluatorOutcome(BaseModel):
    """One evaluator's outcome for one case.

    ok              metrics (all finite, at least one) and optionally a verdict; `score` is a
                    metric like any other
    not_applicable  the case cannot be scored by this evaluator (a reason in `detail`); it is
                    excluded from means and counted in coverage, never a zero
    failed          the evaluator could not produce a result: a classified failure
                    (input | evaluator | infrastructure -- a *target* failure is not this)
    skipped         nothing to evaluate because the case result failed; no metric, no verdict
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    evaluator_key: str = Field(min_length=1, max_length=128)
    status: Literal["ok", "not_applicable", "failed", "skipped"]
    verdict: Literal["PASS", "FAIL", "UNCERTAIN"] | None = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)  # evidence: reasoning, per-criterion, ...
    failure: Failure | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        if len(v) > MAX_METRICS:
            raise ValueError(f"at most {MAX_METRICS} metrics per evaluator result")
        for name, value in v.items():
            if not _METRIC_RE.match(name):
                raise ValueError(f"invalid metric name {name!r}")
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"metric {name!r} is not a finite number")
        return {name: float(value) for name, value in v.items()}

    @field_validator("detail")
    @classmethod
    def _plain_detail(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _json_object(v, "detail")

    @model_validator(mode="after")
    def _coherent(self) -> EvaluatorOutcome:
        s = self.status
        if s != "ok" and (self.metrics or self.verdict is not None):
            raise ValueError(f"a {s} evaluator result carries no metrics and no verdict")
        if s == "ok":
            if not self.metrics:
                raise ValueError("an ok evaluator result needs at least one metric")
            if self.failure is not None:
                raise ValueError("an ok evaluator result carries no failure")
        elif s == "failed":
            if self.failure is None:
                raise ValueError("a failed evaluator result needs a failure")
            _check_failure_classes(self.failure, EVALUATOR_RESULT_CLASSES, "evaluator-result")
        else:
            if self.failure is not None:
                raise ValueError(f"a {s} evaluator result carries no failure")
            reason = self.detail.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"a {s} evaluator result needs detail['reason']")
        return self

    @property
    def score(self) -> float | None:
        return self.metrics.get("score")


class EvaluatorResult(EvaluatorOutcome):
    id: str
    case_result_id: str
    run_id: str
    created_at: AwareDatetime
    attempts: list[RunAttempt] = Field(default_factory=list)  # evaluator calls

    model_config = ConfigDict(extra="forbid", strict=False)  # rows come from JSON text


# -- runs --------------------------------------------------------------------------------------


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    dataset_version_id: str
    dataset_ref: str  # name@version_no, for display; the id is the binding
    name: str | None = None
    status: RunStatus
    stop_reason: str | None = None
    config: RunConfig
    identity_hash: str
    exec_hash: str
    environment: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    created_at: AwareDatetime
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Run:
        s = self.status
        if (s != "created") != (self.started_at is not None):
            raise ValueError("started_at is set exactly once the run has started")
        if (s in TERMINAL_STATUSES) != (self.finished_at is not None):
            raise ValueError("finished_at is set exactly when the run is in a terminal status")
        if (s in _STOP_REASON_STATUSES) != (self.stop_reason is not None):
            raise ValueError("stop_reason is set exactly for partial, failed and cancelled runs")
        return self


@dataclass(frozen=True)
class RunCounts:
    total_cases: int  # cases in the run's dataset version
    pending: int
    complete: int
    failed: int  # target-stage failures (any class)
    missing: int  # cases with no case result row at all
    evaluator_results: dict[str, dict[str, int]] = field(default_factory=dict)  # key -> status -> n


@dataclass(frozen=True)
class WriteItem:
    """One result to persist in a batch: a case result (`key` = case_key) or an evaluator result
    (`key` = case_key of the case result it belongs to)."""

    kind: Literal["case", "evaluator"]
    case_key: str
    outcome: CaseOutcome | EvaluatorOutcome
    attempts: Sequence[RunAttempt] = ()


@dataclass(frozen=True)
class Unit:
    """A unit of work the executor still owes: a pending case result, or a complete one that is
    missing evaluator results (e.g. after a crash)."""

    result_id: str
    case_key: str
    status: Literal["pending", "complete"]
    output: str | None = None
    retrieved: list[str] | None = None


@dataclass(frozen=True)
class FailureRecord:
    scope: Literal["case", "evaluator"]
    result_id: str
    case_key: str
    evaluator_key: str | None  # None for a case (target-stage) failure
    failure: Failure


@dataclass(frozen=True)
class FailureCount:
    scope: Literal["case", "evaluator"]
    failure_class: str
    kind: str
    count: int


@dataclass
class RunVerifyReport:
    ok: bool
    problems: list[str]


class _RunStore(Protocol):  # what RunService needs; SQLiteStore implements it via RunStoreMixin
    def create_run(self, **fields: Any) -> Run: ...
    def get_run(self, run_id: str) -> Run: ...
    def list_runs(self, dataset_version_id: str | None, status: str | None, limit: int): ...
    def transition_run(self, run_id: str, status: str, stop_reason: str | None) -> Run: ...
    def plan_run(self, run_id: str, seed: int = 0) -> int: ...
    def record_case_result(self, run_id, case_key, outcome, attempts) -> CaseResult: ...
    def record_evaluator_result(self, case_result_id, outcome, attempts) -> EvaluatorResult: ...
    def get_case_result(self, run_id: str, case_key: str) -> CaseResult | None: ...
    def iter_case_results(self, run_id, status, batch_size) -> Iterator[CaseResult]: ...
    def list_evaluator_results(self, case_result_id: str) -> list[EvaluatorResult]: ...
    def run_counts(self, run_id: str) -> RunCounts: ...
    def list_failures(self, run_id: str, failure_class: str | None, limit: int): ...
    def failure_counts(self, run_id: str) -> list[FailureCount]: ...
    def verify_run(self, run_id: str) -> RunVerifyReport: ...


def _environment(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        version = importlib.metadata.version("evalkit")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    base = {
        "evalkit_version": version,
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
    }
    return base | _json_object(extra or {}, "environment")


def _validated(model: type[BaseModel], value: Any, what: str) -> Any:
    if isinstance(value, model):
        return value
    try:
        return model.model_validate(value)
    except ValidationError as e:
        raise RunError(f"invalid {what}: {format_validation_error(e)}") from e


class RunService:
    """Create runs and record their results. Obtain one via `EvalKit.runs`."""

    def __init__(self, store: _RunStore, datasets: DatasetService, limits: Limits | None = None):
        self.store = store
        self.datasets = datasets
        self.limits = limits or Limits()

    # -- runs -------------------------------------------------------------------------------

    def create(
        self,
        dataset_ref: str,
        config: RunConfig | Mapping[str, Any],
        *,
        name: str | None = None,
        idempotency_key: str | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> Run:
        """Freeze `config` against the dataset version `dataset_ref` resolves to, right now.

        `name@latest` is resolved once, here; the run keeps pointing at that exact version even
        after newer versions exist. With an `idempotency_key`, repeating the call returns the run
        already created for it, provided it was for the same dataset version and configuration;
        a different one is an error, never a silently different run.
        """
        config = _validated(RunConfig, config, "run config")
        if config.scoring_version != SCORING_VERSION:
            raise RunError(
                f"scoring_version {config.scoring_version} is not the current one "
                f"({SCORING_VERSION}); a new run is measured with the current scoring logic"
            )
        config_json = config.model_dump_json()
        if len(config_json.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise RunError(f"run config is larger than {MAX_CONFIG_BYTES} bytes")
        _label(name, "name")
        _label(idempotency_key, "idempotency_key")
        version = self.datasets.resolve(dataset_ref)
        identity_hash = config.identity_hash(version.content_hash)
        run = self.store.create_run(
            id=_new_id(),
            dataset_version_id=version.id,
            name=name,
            config=config,
            config_json=config_json,
            identity_hash=identity_hash,
            exec_hash=config.exec_hash,
            environment_json=json.dumps(_environment(environment), allow_nan=False),
            idempotency_key=idempotency_key,
            created_at=_now().isoformat(),
        )
        if idempotency_key is not None and (
            run.dataset_version_id != version.id
            or run.identity_hash != identity_hash
            or run.exec_hash != config.exec_hash
        ):
            raise RunError(
                f"idempotency_key {idempotency_key!r} was already used for a different dataset "
                f"version or configuration (run {run.id})"
            )
        return run

    def get(self, run_id: str) -> Run:
        return self.store.get_run(run_id)

    def list(
        self, *, dataset_ref: str | None = None, status: str | None = None, limit: int = 20
    ) -> list[Run]:
        """Newest first. `dataset_ref` filters to one dataset version (resolved now)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise RunError(f"limit must be an integer between 1 and 10000, got {limit!r}")
        if status is not None and status not in TRANSITIONS:
            raise RunError(f"unknown run status {status!r}")
        version_id = None if dataset_ref is None else self.datasets.resolve(dataset_ref).id
        return self.store.list_runs(version_id, status, limit)

    def transition(self, run_id: str, status: str, *, stop_reason: str | None = None) -> Run:
        """Move a run along its lifecycle (§4.2): created -> running -> succeeded | partial |
        cancelled | failed, and partial | cancelled | failed -> running (resume). A run stopped
        early says why (`stop_reason`). `succeeded` needs every case to have a terminal result."""
        if status not in TRANSITIONS:
            raise RunError(f"unknown run status {status!r}")
        if (status in _STOP_REASON_STATUSES) != (stop_reason is not None):
            raise RunError(
                "a partial, failed or cancelled run needs a stop_reason; others have none"
            )
        if stop_reason is not None:
            stop_reason = clip(scrub(stop_reason), MAX_ERROR_CHARS)
        return self.store.transition_run(run_id, status, stop_reason)

    def start(self, run_id: str) -> Run:
        """created -> running, or resume a partial / cancelled / failed run."""
        return self.transition(run_id, "running")

    # -- recording results ------------------------------------------------------------------

    def plan(self, run_id: str, *, seed: int = 0) -> int:
        """Create a `pending` case result for every case of the run's dataset version that has
        none yet; returns how many were created (0 when already planned). This is what makes
        "every case accounted for" checkable: a case is pending, done, failed, or has no row.
        Each row gets a processing order, a hash of (seed, case_key): a run that stops early has
        processed an unbiased sample of the dataset, not a prefix of the file."""
        return self.store.plan_run(run_id, seed)

    def record_case_result(
        self,
        run_id: str,
        case_key: str,
        outcome: CaseOutcome,
        *,
        attempts: Sequence[RunAttempt] = (),
    ) -> CaseResult:
        """Record the target-stage outcome of one case, with every attempt made to get it, in one
        transaction (all of it or none of it). Completes the case's pending result if `plan()` made
        one, otherwise creates it. The case must belong to the run's dataset version, the run must
        be running, and a case has one result: a second one raises `DuplicateResultError`."""
        outcome = _validated(CaseOutcome, outcome, "case outcome")
        if outcome.output is not None:
            _check_size(outcome.output, "output", self.limits.max_field_bytes)
        return self.store.record_case_result(
            run_id, case_key, outcome, _check_attempts(attempts, "case")
        )

    def record_evaluator_result(
        self,
        case_result_id: str,
        outcome: EvaluatorOutcome,
        *,
        attempts: Sequence[RunAttempt] = (),
    ) -> EvaluatorResult:
        """Record one evaluator's outcome for a case result, its metrics, and every attempt made,
        in one transaction. The evaluator must be one of the run's configured evaluators.
        A `complete` case result takes ok / not_applicable / failed; a `failed` one only `skipped`
        (a target failure is not scored); a pending one takes none."""
        outcome = _validated(EvaluatorOutcome, outcome, "evaluator outcome")
        _check_size(
            json.dumps(outcome.detail, ensure_ascii=False), "detail", self.limits.max_field_bytes
        )
        return self.store.record_evaluator_result(
            case_result_id, outcome, _check_attempts(attempts, "evaluator")
        )

    # -- reading ----------------------------------------------------------------------------

    def case_result(self, run_id: str, case_key: str) -> CaseResult | None:
        """One case's result with its target attempts; evaluator results via `evaluator_results`."""
        return self.store.get_case_result(run_id, case_key)

    def case_results(
        self, run_id: str, *, status: str | None = None, batch_size: int = 1000
    ) -> Iterator[CaseResult]:
        """Stream case results in ascending case_key order (attempts are not loaded)."""
        if status is not None and status not in ("pending", "complete", "failed"):
            raise RunError(f"unknown case result status {status!r}")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 10_000:
            raise RunError("batch_size must be an integer between 1 and 10000")
        return self.store.iter_case_results(run_id, status, batch_size)

    def evaluator_results(self, case_result_id: str) -> list[EvaluatorResult]:
        """A case result's evaluator results (metrics and attempts included), ordered by key."""
        return self.store.list_evaluator_results(case_result_id)

    def counts(self, run_id: str) -> RunCounts:
        return self.store.run_counts(run_id)

    def failures(
        self, run_id: str, *, failure_class: str | FailureClass | None = None, limit: int = 100
    ) -> list[FailureRecord]:
        """Failed results of both kinds, optionally of one class -- "the AI system's problems" is
        `failure_class="target"`, "rerun after the outage" is `"infrastructure"` (§7.5)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise RunError(f"limit must be an integer between 1 and 10000, got {limit!r}")
        cls = None if failure_class is None else _failure_class(failure_class).value
        return self.store.list_failures(run_id, cls, limit)

    def failure_counts(self, run_id: str) -> list[FailureCount]:
        return self.store.failure_counts(run_id)

    def verify(self, run_id: str) -> RunVerifyReport:
        """Recompute what can be recomputed (identity from the stored config and dataset content,
        evaluator keys, attempt numbering, result/metric relationships) and report every mismatch:
        detects changes made behind the database's back."""
        return self.store.verify_run(run_id)


def _failure_class(value: str | FailureClass) -> FailureClass:
    try:
        return FailureClass(value)
    except ValueError:
        known = ", ".join(c.value for c in FailureClass)
        raise RunError(f"unknown failure class {value!r} (known: {known})") from None


def _label(value: str | None, what: str) -> None:
    if value is not None and (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_TAG_CHARS
        or not value.isprintable()
    ):
        raise RunError(f"{what} must be 1-{MAX_TAG_CHARS} printable characters")


def _check_size(text: str, what: str, max_bytes: int) -> None:
    size = len(text.encode("utf-8", "replace"))
    if size > max_bytes:
        raise RunError(f"{what} is too large ({size} bytes; limit {max_bytes})")


def _check_attempts(attempts: Sequence[RunAttempt], owner: Literal["case", "evaluator"]):
    """Attempts are numbered 1..k in order (so none is silently missing) and their failure class
    fits what was called: a target call cannot fail as the evaluator, nor the reverse."""
    attempts = [_validated(RunAttempt, a, "attempt") for a in attempts]
    if len(attempts) > MAX_ATTEMPTS:
        raise RunError(f"at most {MAX_ATTEMPTS} attempts can be recorded with one result")
    if [a.n for a in attempts] != list(range(1, len(attempts) + 1)):
        raise RunError("attempts must be numbered 1, 2, 3, ... in order, with none missing")
    forbidden = FailureClass.EVALUATOR if owner == "case" else FailureClass.TARGET
    what = "a target call" if owner == "case" else "an evaluator call"
    for a in attempts:
        if a.error_class == forbidden:
            raise RunError(f"attempt {a.n}: {what} cannot fail with class {forbidden.value!r}")
    return attempts
