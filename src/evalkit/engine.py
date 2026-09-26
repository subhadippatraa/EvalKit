"""Run execution (docs/TARGET-ARCHITECTURE.md §4, §8): the smallest correct executor.

    DatasetVersion -> Run -> [per case] Target -> CaseResult -> Evaluator(s) -> EvaluatorResult

* One process. A bounded window of units is in flight on a thread pool (`window`, default
  2 x concurrency); results go through ONE writer thread that commits batches, so a slow database
  blocks producers (backpressure) instead of growing memory. No asyncio, no broker.
* Units are processed in seeded-hash order, so a run that stops early has processed an unbiased
  sample of the dataset, not a prefix of the file.
* A unit is a case: its target stage, then each of the run's evaluators, independently. A failure
  is classified where it happens and recorded on the result that failed: a target failure is a
  case-result failure (its evaluators are `skipped`, never scored), an evaluator failure is that
  evaluator's alone. Nothing becomes a synthetic zero.
* Stopping. Cancellation and budget stop *dispatch*; in-flight units finish (a request in flight
  is never aborted). A systemic failure (auth, bad model id, quota) stops the run as `failed`; a
  tripped breaker or exhausted budget as `partial`. Everything completed is kept.
* Resume. `execute` on a partial / cancelled / failed run finishes what is left: pending case
  results, and complete ones missing evaluator results (crash recovery). Terminal results are
  never redone (they are write-once). `retry_failed=True` (`--retry-failed`) also gives failed
  units whose failure is retryable a new try, replacing the failed result in place with its failure
  archived in `result_history`; a successful call is never repeated, and a `succeeded` run with
  retryable failures is reopened for it. **Not here:** leases and multi-process workers.
* Calls (P2.1). Every provider call goes through the response cache (when the policy enables it),
  is priced against the run's versioned price table, and is admitted only if its worst case fits
  the token / USD / call budget, which counts what earlier executions of the run spent. A call the
  budget refuses is not made and its unit stays pending, so the run stops `partial(budget)` and a
  resume with a raised budget finishes it. Structured events (`evalkit.events`) narrate all of it.
* Storage failures never lose paid results: on a persistent write failure the unwritten results
  are spilled to `<db>.spill/<run_id>.jsonl`, the run stops `partial(infrastructure.storage)`,
  and the next `execute` replays the spill first.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from evalkit import events
from evalkit.cache import CachePolicy, ResponseCache
from evalkit.calls import (
    Budget,
    CallRunner,
    CancelToken,
    RateLimiter,
    RetryPolicy,
    RunGuard,
    SpendTotals,
    UnitAbandoned,
)
from evalkit.envsnapshot import execution_environment, frozen_environment
from evalkit.errors import ConfigError, DuplicateResultError, RunError
from evalkit.evaluators import (
    EVALUATOR_KINDS,
    CaseEvaluator,
    EvalContext,
    EvalInput,
    NotApplicable,
    build,
    resolve,
)
from evalkit.failures import EvalFailure, Failure, FailureClass
from evalkit.hashing import stable_hash
from evalkit.llm import LLMClient
from evalkit.models import format_validation_error
from evalkit.pricing import NO_PRICING, PriceTable
from evalkit.runlock import RunLock
from evalkit.runs import (
    CaseOutcome,
    EvaluatorOutcome,
    EvaluatorSpec,
    RetryEval,
    Run,
    RunAttempt,
    RunConfig,
    RunCounts,
    Unit,
    WriteItem,
)
from evalkit.targets import (
    CallableTarget,
    ModelTarget,
    PrecomputedTarget,
    ReuseTarget,
    Target,
    TargetInput,
    TargetOutput,
    spec_of,
    view,
)

if TYPE_CHECKING:
    from evalkit.kit import EvalKit

log = logging.getLogger("evalkit.engine")

_PAGE = 256


# ---- policy ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecPolicy:
    """How a run executes (`RunConfig.policy`; hashed into `exec_hash`, never into identity)."""

    concurrency: int = 4
    window: int | None = None  # units in flight; default 2 x concurrency
    batch_size: int = 100  # results per writer transaction
    batch_ms: int = 200  # ...or this long, whichever first
    seed: int = 0  # processing order
    on_missing: Literal["abort", "not_applicable"] = "abort"
    breaker_threshold: int = 10
    # consecutive failures of one systemic kind (auth, a rejected request, quota) from one provider
    # and model, with no success in between, that stop the run (design 8.4); one is not enough
    systemic_threshold: int = 3
    # a run whose evaluators scored less than this share of their applicable cases does not
    # succeed (it stops `partial(insufficient_coverage)`); it must always have scored something
    min_coverage: float = 0.0
    max_calls_per_s: float | None = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    budget: Budget = field(default_factory=Budget)
    cache: CachePolicy = field(default_factory=CachePolicy)
    pricing: PriceTable = NO_PRICING  # versioned prices; unpriced calls have an unknown cost
    # `--retry-failed`: a unit is retried at most this many times over the life of the run
    retry_failed_rounds: int = 3

    def __post_init__(self) -> None:
        def check(name: str, lo: int, hi: int) -> None:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
                raise ValueError(f"{name} must be an integer in {lo}..{hi}")

        check("concurrency", 1, 64)
        check("batch_size", 1, 10_000)
        check("batch_ms", 1, 60_000)
        check("breaker_threshold", 1, 10_000)
        check("systemic_threshold", 1, 10_000)
        check("retry_failed_rounds", 1, 20)
        mc = self.min_coverage
        if isinstance(mc, bool) or not isinstance(mc, int | float) or not 0.0 <= mc <= 1.0:
            raise ValueError("min_coverage must be a number within [0, 1]")
        if self.window is not None:
            check("window", self.concurrency, 10_000)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.on_missing not in ("abort", "not_applicable"):
            raise ValueError("on_missing must be 'abort' or 'not_applicable'")
        if self.max_calls_per_s is not None and not 0 < self.max_calls_per_s < float("inf"):
            raise ValueError("max_calls_per_s must be positive and finite")

    @property
    def in_flight(self) -> int:
        return self.window if self.window is not None else 2 * self.concurrency

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> ExecPolicy:
        m = dict(m)
        unknown = set(m) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown execution setting(s): {sorted(unknown)}")
        if "retry" in m:
            m["retry"] = RetryPolicy.from_mapping(m["retry"])
        if "budget" in m:
            m["budget"] = Budget.from_mapping(m["budget"])
        if "cache" in m:
            m["cache"] = CachePolicy.from_mapping(m["cache"])
        if "pricing" in m:
            m["pricing"] = PriceTable.from_mapping(m["pricing"])
        return cls(**m)


# ---- preflight ------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightIssue:
    level: Literal["error", "warning"]
    code: str
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class PreflightReport:
    """What was known before anything was spent (design 4.4). Stored on the run."""

    dataset: str
    cases: int
    issues: tuple[PreflightIssue, ...] = ()
    estimate: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cases": self.cases,
            "ok": self.ok,
            "issues": [i.to_json() for i in self.issues],
            "estimate": self.estimate,
        }


class PreflightError(RunError):
    """A run was refused before creation because preflight found errors."""

    def __init__(self, report: PreflightReport):
        lines = "; ".join(f"{i.code}: {i.message}" for i in report.errors[:5])
        super().__init__(f"preflight failed: {lines}")
        self.report = report


@dataclass
class ExecutionReport:
    run: Run
    counts: RunCounts
    stop_reason: str | None
    units: int  # units processed by this call
    elapsed_s: float
    spend: SpendTotals
    spilled: int = 0
    worker_errors: int = 0
    writer_error: str | None = None  # why the result writer stopped, if it did
    stop_detail: str | None = None  # the message of the failure that stopped the run, if any
    run_spend: SpendTotals | None = (
        None  # spend of *every* execution of the run (`spend` is this one)
    )
    retry: dict[str, int] = field(default_factory=dict)  # retry_failed: eligible / reopened
    execution_id: str | None = None

    @property
    def status(self) -> str:
        return self.run.status

    @property
    def units_per_s(self) -> float:
        return self.units / self.elapsed_s if self.elapsed_s > 0 else 0.0


# ---- the writer -----------------------------------------------------------------------------

_SENTINEL: Any = object()


def _item_line(item: WriteItem) -> str:
    return json.dumps(
        {
            "kind": item.kind,
            "case_key": item.case_key,
            "outcome": item.outcome.model_dump_json(),
            "attempts": [a.model_dump_json() for a in item.attempts],
            "retry_round": item.retry_round,
        }
    )


def _line_item(line: str) -> WriteItem:
    d = json.loads(line)
    model = CaseOutcome if d["kind"] == "case" else EvaluatorOutcome
    return WriteItem(
        d["kind"],
        d["case_key"],
        model.model_validate_json(d["outcome"]),
        [RunAttempt.model_validate_json(a) for a in d["attempts"]],
        d.get("retry_round"),
    )


class ResultWriter:
    """The single writer: batches results into one transaction each (design 8.1/11.1).

    A *group* of items (`put(a, b, c)`) is never split across transactions: a failed target and its
    `skipped` evaluator rows are stored together or not at all. The writer thread cannot die
    silently: any exception (a storage error, or a bug) is recorded in `failed`, what it held and
    what is still queued is spilled to disk, and every later `put` refuses instead of blocking, so
    the executor always learns of it and stops (audit P1-4)."""

    def __init__(
        self,
        store: Any,
        run_id: str,
        *,
        batch_size: int,
        batch_ms: int,
        depth: int,
        retries: int = 3,
    ):
        self._store, self._run_id = store, run_id
        self._batch_size, self._batch_s, self._retries = batch_size, batch_ms / 1000, retries
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(16, depth))
        self._thread = threading.Thread(target=self._loop, name="evalkit-writer", daemon=True)
        self.failed: BaseException | None = None
        self.written = 0
        self.duplicates = 0
        self.spilled = 0
        self.lost = 0  # results that could neither be stored nor spilled (both failed)
        self.spilled_items: list[WriteItem] = []  # in-memory stores have no spill directory
        self._closing = threading.Event()
        self._batch: list[tuple[WriteItem, ...]] = []  # what the thread holds right now
        self._thread.start()

    @property
    def crashed(self) -> bool:
        """The writer stopped on something other than a storage error: a bug."""
        return self.failed is not None and not isinstance(self.failed, sqlite3.Error | RunError)

    def put(self, item: WriteItem, *more: WriteItem) -> bool:
        """Queue a result group (blocks when the writer is behind: backpressure). False once the
        writer has given up or been closed; the group is then spilled, not queued."""
        group = (item, *more)
        while self.failed is None and not self._closing.is_set():
            try:
                self._queue.put(group, timeout=0.1)
                return True
            except queue.Full:
                continue
        self._spill(list(group))
        return False

    def close(self) -> None:
        """Flush what is queued and stop the writer thread."""
        self._closing.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass  # the loop also stops on its own once closing is set and the queue is empty
        self._thread.join()
        self._drain()  # a group queued in the instant before closing must not be dropped

    def _loop(self) -> None:
        try:
            self._run()
        except BaseException as e:  # noqa: BLE001 - nothing may kill this thread unrecorded
            log.error("the result writer stopped unexpectedly: %r", e)
            self.failed = e
            held, self._batch = self._batch, []
            self._spill([i for g in held for i in g])
            self._drain()

    def _run(self) -> None:
        while True:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    return
                continue
            if first is _SENTINEL:
                return
            batch = [first]
            size = len(first)
            deadline = time.monotonic() + self._batch_s
            stop = False
            while size < self._batch_size:
                try:
                    item = self._queue.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    stop = True
                    break
                batch.append(item)
                size += len(item)
            self._batch = batch
            self._flush(batch)
            self._batch = []
            if stop:
                return

    def _flush(self, batch: list[tuple[WriteItem, ...]]) -> None:
        if self.failed is not None:
            self._spill([i for g in batch for i in g])
            return
        items = [i for g in batch for i in g]
        delay = 0.05
        for attempt in range(self._retries + 1):
            try:
                self._store.write_batch(self._run_id, items)
                self.written += len(items)
                return
            except sqlite3.OperationalError as e:  # locked / busy / disk I/O: transient or fatal
                if attempt == self._retries:
                    self._fail(e, batch)
                    return
                time.sleep(delay)
                delay *= 2
            except (RunError, sqlite3.Error):
                self._one_by_one(batch)  # one bad group must not lose the others
                return

    def _one_by_one(self, batch: list[tuple[WriteItem, ...]]) -> None:
        for i, group in enumerate(batch):
            try:
                self._store.write_batch(self._run_id, list(group))
                self.written += len(group)
            except DuplicateResultError:
                self.duplicates += len(group)  # already stored (e.g. by a crashed earlier attempt)
            except (RunError, sqlite3.Error) as e:
                self._fail(e, batch[i:])
                return

    def _fail(self, error: BaseException, batch: list[tuple[WriteItem, ...]]) -> None:
        log.error("result storage failed (%s); spilling %d group(s)", error, len(batch))
        self.failed = error
        self._spill([i for g in batch for i in g])
        self._drain()

    def _drain(self) -> None:
        while True:
            try:
                group = self._queue.get_nowait()
            except queue.Empty:
                return
            if group is not _SENTINEL:
                self._spill(list(group))

    def _spill(self, items: list[WriteItem]) -> None:
        if not items:
            return
        directory = getattr(self._store, "spill_dir", None)
        self.spilled += len(items)
        if directory is None:
            self.spilled_items.extend(items)
            return
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = directory / f"{self._run_id}.jsonl"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                for item in items:
                    f.write(_item_line(item) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:  # the spill failed too (disk full?): say so, never raise from here
            self.spilled -= len(items)
            self.lost += len(items)
            log.error(
                "could not spill %d result(s) either (%s); they will be re-run", len(items), e
            )


# ---- the controller -------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _unpriced(target: Any, specs: Sequence[EvaluatorSpec], pricing: PriceTable) -> list[str]:
    """`provider/model` of every paid LLM component (model target, llm_judge) with no price."""
    pairs = []
    if target.kind == "model":
        pairs.append((target.identity.get("provider"), target.identity.get("model")))
    pairs += [
        (s.params.get("provider"), s.params.get("model")) for s in specs if s.kind == "llm_judge"
    ]
    return sorted({f"{p}/{m}" for p, m in pairs if pricing.price_of(p, m) is None})


def _internal(stage: str, e: BaseException) -> Failure:
    return Failure(
        failure_class=FailureClass.INFRA,
        kind="internal_error",
        message=f"unexpected {type(e).__name__} in {stage}: {e}",
    )


class RunController:
    """Creates (with preflight) and executes runs. Obtain one via `EvalKit.controller`."""

    def __init__(self, kit: EvalKit):
        self.kit = kit
        self.store = kit.store
        self.runs = kit.runs
        self.datasets = kit.datasets
        self.limits = kit.limits

    # -- planning and preflight -------------------------------------------------------------

    def preflight(
        self,
        dataset_ref: str,
        target: Target,
        evaluators: Sequence[EvaluatorSpec],
        policy: Mapping[str, Any] | None = None,
    ) -> PreflightReport:
        version = self.datasets.resolve(dataset_ref)
        issues: list[PreflightIssue] = []

        def error(code: str, message: str, count: int = 0, examples: Sequence[str] = ()) -> None:
            issues.append(PreflightIssue("error", code, message, count, tuple(examples)))

        def warn(code: str, message: str, count: int = 0, examples: Sequence[str] = ()) -> None:
            issues.append(PreflightIssue("warning", code, message, count, tuple(examples)))

        exec_policy: ExecPolicy | None = None
        try:
            exec_policy = ExecPolicy.from_mapping(policy or {})
        except (ValueError, TypeError) as e:
            error("bad_config", f"invalid execution policy: {e}")
        if not evaluators:
            error("no_evaluators", "a run needs at least one evaluator: it would measure nothing")
        specs: list[EvaluatorSpec] = []
        for spec in evaluators:
            try:
                specs.append(resolve(spec))
            except ConfigError as e:
                error("bad_config", f"evaluator {spec.name!r}: {e}")

        if exec_policy is not None and exec_policy.budget.max_cost_usd is not None:
            for model in _unpriced(spec_of(target), specs, exec_policy.pricing):
                error(
                    "budget_unpriced",
                    f"a USD budget needs a price for every model it pays for; {model} has none "
                    "in the pricing table (unpriced spend cannot be bounded)",
                )
        if isinstance(target, ReuseTarget):
            self._check_reuse(target, version.id, error)
        if isinstance(target, CallableTarget) and not target.pinned:
            warn(
                "unpinned_target",
                "the callable target declares no fingerprint: two runs of different code look "
                "identical (pass fingerprint=... to pin it)",
            )
        if isinstance(target, ModelTarget):
            self._check_contamination(target, version.id, warn)

        # per-evaluator requirements, one streaming pass over the cases
        need: dict[str, frozenset[str]] = {s.key: EVALUATOR_KINDS[s.kind].requires for s in specs}
        static_retrieved = target.kind == "precomputed"
        missing: dict[tuple[str, str], list[str]] = {}
        no_output: list[str] = []
        for case in self.datasets.cases(dataset_ref):
            if target.kind == "precomputed" and case.output is None:
                no_output.append(case.case_key)
            for spec in specs:
                for f in need[spec.key]:
                    if f == "retrieved" and not static_retrieved:
                        continue  # the target produces it
                    if getattr(case, f) is None:
                        missing.setdefault((spec.name, f), []).append(case.case_key)
        if no_output:
            error(
                "missing_field",
                "the precomputed target needs every case to carry an `output`",
                len(no_output),
                no_output[:5],
            )
        on_missing = (policy or {}).get("on_missing", "abort")
        for (name, f), keys in sorted(missing.items()):
            message = f"evaluator {name!r} needs `{f}`, missing in {len(keys)} case(s)"
            if on_missing == "abort":
                error("missing_field", message, len(keys), keys[:5])
            else:
                warn(
                    "missing_field", message + " (recorded as not_applicable)", len(keys), keys[:5]
                )
        for finding in self.datasets.lint(dataset_ref):
            warn(f"lint.{finding.code}", finding.message, finding.count, finding.examples)
        judges = sum(1 for s in specs if s.kind == "llm_judge")
        calls = {
            "target_calls": version.case_count if target.kind in ("callable", "model") else 0,
            "judge_calls": version.case_count * judges,
        }
        return PreflightReport(version.ref, version.case_count, tuple(issues), calls)

    def _check_reuse(self, target: ReuseTarget, version_id: str, error: Callable[..., None]):
        try:
            source = self.runs.get(target.source_run_id)
        except RunError as e:
            error("bad_config", f"source run: {e}")
            return
        if source.dataset_version_id != version_id:
            error(
                "bad_config",
                f"source run {source.id} used {source.dataset_ref}, not this run's dataset version",
            )
            return
        counts = self.runs.counts(source.id)
        if counts.pending or counts.missing:
            error(
                "bad_config",
                f"source run {source.id} is not finished ({counts.pending} pending, "
                f"{counts.missing} without a result); its outputs cannot be reused yet",
                counts.pending + counts.missing,
            )

    def _check_contamination(self, target: ModelTarget, version_id: str, warn: Callable[..., None]):
        """A prompt template must not contain any case's prompt or reference verbatim (few-shot
        leakage from the evaluation set)."""
        hits = []
        for case in self.datasets.store.iter_cases(version_id):
            for text in (case.prompt, case.reference):
                if text and len(text) >= 20 and text in target.template:
                    hits.append(case.case_key)
                    break
        if hits:
            warn(
                "contamination",
                "the model target's template contains evaluation-set text verbatim",
                len(hits),
                hits[:5],
            )

    # -- creation ---------------------------------------------------------------------------

    def create(
        self,
        dataset_ref: str,
        *,
        target: Target,
        evaluators: Sequence[EvaluatorSpec],
        policy: Mapping[str, Any] | None = None,
        name: str | None = None,
        idempotency_key: str | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> Run:
        """Preflight, freeze the run (resolved evaluator specs, the target's identity, the
        policy) and plan its case results. Refuses with `PreflightError` before anything is
        created if preflight found errors."""
        report = self.preflight(dataset_ref, target, evaluators, policy)
        if not report.ok:
            raise PreflightError(report)
        resolved = [resolve(s) for s in evaluators]
        target_spec = spec_of(target)
        config = RunConfig(target=target_spec, evaluators=resolved, policy=dict(policy or {}))
        exec_policy = ExecPolicy.from_mapping(dict(policy or {}))
        version = self.datasets.resolve(dataset_ref)
        # the policy is frozen with the run too; this is the plain-language record of everything
        # that defines the measurement (and what compare names when two runs differ). No secrets.
        retry = asdict(exec_policy.retry) | {"unit_deadline_s": exec_policy.retry.deadline_s}
        env = (
            dict(environment or {})
            | frozen_environment(
                dataset={
                    "ref": version.ref,
                    "version_id": version.id,
                    "content_hash": version.content_hash,
                    "case_count": version.case_count,
                },
                target_kind=target_spec.kind,
                target_identity=target_spec.identity,
                evaluators=resolved,
                scoring_version=config.scoring_version,
                policy=dict(policy or {}),
                retry=retry,
                pricing=exec_policy.pricing,
                cache=asdict(exec_policy.cache),
                budget={k: v for k, v in asdict(exec_policy.budget).items() if v is not None},
            )
            | {"preflight": report.to_json()}
        )
        run = self.runs.create(
            dataset_ref, config, name=name, idempotency_key=idempotency_key, environment=env
        )
        self.runs.plan(run.id, seed=exec_policy.seed)
        return run

    # -- execution --------------------------------------------------------------------------

    def _target_for(self, run: Run, supplied: Target | None) -> Target:
        spec = run.config.target
        if supplied is not None:
            if spec_of(supplied) != spec:
                raise RunError(
                    "the supplied target does not match the run's frozen target "
                    f"(run: {spec.kind} {spec.identity}; supplied: {supplied.kind} "
                    f"{supplied.identity})"
                )
            return supplied
        if spec.kind == "precomputed":
            return PrecomputedTarget()
        if spec.kind == "reuse":
            return ReuseTarget(spec.identity["source_run_id"])
        raise RunError(f"a {spec.kind} target must be supplied to execute (it cannot be persisted)")

    def replay_spill(self, run_id: str) -> int:
        """Write results a failed earlier execution spilled to disk; returns how many. The file is
        removed only after every line is safely stored."""
        directory = getattr(self.store, "spill_dir", None)
        path = None if directory is None else directory / f"{run_id}.jsonl"
        if path is None or not path.is_file():
            return 0
        items: list[WriteItem] = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        items.append(_line_item(line))
                    except (ValueError, KeyError, ValidationError) as e:
                        raise RunError(f"{path}:{n} is not a valid spilled result: {e}") from e
        # spill order across threads is not result order: case results always go first
        items.sort(key=lambda item: item.kind != "case")
        stored = 0
        for item in items:
            try:
                self.store.write_batch(run_id, [item])
                stored += 1
            except DuplicateResultError:
                stored += 1  # already there
        path.unlink()
        return stored

    def execute(
        self,
        run_id: str,
        *,
        target: Target | None = None,
        clients: Sequence[LLMClient] = (),
        token: CancelToken | None = None,
        on_progress: Callable[[int], None] | None = None,
        overrides: Mapping[str, Any] | None = None,
        retry_failed: bool = False,
    ) -> ExecutionReport:
        """Execute (or resume) a run and return what happened. See the module docstring.

        `overrides` adjusts execution settings for this call only (e.g. a raised budget to resume
        a run that stopped on it); they are not persisted and never affect the run's identity.

        `retry_failed` also retries failed units whose failure is eligible
        (`failures.retry_eligible`), at most `policy.retry_failed_rounds` times each. Only units
        that failed are touched: no successful result, and no successful provider call, is ever
        repeated. A `succeeded` run is reopened for it (and left `succeeded` if nothing is
        eligible).

        One executor per run: a second `execute` on a run that is being executed (in this process
        or another on this host) raises `RunBusyError` and makes no call. The lock dies with its
        holder, so a crashed executor never blocks the recovery of its run."""
        started = time.monotonic()
        run = self.runs.get(run_id)
        if run.status == "succeeded" and not retry_failed:
            raise RunError(f"run {run_id} already succeeded (--retry-failed retries its failures)")
        try:
            policy = ExecPolicy.from_mapping({**run.config.policy, **(overrides or {})})
        except (ValueError, TypeError) as e:
            raise RunError(f"run {run_id} has an invalid execution policy: {e}") from e
        if policy.budget.max_cost_usd is not None:
            gaps = _unpriced(run.config.target, run.config.evaluators, policy.pricing)
            if gaps:
                raise RunError(
                    f"a USD budget needs a price for every model the run pays for; no price for "
                    f"{', '.join(gaps)} (pass a pricing table, or budget tokens instead)"
                )
        evaluators = [build(spec, EvalContext(tuple(clients))) for spec in run.config.evaluators]
        tgt = self._target_for(run, target)  # config problems surface before any state change

        with RunLock(self.store.lock_dir, run_id, self.store.lock_scope):
            run = self.runs.get(run_id)  # re-read: the previous holder may have just finished
            if run.status == "succeeded" and not retry_failed:
                raise RunError(f"run {run_id} already succeeded")
            return self._execute_locked(
                run, policy, tgt, evaluators, token, on_progress, started, retry_failed,
                tuple(clients),
            )  # fmt: skip

    def _execute_locked(
        self,
        run: Run,
        policy: ExecPolicy,
        tgt: Target,
        evaluators: list[CaseEvaluator],
        token: CancelToken | None,
        on_progress: Callable[[int], None] | None,
        started: float,
        retry_failed: bool = False,
        clients: tuple[Any, ...] = (),
    ) -> ExecutionReport:
        run_id = run.id
        rounds = policy.retry_failed_rounds
        eligible = (0, 0)
        if retry_failed:
            eligible = self.store.count_retryable(run_id, rounds, cases=tgt.kind != "reuse")
            if run.status == "succeeded" and not any(eligible):
                # nothing to retry: the run stays exactly as it was
                return ExecutionReport(
                    run=run, counts=self.runs.counts(run_id), stop_reason=None, units=0,
                    elapsed_s=time.monotonic() - started, spend=SpendTotals(),
                    run_spend=self._run_spend(run_id),
                    retry={
                        "eligible_cases": 0, "eligible_evaluators": 0,
                        "reopened_cases": 0, "reopened_evaluators": 0,
                    },
                )  # fmt: skip
        env = execution_environment(clients, cache_mode=policy.cache.mode, pricing=policy.pricing)
        exec_id, seq, before = self.store.begin_execution(
            run_id, retry_failed=retry_failed, environment_json=json.dumps(env)
        )
        if before == "succeeded":  # only reachable with retry_failed (checked by the caller)
            self.store.reopen_succeeded_run(run_id)
            run = self.runs.get(run_id)
        elif run.status != "running":
            run = self.runs.start(run_id)
        self.replay_spill(run_id)
        self.runs.plan(run_id, seed=policy.seed)
        reopened = 0
        retry_units: list[Unit] = []
        if retry_failed:
            if tgt.kind != "reuse":  # a reuse target copies its source's failures: retry there
                reopened = self.store.reopen_failed_cases(run_id, rounds)
            retry_units = self.store.retry_units(run_id, rounds)
            events.emit(
                "retry_failed.reopened", run_id=run_id, execution=seq, reopened_cases=reopened,
                reopened_evaluators=sum(len(u.retry_evals) for u in retry_units),
            )  # fmt: skip
        events.emit(
            "run.started", run_id=run_id, execution=seq, status=run.status,
            cases=self.runs.counts(run_id).total_cases, cache_mode=policy.cache.mode,
        )  # fmt: skip

        token = token or CancelToken()
        guard = RunGuard(
            policy.budget,
            policy.breaker_threshold,
            systemic_threshold=policy.systemic_threshold,
            prior=self._spend_of(self.store.spend_totals(run_id)),
        )
        limiter = RateLimiter(policy.max_calls_per_s) if policy.max_calls_per_s else None
        cache = ResponseCache(self.store, policy.cache) if policy.cache.mode != "off" else None
        runner = CallRunner(
            policy.retry, token=token, guard=guard, limiter=limiter, cache=cache,
            pricing=policy.pricing,
        )  # fmt: skip
        writer = ResultWriter(
            self.store,
            run_id,
            batch_size=policy.batch_size,
            batch_ms=policy.batch_ms,
            depth=policy.in_flight * 8,
        )
        ctx = _Exec(run, policy, tgt, evaluators, runner, writer, self.store, self.limits)

        processed = 0
        worker_errors = 0
        in_flight: set[Future[None]] = set()

        def reap(done: set[Future[None]], notify: bool = True) -> None:
            nonlocal processed, worker_errors
            for f in done:
                in_flight.discard(f)
                if f.cancelled():
                    continue
                processed += 1
                if f.exception() is not None:  # _process handles its own errors; this is a bug
                    worker_errors += 1
                    log.error("unit crashed: %r", f.exception())
                if notify and on_progress is not None:
                    on_progress(processed)

        def stop() -> tuple[str, str] | None:
            if guard.abort is not None and guard.abort[0] == "failed":
                return guard.abort
            if token.cancelled:
                return "cancelled", token.reason or "cancelled"
            if guard.abort is not None:
                return guard.abort
            if writer.failed is not None:
                return "partial", (
                    "infrastructure.internal_error" if writer.crashed else "infrastructure.storage"
                )
            if worker_errors:
                return "partial", "infrastructure.internal_error"
            if (reason := guard.budget_stop()) is not None:
                return "partial", reason
            return None

        halted: tuple[str, str] | None = None
        interrupted: BaseException | None = None
        pool = ThreadPoolExecutor(max_workers=policy.concurrency, thread_name_prefix="evalkit")
        try:
            for unit in self._units(run_id, len(evaluators), retry_units):
                while len(in_flight) >= policy.in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    reap(done)
                if (halted := stop()) is not None:
                    break
                in_flight.add(pool.submit(ctx.process, unit))
            if in_flight:
                done, _ = wait(in_flight)
                reap(done)
        except BaseException as e:  # noqa: BLE001 - Ctrl-C or a bug: still flush and stop cleanly
            interrupted = e
            token.cancel("interrupted" if isinstance(e, KeyboardInterrupt) else "internal_error")
            pool.shutdown(wait=False, cancel_futures=True)  # queued units are dropped...
            running = [f for f in in_flight if not f.cancelled()]
            # ...running ones get a bounded grace period to finish and be recorded
            wait(running, timeout=policy.retry.timeout_s + 5)
            reap({f for f in running if f.done()}, notify=False)
        else:
            pool.shutdown(wait=True)
        # a systemic failure that never repeated was that case's own failure: record it now; one
        # that stopped the run is dropped, leaving its cases pending for the resume
        stopped_systemic = guard.abort is not None and guard.abort[0] == "failed"
        ctx.settle_held(record=not stopped_systemic)
        writer.close()
        halted = stop() or halted
        if interrupted is not None and (halted is None or halted[0] != "failed"):
            halted = (
                ("cancelled", "interrupted")
                if isinstance(interrupted, KeyboardInterrupt)
                else ("partial", "infrastructure.internal_error")
            )

        counts = self.runs.counts(run_id)
        reason: str | None = None
        final = self.runs.get(run_id)
        try:
            if halted is None and self._complete(counts, run, policy):
                final = self.runs.transition(run_id, "succeeded")
            else:
                if halted is None:
                    halted = self._incomplete_reason(counts, run)
                status, reason = halted
                final = self.runs.transition(run_id, status, stop_reason=reason)
            try:
                self.kit.snapshot(final.id)
            except Exception:  # noqa: BLE001 - a summary problem must not lose the report
                log.exception(
                    "could not store the run summary snapshot (recompute with summarize())"
                )
        except Exception:
            if interrupted is None:
                raise
            log.exception("could not record the stop of an interrupted run")
        elapsed = time.monotonic() - started
        budget_note = guard.budget_detail if guard.blocked else None
        if guard.blocked:
            events.emit(
                "budget.exhausted", 30, run_id=run_id, reason=guard.blocked[0],
                calls=guard.usage.calls,
            )  # fmt: skip
        self._finish_execution(
            exec_id, final, reopened, retry_units, processed, elapsed, reason, policy, guard
        )
        events.emit(
            "run.finished", run_id=run_id, execution=seq, status=final.status, stop_reason=reason,
            units=processed, elapsed_s=round(elapsed, 3), calls=guard.usage.calls,
            cache_hits=guard.usage.cache_hits, tokens=guard.usage.tokens,
            cost_usd=round(guard.usage.cost_usd, 8),
        )  # fmt: skip
        if interrupted is not None:
            raise interrupted
        return ExecutionReport(
            run=final,
            counts=self.runs.counts(run_id),
            stop_reason=reason,
            units=processed,
            elapsed_s=elapsed,
            spend=guard.usage,
            spilled=writer.spilled,
            worker_errors=worker_errors,
            writer_error=None
            if writer.failed is None
            else f"{type(writer.failed).__name__}: {writer.failed}",
            stop_detail=guard.abort_detail if stopped_systemic else budget_note,
            run_spend=self._run_spend(run_id),
            retry={
                "eligible_cases": eligible[0],
                "eligible_evaluators": eligible[1],
                "reopened_cases": reopened,
                "reopened_evaluators": sum(len(u.retry_evals) for u in retry_units),
            }
            if retry_failed
            else {},
            execution_id=exec_id,
        )

    @staticmethod
    def _spend_of(t: Mapping[str, Any]) -> SpendTotals:
        return SpendTotals(
            calls=t["calls"], failed_calls=t["failed_calls"], input_tokens=t["input_tokens"],
            output_tokens=t["output_tokens"], cache_hits=t["cache_hits"], cost_usd=t["cost_usd"],
            unpriced_calls=t["unpriced_calls"],
        )  # fmt: skip

    def _run_spend(self, run_id: str) -> SpendTotals:
        return self._spend_of(self.store.spend_totals(run_id))

    def _finish_execution(
        self, exec_id, final, reopened, retry_units, processed, elapsed, reason, policy, guard
    ) -> None:
        try:
            self.store.finish_execution(
                exec_id,
                reopened_cases=reopened,
                reopened_evaluators=sum(len(u.retry_evals) for u in retry_units),
                units=processed,
                elapsed_s=round(elapsed, 3),
                outcome_status=final.status,
                stop_reason=reason,
                budget_json=json.dumps(
                    {
                        "limits": {k: v for k, v in asdict(policy.budget).items() if v is not None},
                        "exhausted": None if guard.blocked is None else guard.blocked[0],
                        "detail": guard.budget_detail,
                    }
                ),
                spend_json=json.dumps(asdict(guard.usage)),
            )
        except Exception:  # noqa: BLE001 - bookkeeping must never lose the run's outcome
            log.exception("could not record the execution summary")

    # -- what "succeeded" means -------------------------------------------------------------

    @staticmethod
    def _complete(counts: RunCounts, run: Run, policy: ExecPolicy) -> bool:
        """Every case terminal, every evaluator has a result for each of them (the database
        insists too), and every evaluator actually *scored* enough of what it could score: a run
        in which nothing was measured has not succeeded, however many rows it stored."""
        terminal = counts.complete + counts.failed
        if counts.pending or counts.missing or not run.config.evaluators:
            return False
        for spec in run.config.evaluators:
            states = counts.evaluator_results.get(spec.key, {})
            if sum(states.values()) != terminal:
                return False
            scored = states.get("ok", 0)
            applicable = counts.total_cases - states.get("not_applicable", 0)
            if scored == 0 or scored / applicable < policy.min_coverage:
                return False
        return True

    @staticmethod
    def _incomplete_reason(counts: RunCounts, run: Run) -> tuple[str, str]:
        """Why a run that was not stopped by anything ended without succeeding."""
        terminal = counts.complete + counts.failed
        results_complete = all(
            sum(counts.evaluator_results.get(spec.key, {}).values()) == terminal
            for spec in run.config.evaluators
        )
        if not counts.pending and not counts.missing and results_complete and run.config.evaluators:
            return "partial", "insufficient_coverage"
        return "partial", "incomplete"

    def _units(
        self, run_id: str, n_evaluators: int, retry_units: Sequence[Unit] = ()
    ) -> Iterator[Unit]:
        """Units still owed: terminal case results missing evaluator results first (crash
        recovery, enumerated up front) and, on a retry, complete ones holding failed evaluator
        results to replace (merged into the recovery unit if the case is in both), then pending
        ones in processing order, keyset-paged."""
        owed = {u.result_id: u for u in self.store.pending_units(run_id, n_evaluators)}
        for u in retry_units:
            base = owed.get(u.result_id)
            owed[u.result_id] = u if base is None else replace(base, retry_evals=u.retry_evals)
        yield from owed.values()
        after: tuple[int, str] | None = None
        while True:
            page = self.store.next_pending(run_id, after, _PAGE)
            for order, unit in page:
                after = (order, unit.result_id)
                yield unit
            if len(page) < _PAGE:
                return


class _Exec:
    """The per-run state shared by worker threads; `process` handles one unit."""

    def __init__(self, run, policy, target, evaluators, runner, writer, store, limits):
        self.run, self.policy, self.target = run, policy, target
        self.evaluators: list[CaseEvaluator] = evaluators
        self.runner, self.writer, self.store, self.limits = runner, writer, store, limits
        # results of systemic-looking failures wait here until a later success of the same provider
        # and model shows the failure was that case's own (see RunGuard)
        self._held: list[tuple[tuple[str | None, str | None], int, tuple[WriteItem, ...]]] = []
        self._held_lock = threading.Lock()
        # part of every cache key of a target's calls: what the target *is*
        spec = spec_of(target)
        self._target_scope = "target:" + stable_hash(
            "evalkit-cache-scope-target-v1", {"kind": spec.kind, "identity": spec.identity}
        )

    # -- systemic failures: record only once they are known to be the case's own ----------------

    @staticmethod
    def _key(attempts: Sequence[RunAttempt]) -> tuple[str | None, str | None]:
        last = attempts[-1] if attempts else None
        return (None, None) if last is None else (last.provider, last.model)

    def _put_result(
        self, failure: Failure | None, attempts: Sequence[RunAttempt], *items: WriteItem
    ):
        """Queue a result group; a systemic-looking failure is held instead (see class comment)."""
        if failure is not None and failure.systemic:
            key = self._key(attempts)
            with self._held_lock:
                self._held.append((key, self.runner.guard.success_seq(*key), items))
            return True
        return self.writer.put(*items)

    def release_proven(self) -> None:
        """Record held failures that a later success of the same provider and model has proven to
        be case-specific."""
        guard = self.runner.guard
        with self._held_lock:
            keep, proven = [], []
            for entry in self._held:
                (proven if guard.success_seq(*entry[0]) > entry[1] else keep).append(entry)
            self._held = keep
        for _, _, items in proven:
            self.writer.put(*items)

    def settle_held(self, *, record: bool) -> None:
        with self._held_lock:
            held, self._held = self._held, []
        if not held:
            return
        if not record:
            log.warning(
                "%d result(s) of a systemic failure were not recorded; their cases stay pending",
                len(held),
            )
            return
        for _, _, items in held:
            self.writer.put(*items)

    # -- one unit ------------------------------------------------------------------------------

    @staticmethod
    def _skipped(ev: CaseEvaluator, case_key: str) -> WriteItem:
        outcome = EvaluatorOutcome(
            evaluator_key=ev.key, status="skipped", detail={"reason": "target failed"}
        )
        return WriteItem("evaluator", case_key, outcome, [])

    def process(self, unit: Unit) -> None:
        try:
            self._process(unit)
        except UnitAbandoned:
            pass  # a call was refused (budget) or replay found no response: nothing recorded, the
            # unit stays pending (or keeps the evaluator results it has) for the resume
        finally:
            self.release_proven()

    def _ctx(self, case_key: str, stage: str, evaluator: str | None = None) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "case_key": case_key,
            "stage": stage,
            "evaluator": evaluator,
            "correlation_id": uuid.uuid4().hex[:12],
        }

    def _process(self, unit: Unit) -> None:
        case = self.store.get_case(self.run.dataset_version_id, unit.case_key)
        if case is None:  # cannot happen: units come from this dataset version
            raise RunError(f"case {unit.case_key!r} vanished")
        done = self.store.done_evaluator_keys(unit.result_id) if unit.status != "pending" else set()
        if unit.status == "pending":
            outcome, attempts = self._target_stage(case, unit)
            case_item = WriteItem("case", unit.case_key, outcome, attempts)
            if outcome.failure is not None:
                # a failed target and its skipped evaluator rows are one group: never split
                skipped = [self._skipped(ev, unit.case_key) for ev in self.evaluators]
                self._put_result(outcome.failure, attempts, case_item, *skipped)
                return
            # A target that paid for its output is recorded at once (a crash must not lose it). A
            # free one (precomputed, reuse) is recorded together with its first evaluator result,
            # so a unit whose evaluator call the budget refuses is left wholly pending.
            held_case: WriteItem | None = None
            if attempts:
                if not self.writer.put(case_item):
                    return
            else:
                held_case = case_item
            output, retrieved, meta, failed = outcome.output, outcome.retrieved, outcome.meta, False
        else:
            held_case = None
            output, retrieved, meta = unit.output, unit.retrieved, unit.meta
            failed = unit.status == "failed"
        # a retry replaces the failed result of an evaluator that already has a row
        pending = [ev for ev in self.evaluators if ev.key not in done or ev.key in unit.retry_evals]
        if failed:  # a stranded failed case result: only its skipped rows are missing
            if pending:
                self.writer.put(*[self._skipped(ev, unit.case_key) for ev in pending])
            return
        for ev in pending:
            einp = EvalInput(
                case.case_key, case.prompt, output or "", case.reference, case.context,
                case.relevance, retrieved, meta,
            )  # fmt: skip
            retry = unit.retry_evals.get(ev.key)
            eo, attempts = self._evaluate(ev, einp, retry)
            item = WriteItem(
                "evaluator" if retry is None else "evaluator_retry",
                unit.case_key,
                eo,
                attempts,
                None if retry is None else retry.retry_round,
            )
            group = item if held_case is None else (held_case, item)
            held_case = None
            if not self._put_result(
                eo.failure, attempts, *(group if isinstance(group, tuple) else (group,))
            ):
                return
        if held_case is not None:  # no evaluator had anything to do
            self.writer.put(held_case)

    # -- target stage -----------------------------------------------------------------------

    def _target_input(self, case: Any) -> TargetInput:
        t = self.target
        if t.kind == "precomputed":
            return view(case, provide_output=True)
        if t.kind == "reuse":
            src = self.store.get_case_result(t.source_run_id, case.case_key)  # type: ignore[attr-defined]
            base = view(case)
            if src is None or src.status == "pending":
                return base
            if src.failure is not None:
                return TargetInput(base.case_key, base.prompt, base.context, None, src.failure)
            return TargetInput(
                base.case_key,
                base.prompt,
                base.context,
                TargetOutput(src.output or "", src.retrieved, meta=src.meta or {}),
            )
        return view(case)

    def _target_stage(self, case: Any, unit: Unit) -> tuple[CaseOutcome, list[RunAttempt]]:
        calls = self.runner.unit(
            first_n=unit.prior_attempts + 1,  # a retried case continues its attempt numbering
            scope=self._target_scope,
            retry_round=unit.retry_round,
            ctx=self._ctx(case.case_key, "target"),
        )
        started, t0 = _now(), time.monotonic()

        def timing() -> dict[str, Any]:
            return {
                "started_at": started,
                "finished_at": _now(),
                "duration_ms": round((time.monotonic() - t0) * 1000),
            }

        try:
            out = self.target.generate(self._target_input(case), calls)
            limit = self.limits.max_field_bytes
            if len(out.output.encode("utf-8", "replace")) > limit:
                raise EvalFailure(
                    FailureClass.TARGET,
                    "contract_violation",
                    f"the target's output is larger than {limit} bytes",
                )
            done = CaseOutcome.complete(
                out.output, retrieved=out.retrieved, meta=out.meta or None, **timing()
            )
            events.emit(
                "target.finished", 10, run_id=self.run.id, case_key=case.case_key,
                duration_ms=done.duration_ms, attempt=len(calls.attempts),
                retry_round=unit.retry_round,
            )  # fmt: skip
            return done, calls.attempts
        except UnitAbandoned:
            if calls.attempts:  # cannot happen (run() converts it); never lose paid evidence
                return CaseOutcome.fail(
                    Failure(
                        failure_class=FailureClass.INFRA, kind="budget_exceeded",
                        message="the budget was exhausted",
                    ), **timing(),
                ), calls.attempts  # fmt: skip
            raise
        except EvalFailure as f:
            events.emit(
                "target.failed", 30, run_id=self.run.id, case_key=case.case_key,
                failure_class=f.failure_class.value, failure_kind=f.kind,
                attempt=len(calls.attempts), retry_round=unit.retry_round,
            )  # fmt: skip
            return CaseOutcome.fail(f.failure, **timing()), calls.attempts
        except Exception as e:  # a bug in EvalKit, not the target's fault
            log.exception("unexpected error in the target stage")
            return CaseOutcome.fail(_internal("the target stage", e), **timing()), calls.attempts

    # -- evaluator stage --------------------------------------------------------------------

    def _evaluate(
        self, ev: CaseEvaluator, einp: EvalInput, retry: RetryEval | None = None
    ) -> tuple[EvaluatorOutcome, list[RunAttempt]]:
        round_ = 0 if retry is None else retry.retry_round
        calls = self.runner.unit(
            first_n=1 if retry is None else retry.prior_attempts + 1,
            scope=ev.key,
            retry_round=round_,
            ctx=self._ctx(einp.case_key, "evaluator", ev.key),
        )
        t0 = time.monotonic()

        def duration() -> int:
            return round((time.monotonic() - t0) * 1000)

        def failed(f: Failure) -> EvaluatorOutcome:
            events.emit(
                "evaluator.failed", 30, run_id=self.run.id, case_key=einp.case_key,
                evaluator=ev.key, failure_class=f.failure_class.value, failure_kind=f.kind,
                duration_ms=duration(), retry_round=round_,
            )  # fmt: skip
            return EvaluatorOutcome(
                evaluator_key=ev.key, status="failed", failure=f, duration_ms=duration()
            )

        missing = sorted(f for f in ev.requires if getattr(einp, f) is None)
        if missing:
            reason = f"the case has no {', '.join(missing)}"
            if self.policy.on_missing == "not_applicable":
                return EvaluatorOutcome(
                    evaluator_key=ev.key, status="not_applicable", detail={"reason": reason}
                ), []
            return failed(
                Failure(failure_class=FailureClass.INPUT, kind="missing_field", message=reason)
            ), []
        try:
            eo = ev.evaluate(einp, calls)
            outcome = EvaluatorOutcome(
                evaluator_key=ev.key,
                status="ok",
                verdict=eo.verdict,  # type: ignore[arg-type]
                metrics=eo.metrics,
                detail=eo.detail,
                duration_ms=duration(),
            )
            if (
                len(json.dumps(outcome.detail, ensure_ascii=False).encode())
                > self.limits.max_field_bytes
            ):
                raise EvalFailure(
                    FailureClass.EVALUATOR, "internal_error", "the evaluator's detail is too large"
                )
            events.emit(
                "evaluator.finished", 10, run_id=self.run.id, case_key=einp.case_key,
                evaluator=ev.key, duration_ms=duration(), retry_round=round_,
            )  # fmt: skip
            return outcome, calls.attempts
        except UnitAbandoned:
            if calls.attempts:  # cannot happen (run() converts it); never lose paid evidence
                return failed(
                    Failure(
                        failure_class=FailureClass.INFRA, kind="budget_exceeded",
                        message="the budget was exhausted",
                    )
                ), calls.attempts  # fmt: skip
            raise
        except NotApplicable as na:
            return EvaluatorOutcome(
                evaluator_key=ev.key,
                status="not_applicable",
                detail={"reason": na.reason},
                duration_ms=duration(),
            ), calls.attempts
        except EvalFailure as f:
            return failed(f.failure), calls.attempts
        except ValidationError as e:  # e.g. a non-finite metric: an evaluator bug, never a score
            msg = f"the evaluator produced an invalid outcome: {format_validation_error(e)}"
            return failed(
                Failure(failure_class=FailureClass.EVALUATOR, kind="internal_error", message=msg)
            ), calls.attempts
        except Exception as e:
            log.exception("unexpected error in evaluator %s", ev.key)
            return failed(
                Failure(
                    failure_class=FailureClass.EVALUATOR,
                    kind="internal_error",
                    message=f"{type(e).__name__}: {e}",
                )
            ), calls.attempts
