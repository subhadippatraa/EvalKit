"""SQLite persistence for runs and results, mixed into `SQLiteStore` (uses its connection and lock).

Every write is one `BEGIN IMMEDIATE` transaction: a result and all of its attempts and metrics are
stored together or not at all. The Python checks here give clear messages; the CHECKs and triggers
of migration 4 are the backstop that holds even for a writer that skips them.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from evalkit.errors import DuplicateResultError, RunError
from evalkit.failures import KINDS, RETRY_ALSO, RETRY_NEVER, Failure, FailureClass, check_kind
from evalkit.models import format_validation_error
from evalkit.runs import (
    TRANSITIONS,
    CaseOutcome,
    CaseResult,
    EvaluatorOutcome,
    EvaluatorResult,
    FailureCount,
    FailureRecord,
    RetryEval,
    Run,
    RunAttempt,
    RunConfig,
    RunCounts,
    RunVerifyReport,
    Unit,
    WriteItem,
)

_MAX_PROBLEMS = 100

_RUN_SELECT = (
    "SELECT r.*, d.name AS dataset_name, v.version_no AS dataset_version_no "
    "FROM runs r JOIN dataset_versions v ON v.id = r.dataset_version_id "
    "JOIN datasets d ON d.id = v.dataset_id"
)
_ATTEMPT_COLUMNS = (
    "id, n, outcome, started_at, duration_ms, provider, model, error_class, error_kind, "
    "error_type, error, http_status, request_id, input_tokens, output_tokens, raw_payload, "
    "raw_sha256, raw_truncated, cache_hit, cache_key, cost_usd, price_version, retry_round"
)


def _retry_predicate(alias: str, max_rounds: int) -> tuple[str, list[Any]]:
    """SQL (and its parameters) selecting failed rows that `--retry-failed` may retry: fewer than
    `max_rounds` earlier retries, and `evalkit.failures.retry_eligible` (kept in one place; this is
    that function as a WHERE clause, cross-checked by a test)."""
    also = sorted(f"{c.value}/{k}" for c, k in RETRY_ALSO)
    blocked = sorted(
        {f"{c.value}/{k}" for c, k in RETRY_NEVER}
        | {
            f"{c.value}/{k}"
            for c, kinds in KINDS.items()
            for k, info in kinds.items()
            if info.systemic and (c, k) not in RETRY_ALSO
        }
    )
    pair = f"({alias}.failure_class || '/' || {alias}.failure_kind)"
    marks = lambda xs: ",".join("?" * len(xs))  # noqa: E731
    sql = (
        f"({alias}.retry_round < ? AND "
        f"(({alias}.retryable = 1 AND {pair} NOT IN ({marks(blocked)})) "
        f"OR {pair} IN ({marks(also)})))"
    )
    return sql, [max_rounds, *blocked, *also]


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _corrupt(what: str, e: Exception) -> RunError:
    detail = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
    return RunError(f"stored {what} is corrupt or incoherent: {detail}")


def _run_from_row(row: sqlite3.Row) -> Run:
    try:
        return Run(
            id=row["id"],
            dataset_version_id=row["dataset_version_id"],
            dataset_ref=f"{row['dataset_name']}@{row['dataset_version_no']}",
            name=row["name"],
            status=row["status"],
            stop_reason=row["stop_reason"],
            config=RunConfig.model_validate_json(row["config_json"]),
            identity_hash=row["identity_hash"],
            exec_hash=row["exec_hash"],
            environment=json.loads(row["environment_json"]),
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
    except (ValidationError, ValueError) as e:
        raise _corrupt(f"run {row['id']!r}", e) from e


def _failure_from(row: sqlite3.Row, prefix: str = "failure") -> Failure | None:
    if row[f"{prefix}_class"] is None:
        return None
    return Failure(
        failure_class=FailureClass(row[f"{prefix}_class"]),
        kind=row[f"{prefix}_kind"],
        message=row[f"{prefix}_message"],
        retryable=bool(row["retryable"]),
    )


def _case_result_from(row: sqlite3.Row, attempts: list[RunAttempt]) -> CaseResult:
    try:
        return CaseResult(
            id=row["id"],
            run_id=row["run_id"],
            case_id=row["case_id"],
            case_key=row["case_key"],
            status=row["status"],
            output=row["output"],
            retrieved=None if row["retrieved_json"] is None else json.loads(row["retrieved_json"]),
            meta=None if row["meta_json"] is None else json.loads(row["meta_json"]),
            failure=_failure_from(row),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_ms=row["duration_ms"],
            retry_round=row["retry_round"],
            attempts=attempts,
        )
    except (ValidationError, ValueError) as e:
        raise _corrupt(f"case result {row['id']!r}", e) from e


def _attempt_from_row(row: sqlite3.Row) -> RunAttempt:
    try:
        return RunAttempt(
            id=row["id"],
            n=row["n"],
            outcome=row["outcome"],
            started_at=datetime.fromisoformat(row["started_at"]),
            duration_ms=row["duration_ms"],
            provider=row["provider"],
            model=row["model"],
            error_class=None if row["error_class"] is None else FailureClass(row["error_class"]),
            error_kind=row["error_kind"],
            error_type=row["error_type"],
            error=row["error"],
            http_status=row["http_status"],
            request_id=row["request_id"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            raw=row["raw_payload"],
            raw_sha256=row["raw_sha256"],
            raw_truncated=bool(row["raw_truncated"]),
            cache_hit=bool(row["cache_hit"]),
            cache_key=row["cache_key"],
            cost_usd=row["cost_usd"],
            price_version=row["price_version"],
            retry_round=row["retry_round"],
        )
    except (ValidationError, ValueError) as e:
        raise _corrupt(f"attempt {row['id']!r}", e) from e


def _translate(e: sqlite3.IntegrityError) -> RunError:
    text = str(e)
    if text.startswith("UNIQUE constraint failed") and (
        "case_results.run_id" in text or "evaluator_results.case_result_id" in text
    ):
        return DuplicateResultError(
            "a result already exists for this case (or evaluator) in this run; results are "
            "write-once"
        )
    return RunError(text)


class RunStoreMixin:
    _conn: sqlite3.Connection
    _lock: Any  # threading.Lock

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Lock, BEGIN IMMEDIATE (writers serialize), commit -- or roll everything back."""
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except sqlite3.IntegrityError as e:
                if conn.in_transaction:
                    conn.rollback()
                raise _translate(e) from e
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise

    # -- runs -----------------------------------------------------------------------------

    def _run_row(self, run_id: str) -> sqlite3.Row:
        row = self._conn.execute(_RUN_SELECT + " WHERE r.id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunError(f"run {run_id!r} not found")
        return row

    def create_run(
        self,
        *,
        id: str,
        dataset_version_id: str,
        name: str | None,
        config: RunConfig,
        config_json: str,
        identity_hash: str,
        exec_hash: str,
        environment_json: str,
        idempotency_key: str | None,
        created_at: str,
    ) -> Run:
        """Insert the run and its evaluators (one transaction). If `idempotency_key` already
        names a run, that run is returned untouched and the caller compares it."""
        with self._tx() as conn:
            if idempotency_key is not None:
                found = conn.execute(
                    "SELECT id FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if found is not None:
                    return _run_from_row(self._run_row(found["id"]))
            conn.execute(
                "INSERT INTO runs (id, dataset_version_id, name, status, config_json, "
                "identity_hash, exec_hash, evaluator_count, environment_json, idempotency_key, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    id,
                    dataset_version_id,
                    name,
                    "created",
                    config_json,
                    identity_hash,
                    exec_hash,
                    len(config.evaluators),
                    environment_json,
                    idempotency_key,
                    created_at,
                ),
            )
            conn.executemany(
                "INSERT INTO run_evaluators (run_id, evaluator_key, kind, name, version, spec_json)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (id, e.key, e.kind, e.name, e.version, e.model_dump_json())
                    for e in config.evaluators
                ],
            )
            return _run_from_row(self._run_row(id))

    def get_run(self, run_id: str) -> Run:
        with self._lock:
            return _run_from_row(self._run_row(run_id))

    def list_runs(self, dataset_version_id: str | None, status: str | None, limit: int):
        where, params = [], []
        if dataset_version_id is not None:
            where.append("r.dataset_version_id = ?")
            params.append(dataset_version_id)
        if status is not None:
            where.append("r.status = ?")
            params.append(status)
        sql = _RUN_SELECT + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [_run_from_row(r) for r in rows]

    def transition_run(self, run_id: str, status: str, stop_reason: str | None) -> Run:
        with self._tx() as conn:
            row = self._run_row(run_id)
            old = row["status"]
            if status not in TRANSITIONS[old]:
                allowed = ", ".join(sorted(TRANSITIONS[old])) or "none (terminal)"
                raise RunError(f"a {old} run cannot become {status} (allowed: {allowed})")
            if status == "succeeded":
                self._require_every_case_terminal(conn, row)
            now = datetime.now(UTC).isoformat()
            if status == "running":
                started, finished = row["started_at"] or now, None
            else:
                started, finished = row["started_at"], now
            conn.execute(
                "UPDATE runs SET status = ?, stop_reason = ?, started_at = ?, finished_at = ? "
                "WHERE id = ? AND status = ?",
                (status, stop_reason, started, finished, run_id, old),
            )
            return _run_from_row(self._run_row(run_id))

    def _require_every_case_terminal(self, conn: sqlite3.Connection, run: sqlite3.Row) -> None:
        done, total, evals = conn.execute(
            "SELECT (SELECT COUNT(*) FROM case_results WHERE run_id = ?1 AND status <> 'pending'), "
            "case_count, (SELECT COUNT(*) FROM evaluator_results WHERE run_id = ?1) "
            "FROM dataset_versions WHERE id = ?2",
            (run["id"], run["dataset_version_id"]),
        ).fetchone()
        if done != total:
            raise RunError(
                f"a run succeeds only when every case has a terminal result "
                f"({done} of {total} have one); stop it as partial instead"
            )
        expected = done * run["evaluator_count"]
        if evals != expected:
            raise RunError(
                f"a run succeeds only when every terminal case has an evaluator result for each "
                f"of its {run['evaluator_count']} evaluator(s) ({evals} of {expected} exist); "
                "stop it as partial instead"
            )

    # -- writing results ------------------------------------------------------------------

    def plan_run(self, run_id: str, seed: int = 0) -> int:
        with self._tx() as conn:
            row = self._run_row(run_id)
            if row["status"] not in ("created", "running"):
                raise RunError(f"cannot plan a {row['status']} run; resume it first")
            cursor = conn.execute(
                "INSERT INTO case_results (id, run_id, case_id, status, created_at, ord) "
                "SELECT lower(hex(randomblob(16))), ?, c.id, 'pending', ?, "
                "evalkit_ord(?, c.case_key) FROM cases c "
                "WHERE c.dataset_version_id = ? AND NOT EXISTS "
                "(SELECT 1 FROM case_results r WHERE r.run_id = ? AND r.case_id = c.id)",
                (
                    run_id,
                    datetime.now(UTC).isoformat(),
                    int(seed),
                    row["dataset_version_id"],
                    run_id,
                ),
            )
            return cursor.rowcount

    def _insert_attempts(
        self,
        conn: sqlite3.Connection,
        owner_column: str,
        owner_id: str,
        attempts: Sequence[RunAttempt],
    ) -> None:
        conn.executemany(
            f"INSERT INTO attempts (id, {owner_column}, n, provider, model, started_at, "
            "duration_ms, outcome, error_class, error_kind, error_type, error, http_status, "
            "request_id, input_tokens, output_tokens, raw_payload, raw_sha256, raw_truncated, "
            "cache_hit, cache_key, cost_usd, price_version, retry_round) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    a.id,
                    owner_id,
                    a.n,
                    a.provider,
                    a.model,
                    a.started_at.isoformat(),
                    a.duration_ms,
                    a.outcome,
                    None if a.error_class is None else a.error_class.value,
                    a.error_kind,
                    a.error_type,
                    a.error,
                    a.http_status,
                    a.request_id,
                    a.input_tokens,
                    a.output_tokens,
                    a.raw,
                    a.raw_sha256,
                    int(a.raw_truncated),
                    int(a.cache_hit),
                    a.cache_key,
                    a.cost_usd,
                    a.price_version,
                    a.retry_round,
                )
                for a in attempts
            ],
        )

    def record_case_result(
        self, run_id: str, case_key: str, outcome: CaseOutcome, attempts: Sequence[RunAttempt]
    ) -> CaseResult:
        with self._tx() as conn:
            result_id = self._write_case_result(conn, run_id, case_key, outcome, attempts)
            return self._case_result(result_id)

    def _write_case_result(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        case_key: str,
        outcome: CaseOutcome,
        attempts: Sequence[RunAttempt],
    ) -> str:
        """Must be called inside `_tx`."""
        run = self._run_row(run_id)
        case = conn.execute(
            "SELECT id FROM cases WHERE dataset_version_id = ? AND case_key = ?",
            (run["dataset_version_id"], case_key),
        ).fetchone()
        if case is None:
            raise RunError(
                f"case {case_key!r} is not part of run {run_id}'s dataset version "
                f"({run['dataset_name']}@{run['dataset_version_no']})"
            )
        existing = conn.execute(
            "SELECT id, status FROM case_results WHERE run_id = ? AND case_id = ?",
            (run_id, case["id"]),
        ).fetchone()
        if existing is not None and existing["status"] != "pending":
            raise DuplicateResultError(f"case {case_key!r} already has a result in this run")
        if run["status"] != "running":
            raise RunError(f"run {run_id} is {run['status']}; results need a running run")
        failure = outcome.failure
        values = {
            "status": "failed" if failure else "complete",
            "output": outcome.output,
            "retrieved_json": None if outcome.retrieved is None else json.dumps(outcome.retrieved),
            "meta_json": None
            if outcome.meta is None
            else json.dumps(outcome.meta, allow_nan=False),
            "failure_class": failure and failure.failure_class.value,
            "failure_kind": failure and failure.kind,
            "failure_message": failure and failure.message,
            "retryable": failure and int(failure.retryable),
            "started_at": _iso(outcome.started_at),
            "finished_at": _iso(outcome.finished_at),
            "duration_ms": outcome.duration_ms,
        }
        if existing is None:
            result_id = str(uuid.uuid4())
            cols = {
                "id": result_id,
                "run_id": run_id,
                "case_id": case["id"],
                "created_at": datetime.now(UTC).isoformat(),
                **values,
            }
            conn.execute(
                f"INSERT INTO case_results ({', '.join(cols)}) "
                f"VALUES ({', '.join(f':{k}' for k in cols)})",
                cols,
            )
        else:
            result_id = existing["id"]
            conn.execute(
                f"UPDATE case_results SET {', '.join(f'{k} = :{k}' for k in values)} "
                "WHERE id = :id AND status = 'pending'",
                {**values, "id": result_id},
            )
        self._insert_attempts(conn, "case_result_id", result_id, attempts)
        return result_id

    def record_evaluator_result(
        self, case_result_id: str, outcome: EvaluatorOutcome, attempts: Sequence[RunAttempt]
    ) -> EvaluatorResult:
        with self._tx() as conn:
            result_id = self._write_evaluator_result(conn, case_result_id, outcome, attempts)
            return self._evaluator_results("e.id = ?", (result_id,))[0]

    def _write_evaluator_result(
        self,
        conn: sqlite3.Connection,
        case_result_id: str,
        outcome: EvaluatorOutcome,
        attempts: Sequence[RunAttempt],
    ) -> str:
        """Must be called inside `_tx`."""
        cr = conn.execute(
            "SELECT r.run_id, r.status, x.status AS run_status FROM case_results r "
            "JOIN runs x ON x.id = r.run_id WHERE r.id = ?",
            (case_result_id,),
        ).fetchone()
        if cr is None:
            raise RunError(f"case result {case_result_id!r} not found")
        run_id = cr["run_id"]
        if cr["run_status"] != "running":
            raise RunError(f"run {run_id} is {cr['run_status']}; results need a running run")
        if not conn.execute(
            "SELECT 1 FROM run_evaluators WHERE run_id = ? AND evaluator_key = ?",
            (run_id, outcome.evaluator_key),
        ).fetchone():
            raise RunError(
                f"evaluator {outcome.evaluator_key!r} is not one of run {run_id}'s evaluators"
            )
        allowed = {
            "complete": ("ok", "not_applicable", "failed"),
            "failed": ("skipped",),
            "pending": (),
        }[cr["status"]]
        if outcome.status not in allowed:
            hint = ", ".join(allowed) or "none: the target stage has no outcome yet"
            raise RunError(
                f"a {outcome.status} evaluator result does not fit a {cr['status']} case "
                f"result (allowed: {hint})"
            )
        failure = outcome.failure
        result_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
            "verdict, detail_json, failure_class, failure_kind, failure_message, retryable, "
            "created_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                result_id,
                case_result_id,
                run_id,
                outcome.evaluator_key,
                outcome.status,
                outcome.verdict,
                json.dumps(outcome.detail, allow_nan=False),
                failure and failure.failure_class.value,
                failure and failure.kind,
                failure and failure.message,
                failure and int(failure.retryable),
                datetime.now(UTC).isoformat(),
                outcome.duration_ms,
            ),
        )
        conn.executemany(
            "INSERT INTO metrics (evaluator_result_id, run_id, evaluator_key, name, value) "
            "VALUES (?,?,?,?,?)",
            [
                (result_id, run_id, outcome.evaluator_key, name, value)
                for name, value in outcome.metrics.items()
            ],
        )
        self._insert_attempts(conn, "evaluator_result_id", result_id, attempts)
        return result_id

    def write_batch(self, run_id: str, items: Sequence[WriteItem]) -> None:
        """Persist many results in ONE transaction (the single-writer batch of design 8.1/11.1):
        all of them or none. An evaluator item finds its case result by case_key, so it may follow
        its case item in the same batch."""
        with self._tx() as conn:
            for item in items:
                if item.kind == "case":
                    self._write_case_result(
                        conn,
                        run_id,
                        item.case_key,
                        item.outcome,
                        item.attempts,  # type: ignore[arg-type]
                    )
                elif item.kind == "evaluator_retry":
                    self._replace_evaluator_result(conn, run_id, item)
                else:
                    row = conn.execute(
                        "SELECT r.id FROM case_results r JOIN cases c ON c.id = r.case_id "
                        "JOIN runs x ON x.id = r.run_id "
                        "AND c.dataset_version_id = x.dataset_version_id "
                        "WHERE r.run_id = ? AND c.case_key = ?",
                        (run_id, item.case_key),
                    ).fetchone()
                    if row is None:
                        raise RunError(f"case {item.case_key!r} has no case result in run {run_id}")
                    self._write_evaluator_result(
                        conn,
                        row["id"],
                        item.outcome,
                        item.attempts,  # type: ignore[arg-type]
                    )

    def _replace_evaluator_result(
        self, conn: sqlite3.Connection, run_id: str, item: WriteItem
    ) -> None:
        """A retried evaluator: archive the failed result, then give the same row the new outcome.
        Everything the failed try recorded (its attempts) stays attached to the row. Must be called
        inside `_tx`. A row that is no longer the failed one this try replaces (another writer, or
        a crash that stored it already) is a duplicate, never overwritten."""
        outcome: EvaluatorOutcome = item.outcome  # type: ignore[assignment]
        row = conn.execute(
            "SELECT e.*, r.status AS case_status, r.id AS cr_id FROM evaluator_results e "
            "JOIN case_results r ON r.id = e.case_result_id JOIN cases c ON c.id = r.case_id "
            "JOIN runs x ON x.id = r.run_id AND c.dataset_version_id = x.dataset_version_id "
            "WHERE r.run_id = ? AND c.case_key = ? AND e.evaluator_key = ?",
            (run_id, item.case_key, outcome.evaluator_key),
        ).fetchone()
        if row is None:
            raise RunError(f"case {item.case_key!r} has no result for {outcome.evaluator_key!r}")
        new_round = row["retry_round"] + 1
        if row["status"] != "failed" or (item.retry_round not in (None, new_round)):
            raise DuplicateResultError(
                f"the failed result of {outcome.evaluator_key!r} for case {item.case_key!r} was "
                "already superseded; results are never overwritten"
            )
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO result_history (id, run_id, scope, case_result_id, evaluator_result_id, "
            "evaluator_key, retry_round, failure_class, failure_kind, failure_message, retryable, "
            "duration_ms, finished_at, superseded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4()), run_id, "evaluator", row["cr_id"], row["id"],
                row["evaluator_key"], row["retry_round"], row["failure_class"],
                row["failure_kind"], row["failure_message"], row["retryable"],
                row["duration_ms"], row["created_at"], now,
            ),
        )  # fmt: skip
        failure = outcome.failure
        conn.execute(
            "UPDATE evaluator_results SET status = ?, verdict = ?, detail_json = ?, "
            "failure_class = ?, failure_kind = ?, failure_message = ?, retryable = ?, "
            "duration_ms = ?, retry_round = ? WHERE id = ? AND status = 'failed'",
            (
                outcome.status, outcome.verdict, json.dumps(outcome.detail, allow_nan=False),
                failure and failure.failure_class.value, failure and failure.kind,
                failure and failure.message, failure and int(failure.retryable),
                outcome.duration_ms, new_round, row["id"],
            ),
        )  # fmt: skip
        conn.executemany(
            "INSERT INTO metrics (evaluator_result_id, run_id, evaluator_key, name, value) "
            "VALUES (?,?,?,?,?)",
            [(row["id"], run_id, outcome.evaluator_key, n, v) for n, v in outcome.metrics.items()],
        )
        self._insert_attempts(conn, "evaluator_result_id", row["id"], item.attempts)  # type: ignore[arg-type]

    # -- retrying failed results (`--retry-failed`) ---------------------------------------

    def count_retryable(
        self, run_id: str, max_rounds: int, *, cases: bool = True
    ) -> tuple[int, int]:
        """(failed case results, failed evaluator results) that a retry would pick up."""
        case_sql, case_p = _retry_predicate("r", max_rounds)
        ev_sql, ev_p = _retry_predicate("e", max_rounds)
        with self._lock:
            self._run_row(run_id)
            n_case = 0
            if cases:
                n_case = self._conn.execute(
                    "SELECT COUNT(*) FROM case_results r WHERE r.run_id = ? "
                    f"AND r.status = 'failed' AND {case_sql}",
                    (run_id, *case_p),
                ).fetchone()[0]
            n_ev = self._conn.execute(
                "SELECT COUNT(*) FROM evaluator_results e JOIN case_results r "
                "ON r.id = e.case_result_id WHERE e.run_id = ? AND e.status = 'failed' "
                f"AND r.status = 'complete' AND {ev_sql}",
                (run_id, *ev_p),
            ).fetchone()[0]
        return n_case, n_ev

    def reopen_failed_cases(self, run_id: str, max_rounds: int, chunk: int = 500) -> int:
        """Reopen (pending again) every retryable failed case result: its failure is archived in
        `result_history` first, its `skipped` evaluator rows go, and its earlier attempts stay. In
        chunks of one transaction each, so a crash leaves some reopened and the rest untouched,
        each consistent. The run must be running."""
        sql, params = _retry_predicate("r", max_rounds)
        total = 0
        while True:
            with self._tx() as conn:
                rows = conn.execute(
                    "SELECT r.* FROM case_results r WHERE r.run_id = ? AND r.status = 'failed' "
                    f"AND {sql} ORDER BY r.id LIMIT ?",
                    (run_id, *params, chunk),
                ).fetchall()
                if not rows:
                    return total
                now = datetime.now(UTC).isoformat()
                conn.executemany(
                    "INSERT INTO result_history (id, run_id, scope, case_result_id, retry_round, "
                    "failure_class, failure_kind, failure_message, retryable, started_at, "
                    "finished_at, duration_ms, superseded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            str(uuid.uuid4()), run_id, "case", r["id"], r["retry_round"],
                            r["failure_class"], r["failure_kind"], r["failure_message"],
                            r["retryable"], r["started_at"], r["finished_at"], r["duration_ms"],
                            now,
                        )
                        for r in rows
                    ],
                )  # fmt: skip
                ids = [r["id"] for r in rows]
                marks = ",".join("?" * len(ids))
                conn.execute(
                    "UPDATE case_results SET status = 'pending', output = NULL, "
                    "retrieved_json = NULL, meta_json = NULL, failure_class = NULL, "
                    "failure_kind = NULL, failure_message = NULL, retryable = NULL, "
                    "started_at = NULL, finished_at = NULL, duration_ms = NULL, "
                    f"retry_round = retry_round + 1 WHERE status = 'failed' AND id IN ({marks})",
                    ids,
                )
                conn.execute(
                    f"DELETE FROM evaluator_results WHERE status = 'skipped' "
                    f"AND case_result_id IN ({marks})",
                    ids,
                )
                total += len(rows)

    def retry_units(self, run_id: str, max_rounds: int) -> list[Unit]:
        """Complete case results holding failed evaluator results a retry will replace, with what
        the executor needs: the stored output and, per evaluator, the new round and the number of
        attempts already made (numbering continues after them)."""
        sql, params = _retry_predicate("e", max_rounds)
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.id AS rid, r.output, r.retrieved_json, r.meta_json, c.case_key, "
                "e.id AS eid, e.evaluator_key, e.retry_round, "
                "(SELECT COUNT(*) FROM attempts a WHERE a.evaluator_result_id = e.id) AS prior "
                "FROM evaluator_results e JOIN case_results r ON r.id = e.case_result_id "
                "JOIN cases c ON c.id = r.case_id WHERE e.run_id = ? AND e.status = 'failed' "
                f"AND r.status = 'complete' AND {sql} "
                "ORDER BY COALESCE(r.ord, 0), r.id, e.evaluator_key",
                (run_id, *params),
            ).fetchall()
        units: dict[str, tuple[sqlite3.Row, dict[str, RetryEval]]] = {}
        for r in rows:
            _, evals = units.setdefault(r["rid"], (r, {}))
            evals[r["evaluator_key"]] = RetryEval(r["eid"], r["retry_round"] + 1, r["prior"])
        return [
            Unit(
                rid, r["case_key"], "complete", r["output"],
                None if r["retrieved_json"] is None else json.loads(r["retrieved_json"]),
                None if r["meta_json"] is None else json.loads(r["meta_json"]),
                retry_evals=evals,
            )
            for rid, (r, evals) in units.items()
        ]  # fmt: skip

    def failure_history(self, run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Superseded failures, oldest first: what each retried unit failed with before."""
        with self._lock:
            self._run_row(run_id)
            rows = self._conn.execute(
                "SELECT h.scope, c.case_key, h.evaluator_key, h.retry_round, h.failure_class, "
                "h.failure_kind, h.failure_message, h.retryable, h.superseded_at "
                "FROM result_history h JOIN case_results r ON r.id = h.case_result_id "
                "JOIN cases c ON c.id = r.case_id WHERE h.run_id = ? "
                "ORDER BY h.superseded_at, c.case_key, h.evaluator_key LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def retry_stats(self, run_id: str) -> dict[str, int]:
        with self._lock:
            n_case, n_ev, units = self._conn.execute(
                "SELECT COALESCE(SUM(scope = 'case'), 0), COALESCE(SUM(scope = 'evaluator'), 0), "
                "COUNT(DISTINCT COALESCE(evaluator_result_id, case_result_id)) "
                "FROM result_history WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return {"superseded_case_failures": n_case, "superseded_evaluator_failures": n_ev,
                "units_retried": units}  # fmt: skip

    def pending_units(self, run_id: str, n_evaluators: int) -> list[Unit]:
        """Terminal case results still missing evaluator results (crash recovery): complete ones
        missing real results, and failed ones missing their `skipped` rows."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.id, r.status, r.output, r.retrieved_json, r.meta_json, c.case_key "
                "FROM case_results r "
                "JOIN cases c ON c.id = r.case_id WHERE r.run_id = ? "
                "AND r.status IN ('complete', 'failed') "
                "AND (SELECT COUNT(*) FROM evaluator_results e WHERE e.case_result_id = r.id) < ? "
                "ORDER BY COALESCE(r.ord, 0), r.id",
                (run_id, n_evaluators),
            ).fetchall()
        return [
            Unit(
                r["id"],
                r["case_key"],
                r["status"],
                r["output"],
                None if r["retrieved_json"] is None else json.loads(r["retrieved_json"]),
                None if r["meta_json"] is None else json.loads(r["meta_json"]),
            )
            for r in rows
        ]

    def next_pending(
        self, run_id: str, after: tuple[int, str] | None, limit: int
    ) -> list[tuple[int, Unit]]:
        """Pending case results in processing order, keyset-paged after `(ord, id)`.

        The predicate and the ORDER BY are exactly the columns of `idx_case_results_claim (run_id,
        status, ord, id)`, so a page is an index range scan: the cost of a page does not depend on
        how many pending rows remain. (An expression such as COALESCE(ord, 0) here defeats the
        index and turns a run into a quadratic sort; migration 6 guarantees a pending row has an
        `ord`, so none is needed.)"""
        where, params = "", []
        if after is not None:
            where, params = " AND (r.ord, r.id) > (?, ?)", [after[0], after[1]]
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.id, r.ord AS o, c.case_key, r.retry_round AS rr, CASE WHEN "
                "r.retry_round > 0 THEN (SELECT COUNT(*) FROM attempts a "
                "WHERE a.case_result_id = r.id) ELSE 0 END AS prior "
                "FROM case_results r JOIN cases c ON c.id = r.case_id "
                f"WHERE r.run_id = ? AND r.status = 'pending'{where} "
                "ORDER BY r.ord, r.id LIMIT ?",
                (run_id, *params, limit),
            ).fetchall()
        return [
            (
                r["o"],
                Unit(
                    r["id"],
                    r["case_key"],
                    "pending",
                    retry_round=r["rr"],
                    prior_attempts=r["prior"],
                ),
            )
            for r in rows
        ]

    def done_evaluator_keys(self, case_result_id: str) -> set[str]:
        with self._lock:
            return {
                r[0]
                for r in self._conn.execute(
                    "SELECT evaluator_key FROM evaluator_results WHERE case_result_id = ?",
                    (case_result_id,),
                )
            }

    # -- reading results ------------------------------------------------------------------

    def _attempts_of(self, column: str, owner_ids: list[str]) -> dict[str, list[RunAttempt]]:
        found: dict[str, list[RunAttempt]] = {}
        for i in range(0, len(owner_ids), 500):
            chunk = owner_ids[i : i + 500]
            rows = self._conn.execute(
                f"SELECT {column} AS owner, {_ATTEMPT_COLUMNS} FROM attempts "
                f"WHERE {column} IN ({','.join('?' * len(chunk))}) ORDER BY {column}, n",
                chunk,
            ).fetchall()
            for row in rows:
                found.setdefault(row["owner"], []).append(_attempt_from_row(row))
        return found

    def _case_result(self, result_id: str) -> CaseResult:
        """Must be called with the lock held."""
        row = self._conn.execute(
            "SELECT r.*, c.case_key FROM case_results r JOIN cases c ON c.id = r.case_id "
            "WHERE r.id = ?",
            (result_id,),
        ).fetchone()
        attempts = self._attempts_of("case_result_id", [result_id]).get(result_id, [])
        return _case_result_from(row, attempts)

    def get_case_result(self, run_id: str, case_key: str) -> CaseResult | None:
        with self._lock:
            run = self._run_row(run_id)
            row = self._conn.execute(
                "SELECT r.id FROM case_results r JOIN cases c ON c.id = r.case_id "
                "WHERE r.run_id = ? AND c.dataset_version_id = ? AND c.case_key = ?",
                (run_id, run["dataset_version_id"], case_key),
            ).fetchone()
            return None if row is None else self._case_result(row["id"])

    def iter_case_results(
        self, run_id: str, status: str | None, batch_size: int
    ) -> Iterator[CaseResult]:
        """Ascending case_key, keyset-paged (bounded memory, lock released between batches)."""
        with self._lock:
            version_id = self._run_row(run_id)["dataset_version_id"]
        after = None
        while True:
            where, params = "", []
            if status is not None:
                where += " AND r.status = ?"
                params.append(status)
            if after is not None:
                where += " AND c.case_key > ?"
                params.append(after)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT r.*, c.case_key FROM cases c JOIN case_results r "
                    "ON r.case_id = c.id AND r.run_id = ? "
                    f"WHERE c.dataset_version_id = ?{where} ORDER BY c.case_key LIMIT ?",
                    (run_id, version_id, *params, batch_size),
                ).fetchall()
            for row in rows:
                yield _case_result_from(row, [])
            if len(rows) < batch_size:
                return
            after = rows[-1]["case_key"]

    def _evaluator_results(self, where: str, params: tuple) -> list[EvaluatorResult]:
        """Must be called with the lock held."""
        rows = self._conn.execute(
            f"SELECT e.* FROM evaluator_results e WHERE {where} ORDER BY e.evaluator_key", params
        ).fetchall()
        ids = [r["id"] for r in rows]
        metrics: dict[str, dict[str, float]] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            for m in self._conn.execute(
                "SELECT evaluator_result_id, name, value FROM metrics WHERE evaluator_result_id "
                f"IN ({','.join('?' * len(chunk))}) ORDER BY name",
                chunk,
            ):
                metrics.setdefault(m["evaluator_result_id"], {})[m["name"]] = m["value"]
        attempts = self._attempts_of("evaluator_result_id", ids)
        out = []
        for r in rows:
            try:
                out.append(
                    EvaluatorResult(
                        id=r["id"],
                        case_result_id=r["case_result_id"],
                        run_id=r["run_id"],
                        evaluator_key=r["evaluator_key"],
                        status=r["status"],
                        verdict=r["verdict"],
                        metrics=metrics.get(r["id"], {}),
                        detail=json.loads(r["detail_json"]),
                        failure=_failure_from(r),
                        duration_ms=r["duration_ms"],
                        created_at=r["created_at"],
                        retry_round=r["retry_round"],
                        attempts=attempts.get(r["id"], []),
                    )
                )
            except (ValidationError, ValueError) as e:
                raise _corrupt(f"evaluator result {r['id']!r}", e) from e
        return out

    def list_evaluator_results(self, case_result_id: str) -> list[EvaluatorResult]:
        with self._lock:
            return self._evaluator_results("e.case_result_id = ?", (case_result_id,))

    # -- summaries ------------------------------------------------------------------------

    def run_counts(self, run_id: str) -> RunCounts:
        with self._lock:
            run = self._run_row(run_id)
            (total,) = self._conn.execute(
                "SELECT case_count FROM dataset_versions WHERE id = ?", (run["dataset_version_id"],)
            ).fetchone()
            by_status = dict.fromkeys(("pending", "complete", "failed"), 0)
            for r in self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM case_results WHERE run_id = ? GROUP BY status",
                (run_id,),
            ):
                by_status[r["status"]] = r["n"]
            evaluators: dict[str, dict[str, int]] = {}
            for r in self._conn.execute(
                "SELECT evaluator_key, status, COUNT(*) AS n FROM evaluator_results "
                "WHERE run_id = ? GROUP BY evaluator_key, status",
                (run_id,),
            ):
                evaluators.setdefault(r["evaluator_key"], {})[r["status"]] = r["n"]
        return RunCounts(
            total_cases=total,
            pending=by_status["pending"],
            complete=by_status["complete"],
            failed=by_status["failed"],
            missing=total - sum(by_status.values()),
            evaluator_results=evaluators,
        )

    def list_failures(
        self, run_id: str, failure_class: str | None, limit: int
    ) -> list[FailureRecord]:
        extra = "" if failure_class is None else " AND {t}.failure_class = ?"
        args = [] if failure_class is None else [failure_class]
        sql = (
            "SELECT 'case' AS scope, r.id AS result_id, c.case_key, NULL AS evaluator_key, "
            "r.failure_class, r.failure_kind, r.failure_message, r.retryable "
            "FROM case_results r JOIN cases c ON c.id = r.case_id "
            f"WHERE r.run_id = ? AND r.status = 'failed'{extra.format(t='r')} "
            "UNION ALL "
            "SELECT 'evaluator', e.id, c.case_key, e.evaluator_key, "
            "e.failure_class, e.failure_kind, e.failure_message, e.retryable "
            "FROM evaluator_results e JOIN case_results r ON r.id = e.case_result_id "
            "JOIN cases c ON c.id = r.case_id "
            f"WHERE e.run_id = ? AND e.status = 'failed'{extra.format(t='e')} "
            "ORDER BY case_key, evaluator_key LIMIT ?"
        )
        with self._lock:
            self._run_row(run_id)
            rows = self._conn.execute(sql, (run_id, *args, run_id, *args, limit)).fetchall()
        try:
            return [
                FailureRecord(
                    scope=r["scope"],
                    result_id=r["result_id"],
                    case_key=r["case_key"],
                    evaluator_key=r["evaluator_key"],
                    failure=Failure(
                        failure_class=FailureClass(r["failure_class"]),
                        kind=r["failure_kind"],
                        message=r["failure_message"],
                        retryable=bool(r["retryable"]),
                    ),
                )
                for r in rows
            ]
        except (ValidationError, ValueError) as e:
            raise _corrupt(f"failure record in run {run_id!r}", e) from e

    def failure_counts(self, run_id: str) -> list[FailureCount]:
        with self._lock:
            self._run_row(run_id)
            rows = self._conn.execute(
                "SELECT 'case' AS scope, failure_class, failure_kind, COUNT(*) AS n "
                "FROM case_results WHERE run_id = ? AND status = 'failed' "
                "GROUP BY failure_class, failure_kind "
                "UNION ALL "
                "SELECT 'evaluator', failure_class, failure_kind, COUNT(*) "
                "FROM evaluator_results WHERE run_id = ? AND status = 'failed' "
                "GROUP BY failure_class, failure_kind ORDER BY 1, 2, 3",
                (run_id, run_id),
            ).fetchall()
        return [
            FailureCount(r["scope"], r["failure_class"], r["failure_kind"], r["n"]) for r in rows
        ]

    # -- integrity ------------------------------------------------------------------------

    def verify_run(self, run_id: str) -> RunVerifyReport:
        problems: list[str] = []

        def problem(text: str) -> None:
            if len(problems) < _MAX_PROBLEMS:
                problems.append(text)

        with self._lock:
            q = self._conn.execute
            row = self._run_row(run_id)
            version_id = row["dataset_version_id"]
            try:
                config = RunConfig.model_validate_json(row["config_json"])
            except ValidationError as e:
                problem(f"stored config is invalid: {format_validation_error(e)}")
                config = None
            content_hash, case_count = q(
                "SELECT content_hash, case_count FROM dataset_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if config is not None:
                if config.identity_hash(content_hash) != row["identity_hash"]:
                    problem("identity_hash does not match the stored config and dataset content")
                if config.exec_hash != row["exec_hash"]:
                    problem("exec_hash does not match the stored config")
                stored = {
                    (r["evaluator_key"], r["kind"], r["name"], r["version"])
                    for r in q(
                        "SELECT evaluator_key, kind, name, version FROM run_evaluators "
                        "WHERE run_id = ?",
                        (run_id,),
                    )
                }
                if stored != {(e.key, e.kind, e.name, e.version) for e in config.evaluators}:
                    problem("registered evaluators do not match the stored config")
                if row["evaluator_count"] != len(config.evaluators):
                    problem("evaluator_count does not match the stored config")

            def count(sql: str, *args: Any) -> int:
                return q(sql, args).fetchone()[0]

            n = count(
                "SELECT COUNT(*) FROM case_results r LEFT JOIN cases c ON c.id = r.case_id "
                "WHERE r.run_id = ? AND (c.id IS NULL OR c.dataset_version_id <> ?)",
                run_id,
                version_id,
            )
            if n:
                problem(f"{n} case result(s) reference a case outside the run's dataset version")
            n = count(
                "SELECT COUNT(*) FROM evaluator_results e JOIN case_results r "
                "ON r.id = e.case_result_id WHERE e.run_id = ? AND r.run_id <> e.run_id",
                run_id,
            )
            if n:
                problem(f"{n} evaluator result(s) carry a different run than their case result")
            n = count(
                "SELECT COUNT(*) FROM evaluator_results e WHERE e.run_id = ? AND NOT EXISTS "
                "(SELECT 1 FROM run_evaluators v WHERE v.run_id = e.run_id "
                "AND v.evaluator_key = e.evaluator_key)",
                run_id,
            )
            if n:
                problem(f"{n} evaluator result(s) use an evaluator the run does not have")
            n = count(
                "SELECT COUNT(*) FROM evaluator_results e JOIN case_results r "
                "ON r.id = e.case_result_id WHERE e.run_id = ? AND NOT ("
                "(r.status = 'complete' AND e.status <> 'skipped') "
                "OR (r.status = 'failed' AND e.status = 'skipped'))",
                run_id,
            )
            if n:
                problem(f"{n} evaluator result(s) do not fit their case result's status")
            n = count(
                "SELECT COUNT(*) FROM evaluator_results e WHERE e.run_id = ? AND e.status = 'ok' "
                "AND NOT EXISTS (SELECT 1 FROM metrics m WHERE m.evaluator_result_id = e.id)",
                run_id,
            )
            if n:
                problem(f"{n} ok evaluator result(s) have no metrics")
            n = count(
                "SELECT COUNT(*) FROM metrics m JOIN evaluator_results e "
                "ON e.id = m.evaluator_result_id WHERE m.run_id = ? AND "
                "(e.status <> 'ok' OR e.run_id <> m.run_id OR e.evaluator_key <> m.evaluator_key)",
                run_id,
            )
            if n:
                problem(f"{n} metric(s) do not match their evaluator result")
            for label, sql in (
                (
                    "target",
                    "SELECT COUNT(*) FROM (SELECT a.case_result_id FROM attempts a JOIN "
                    "case_results r ON r.id = a.case_result_id WHERE r.run_id = ? "
                    "GROUP BY a.case_result_id HAVING MIN(a.n) <> 1 OR MAX(a.n) <> COUNT(*))",
                ),
                (
                    "evaluator",
                    "SELECT COUNT(*) FROM (SELECT a.evaluator_result_id FROM attempts a JOIN "
                    "evaluator_results e ON e.id = a.evaluator_result_id WHERE e.run_id = ? "
                    "GROUP BY a.evaluator_result_id HAVING MIN(a.n) <> 1 OR MAX(a.n) <> COUNT(*))",
                ),
            ):
                if n := count(sql, run_id):
                    problem(f"{n} result(s) have {label} attempts that are not numbered 1..k")
            for label, sql in (
                (
                    "case",
                    "SELECT COUNT(*) FROM case_results r WHERE r.run_id = ? AND r.retry_round <> "
                    "(SELECT COUNT(*) FROM result_history h WHERE h.case_result_id = r.id "
                    "AND h.scope = 'case')",
                ),
                (
                    "evaluator",
                    "SELECT COUNT(*) FROM evaluator_results e WHERE e.run_id = ? AND e.retry_round "
                    "<> (SELECT COUNT(*) FROM result_history h WHERE h.evaluator_result_id = e.id)",
                ),
            ):
                if n := count(sql, run_id):
                    problem(
                        f"{n} {label} result(s) have a retry round that does not match their "
                        "archived failures"
                    )
            n = count(
                "SELECT COUNT(*) FROM result_history h JOIN case_results r "
                "ON r.id = h.case_result_id WHERE h.run_id = ? AND r.run_id <> h.run_id",
                run_id,
            )
            if n:
                problem(f"{n} history row(s) belong to another run's result")
            for scope, table in (("case", "case_results"), ("evaluator", "evaluator_results")):
                for cls, kind in q(
                    f"SELECT DISTINCT failure_class, failure_kind FROM {table} "
                    "WHERE run_id = ? AND failure_class IS NOT NULL",
                    (run_id,),
                ):
                    try:
                        check_kind(FailureClass(cls), kind)
                    except ValueError:
                        problem(f"{scope} result has unknown failure {cls}/{kind}")
            if row["status"] == "succeeded":
                done = count(
                    "SELECT COUNT(*) FROM case_results WHERE run_id = ? AND status <> 'pending'",
                    run_id,
                )
                if done != case_count:
                    problem(f"run is succeeded but {done} of {case_count} cases have results")
                evals = count("SELECT COUNT(*) FROM evaluator_results WHERE run_id = ?", run_id)
                if evals != done * row["evaluator_count"]:
                    problem(
                        f"run is succeeded but has {evals} evaluator result(s), expected "
                        f"{done * row['evaluator_count']} (evaluators x terminal cases)"
                    )
        return RunVerifyReport(ok=not problems, problems=problems)
