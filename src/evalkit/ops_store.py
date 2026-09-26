"""SQLite persistence for production operations, mixed into `SQLiteStore`: the response cache and
the per-execution records (migration 7). Uses the store's connection and lock."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Any


class OpsStoreMixin:
    _conn: sqlite3.Connection
    _lock: Any  # threading.Lock

    # -- response cache --------------------------------------------------------------------

    def cache_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else row[0]

    def cache_put(self, key: str, provider: str, model: str, response_json: str) -> None:
        """First writer wins (identical requests give interchangeable answers); a corrupt entry
        is replaced."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache (key, provider, model, response_json, created_at) "
                "VALUES (?,?,?,?,?)",
                (key, provider, model, response_json, datetime.now(UTC).isoformat()),
            )

    def cache_delete(self, key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))

    def cache_stats(self) -> dict[str, Any]:
        with self._lock:
            n, size, oldest, newest = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(length(response_json)), 0), MIN(created_at), "
                "MAX(created_at) FROM llm_cache"
            ).fetchone()
            by_model = self._conn.execute(
                "SELECT provider, model, COUNT(*) FROM llm_cache GROUP BY 1, 2 ORDER BY 1, 2"
            ).fetchall()
        return {
            "entries": n,
            "bytes": size,
            "oldest": oldest,
            "newest": newest,
            "by_model": [{"provider": p, "model": m, "entries": c} for p, m, c in by_model],
        }

    def cache_clear(self, older_than: str | None = None) -> int:
        """Delete every entry (or those created before `older_than`, an ISO timestamp)."""
        with self._lock, self._conn:
            if older_than is None:
                return self._conn.execute("DELETE FROM llm_cache").rowcount
            return self._conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?", (older_than,)
            ).rowcount

    # -- executions (one row per `execute`) -------------------------------------------------

    def begin_execution(
        self, run_id: str, *, retry_failed: bool, environment_json: str
    ) -> tuple[str, int, str]:
        """Record that an execution starts: returns (id, seq, status of the run before it). For a
        `succeeded` run being retried this row is also the ticket the database requires to reopen
        it (it names the `finished_at` reopened)."""
        import uuid

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                run = self._conn.execute(
                    "SELECT status, finished_at FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                seq = self._conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_executions WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
                eid = str(uuid.uuid4())
                reopens = (
                    run["finished_at"] if (retry_failed and run["status"] == "succeeded") else None
                )
                self._conn.execute(
                    "INSERT INTO run_executions (id, run_id, seq, started_at, status_before, "
                    "retry_failed, reopens_finished_at, environment_json) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        eid, run_id, seq, datetime.now(UTC).isoformat(), run["status"],
                        int(retry_failed), reopens, environment_json,
                    ),
                )  # fmt: skip
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return eid, seq, run["status"]

    def reopen_succeeded_run(self, run_id: str) -> None:
        """succeeded -> running, allowed by the database only with a matching execution row."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE runs SET status = 'running', stop_reason = NULL, finished_at = NULL "
                    "WHERE id = ? AND status = 'succeeded'",
                    (run_id,),
                )
                self._conn.commit()
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise

    def finish_execution(
        self,
        execution_id: str,
        *,
        reopened_cases: int,
        reopened_evaluators: int,
        units: int,
        elapsed_s: float,
        outcome_status: str,
        stop_reason: str | None,
        budget_json: str,
        spend_json: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE run_executions SET finished_at = ?, reopened_cases = ?, "
                "reopened_evaluators = ?, units = ?, elapsed_s = ?, outcome_status = ?, "
                "stop_reason = ?, budget_json = ?, spend_json = ? WHERE id = ? "
                "AND finished_at IS NULL",
                (
                    datetime.now(UTC).isoformat(), reopened_cases, reopened_evaluators, units,
                    elapsed_s, outcome_status, stop_reason, budget_json, spend_json, execution_id,
                ),
            )  # fmt: skip

    def list_executions(self, run_id: str) -> list[dict[str, Any]]:
        import json

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_executions WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("budget_json", "spend_json", "environment_json"):
                d[k[:-5]] = None if d.pop(k) is None else json.loads(r[k])
            out.append(d)
        return out

    def spend_totals(self, run_id: str) -> dict[str, Any]:
        """Everything the run's attempts recorded so far (all executions, all rounds)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cache_hit = 0), 0), COALESCE(SUM(cache_hit), 0), "
                "COALESCE(SUM(cache_hit = 0 AND outcome = 'failed'), 0), "
                "COALESCE(SUM(COALESCE(input_tokens, 0)), 0), "
                "COALESCE(SUM(COALESCE(output_tokens, 0)), 0), "
                "COALESCE(SUM(COALESCE(cost_usd, 0)), 0), "
                "COALESCE(SUM(cache_hit = 0 AND cost_usd IS NULL "
                "AND (outcome = 'ok' OR input_tokens IS NOT NULL)), 0), "
                "COALESCE(SUM(cache_key IS NOT NULL), 0), COALESCE(SUM(retry_round > 0), 0), "
                "COALESCE(SUM(n > 1), 0), COALESCE(SUM(duration_ms), 0) FROM ("
                "SELECT a.* FROM attempts a JOIN case_results r ON r.id = a.case_result_id "
                "WHERE r.run_id = ?1 UNION ALL SELECT a.* FROM attempts a "
                "JOIN evaluator_results e ON e.id = a.evaluator_result_id WHERE e.run_id = ?1)",
                (run_id,),
            ).fetchone()
        keys = (
            "calls", "cache_hits", "failed_calls", "input_tokens", "output_tokens", "cost_usd",
            "unpriced_calls", "cache_lookups", "retry_round_attempts", "in_unit_retries",
            "call_ms",
        )  # fmt: skip
        return dict(zip(keys, row, strict=True))
