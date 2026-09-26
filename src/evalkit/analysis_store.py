"""Read-side queries for analysis, and the small write side (summaries, tags, reviews, judge
checks), mixed into `SQLiteStore`. Aggregation is done by SQL where it can be; only the values a
statistic needs (percentiles, intervals) are streamed into Python."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from evalkit.errors import RunError

_MAX_TAG = 128


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnalysisStoreMixin:
    _conn: sqlite3.Connection
    _lock: Any

    def _run_or_raise(self, run_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT r.id, r.status, r.dataset_version_id, r.identity_hash, r.stop_reason, "
            "r.config_json, v.case_count, v.content_hash FROM runs r "
            "JOIN dataset_versions v ON v.id = r.dataset_version_id WHERE r.id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RunError(f"run {run_id!r} not found")
        return row

    # -- accounting ---------------------------------------------------------------------------

    def case_accounting(self, run_id: str) -> list[tuple[str, str, int]]:
        """(status, failure_class or '', count) of the run's case results."""
        with self._lock:
            self._run_or_raise(run_id)
            rows = self._conn.execute(
                "SELECT status, COALESCE(failure_class, ''), COUNT(*) FROM case_results "
                "WHERE run_id = ? GROUP BY 1, 2",
                (run_id,),
            ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def evaluator_accounting(self, run_id: str) -> list[tuple[str, str, str, int]]:
        """(evaluator_key, status, failure_class or '', count)."""
        with self._lock:
            self._run_or_raise(run_id)
            rows = self._conn.execute(
                "SELECT evaluator_key, status, COALESCE(failure_class, ''), COUNT(*) "
                "FROM evaluator_results WHERE run_id = ? GROUP BY 1, 2, 3",
                (run_id,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def run_info(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._run_or_raise(run_id)
            return {k: row[k] for k in row.keys()}  # noqa: SIM118 - sqlite3.Row has no items()

    # -- metric data --------------------------------------------------------------------------

    def metric_names(self, run_id: str) -> list[tuple[str, str]]:
        with self._lock:
            self._run_or_raise(run_id)
            rows = self._conn.execute(
                "SELECT DISTINCT evaluator_key, name FROM metrics WHERE run_id = ? ORDER BY 1, 2",
                (run_id,),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def metric_values(self, run_id: str, key: str, name: str) -> list[float]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM metrics WHERE run_id = ? AND evaluator_key = ? AND name = ?",
                (run_id, key, name),
            ).fetchall()
        return [r[0] for r in rows]

    def metric_by_case(self, run_id: str, key: str, name: str) -> dict[str, float]:
        """case_key -> value, for pairing two runs."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.case_key, m.value FROM metrics m "
                "JOIN evaluator_results e ON e.id = m.evaluator_result_id "
                "JOIN case_results r ON r.id = e.case_result_id JOIN cases c ON c.id = r.case_id "
                "WHERE m.run_id = ? AND m.evaluator_key = ? AND m.name = ?",
                (run_id, key, name),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def case_outcomes(self, run_id: str) -> dict[str, dict[str, Any]]:
        """case_key -> {case: (status, class, kind), evaluators: {key: (status, class, verdict)},
        input_hash} -- everything needed to pair and exclude cases between two runs. `input_hash`
        is the question side of the case (never the system's output); it is what pairing uses."""
        with self._lock:
            self._run_or_raise(run_id)
            out: dict[str, dict[str, Any]] = {}
            for r in self._conn.execute(
                "SELECT c.case_key, c.input_hash, r.status, r.failure_class, r.failure_kind, "
                "r.duration_ms FROM case_results r JOIN cases c ON c.id = r.case_id "
                "WHERE r.run_id = ?",
                (run_id,),
            ):
                out[r[0]] = {
                    "input_hash": r[1],
                    "case": (r[2], r[3], r[4]),
                    "duration_ms": r[5],
                    "evaluators": {},
                }
            for r in self._conn.execute(
                "SELECT c.case_key, e.evaluator_key, e.status, e.failure_class, e.verdict "
                "FROM evaluator_results e JOIN case_results x ON x.id = e.case_result_id "
                "JOIN cases c ON c.id = x.case_id WHERE e.run_id = ?",
                (run_id,),
            ):
                out[r[0]]["evaluators"][r[1]] = (r[2], r[3], r[4])
        return out

    def verdict_counts(self, run_id: str, key: str) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(verdict, ''), COUNT(*) FROM evaluator_results "
                "WHERE run_id = ? AND evaluator_key = ? AND status = 'ok' GROUP BY 1",
                (run_id, key),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def durations(self, run_id: str) -> tuple[list[float], dict[str, list[float]]]:
        """Target latency of complete case results, and each evaluator's latency (ms)."""
        with self._lock:
            target = [
                r[0]
                for r in self._conn.execute(
                    "SELECT duration_ms FROM case_results WHERE run_id = ? AND status = 'complete' "
                    "AND duration_ms IS NOT NULL",
                    (run_id,),
                )
            ]
            evaluators: dict[str, list[float]] = {}
            for key, ms in self._conn.execute(
                "SELECT evaluator_key, duration_ms FROM evaluator_results WHERE run_id = ? "
                "AND status = 'ok' AND duration_ms IS NOT NULL",
                (run_id,),
            ):
                evaluators.setdefault(key, []).append(ms)
        return target, evaluators

    def attempt_stats(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope, provider, model, COUNT(*), SUM(outcome = 'failed'), SUM(n > 1), "
                "SUM(COALESCE(input_tokens, 0)), SUM(COALESCE(output_tokens, 0)), "
                "SUM(duration_ms) FROM ("
                "SELECT 'target' AS scope, a.* FROM attempts a "
                "JOIN case_results r ON r.id = a.case_result_id WHERE r.run_id = ? "
                "UNION ALL SELECT 'evaluator', a.* FROM attempts a "
                "JOIN evaluator_results e ON e.id = a.evaluator_result_id WHERE e.run_id = ?) "
                "GROUP BY scope, provider, model ORDER BY scope, provider, model",
                (run_id, run_id),
            ).fetchall()
        keys = ("scope", "provider", "model", "attempts", "failed", "retries", "in", "out", "ms")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def slice_stats(self, run_id: str) -> list[tuple[str, str, str, int, float]]:
        """(tag, evaluator_key, metric, n, mean) for every tag carried by scored cases."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.value, m.evaluator_key, m.name, COUNT(*), AVG(m.value) FROM metrics m "
                "JOIN evaluator_results e ON e.id = m.evaluator_result_id "
                "JOIN case_results r ON r.id = e.case_result_id JOIN cases c ON c.id = r.case_id, "
                "json_each(c.tags_json) t WHERE m.run_id = ? GROUP BY 1, 2, 3",
                (run_id,),
            ).fetchall()
        return [(r[0], r[1], r[2], r[3], r[4]) for r in rows]

    def score_vs_length(self, run_id: str, key: str) -> list[tuple[float, int]]:
        """(score, output length in characters) of every scored case of one evaluator."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.value, length(r.output) FROM metrics m "
                "JOIN evaluator_results e ON e.id = m.evaluator_result_id "
                "JOIN case_results r ON r.id = e.case_result_id "
                "WHERE m.run_id = ? AND m.evaluator_key = ? AND m.name = 'score'",
                (run_id, key),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def must_pass_failures(self, run_id: str, key: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM evaluator_results WHERE run_id = ? AND evaluator_key = ? "
                "AND status = 'ok' AND json_array_length(json_extract(detail_json, "
                "'$.failed_gates')) > 0",
                (run_id, key),
            ).fetchone()[0]

    # -- summaries ----------------------------------------------------------------------------

    def save_summary(self, run_id: str, version: str, summary: dict[str, Any]) -> str:
        computed_at = _now()
        with self._lock, self._conn:
            self._run_or_raise(run_id)
            self._conn.execute(
                "INSERT INTO run_summaries (run_id, computed_at, evalkit_version, summary_json) "
                "VALUES (?,?,?,?)",
                (run_id, computed_at, version, json.dumps(summary, allow_nan=False)),
            )
        return computed_at

    def latest_summary(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary_json, computed_at, evalkit_version FROM run_summaries "
                "WHERE run_id = ? ORDER BY computed_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]) | {"computed_at": row[1], "evalkit_version": row[2]}

    # -- tags ---------------------------------------------------------------------------------

    def add_run_tag(self, run_id: str, tag: str) -> None:
        if not isinstance(tag, str) or not 1 <= len(tag) <= _MAX_TAG or not tag.isprintable():
            raise RunError(f"a tag is 1-{_MAX_TAG} printable characters")
        with self._lock, self._conn:
            self._run_or_raise(run_id)
            self._conn.execute(
                "INSERT OR IGNORE INTO run_tags (run_id, tag, created_at) VALUES (?,?,?)",
                (run_id, tag, _now()),
            )

    def remove_run_tag(self, run_id: str, tag: str) -> bool:
        with self._lock, self._conn:
            return (
                self._conn.execute(
                    "DELETE FROM run_tags WHERE run_id = ? AND tag = ?", (run_id, tag)
                ).rowcount
                > 0
            )

    def run_tags(self, run_id: str) -> list[str]:
        with self._lock:
            return [
                r[0]
                for r in self._conn.execute(
                    "SELECT tag FROM run_tags WHERE run_id = ? ORDER BY tag", (run_id,)
                )
            ]

    def latest_run_with_tag(
        self, tag: str, dataset_version_id: str | None = None, exclude: str | None = None
    ) -> str | None:
        """The most recently created *succeeded* run carrying `tag` (optionally of one dataset
        version's dataset lineage): how a baseline like "tag:main" is resolved."""
        sql = (
            "SELECT r.id FROM runs r JOIN run_tags t ON t.run_id = r.id "
            "WHERE t.tag = ? AND r.status = 'succeeded'"
        )
        params: list[Any] = [tag]
        if exclude is not None:
            sql += " AND r.id <> ?"
            params.append(exclude)
        if dataset_version_id is not None:
            sql += (
                " AND r.dataset_version_id IN (SELECT id FROM dataset_versions WHERE dataset_id = "
                "(SELECT dataset_id FROM dataset_versions WHERE id = ?))"
            )
            params.append(dataset_version_id)
        sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return None if row is None else row[0]

    # -- reviews of case results --------------------------------------------------------------

    def add_case_review(
        self,
        run_id: str,
        case_key: str,
        *,
        evaluator_key: str | None,
        reviewer: str,
        verdict: str,
        score: float | None,
        comment: str | None,
        sample: str,
    ) -> str:
        review_id = str(uuid.uuid4())
        with self._lock, self._conn:
            self._run_or_raise(run_id)
            row = self._conn.execute(
                "SELECT r.id, r.status FROM case_results r JOIN cases c ON c.id = r.case_id "
                "JOIN runs x ON x.id = r.run_id AND c.dataset_version_id = x.dataset_version_id "
                "WHERE r.run_id = ? AND c.case_key = ?",
                (run_id, case_key),
            ).fetchone()
            if row is None:
                raise RunError(f"case {case_key!r} has no result in run {run_id}")
            if (
                evaluator_key is not None
                and not self._conn.execute(
                    "SELECT 1 FROM run_evaluators WHERE run_id = ? AND evaluator_key = ?",
                    (run_id, evaluator_key),
                ).fetchone()
            ):
                raise RunError(
                    f"evaluator {evaluator_key!r} is not one of run {run_id}'s evaluators"
                )
            self._conn.execute(
                "INSERT INTO reviews (id, case_result_id, evaluator_key, sample, reviewer, score, "
                "verdict, comment, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    review_id,
                    row["id"],
                    evaluator_key,
                    sample,
                    reviewer,
                    score,
                    verdict,
                    comment,
                    _now(),
                ),
            )
        return review_id

    def case_reviews(self, run_id: str, evaluator_key: str | None = None) -> list[dict[str, Any]]:
        """Reviews of the run's case results (optionally those grading one evaluator)."""
        sql = (
            "SELECT c.case_key, v.evaluator_key, v.reviewer, v.verdict, v.score, v.comment, "
            "v.sample, v.created_at, v.id FROM reviews v "
            "JOIN case_results r ON r.id = v.case_result_id "
            "JOIN cases c ON c.id = r.case_id WHERE r.run_id = ?"
        )
        params: list[Any] = [run_id]
        if evaluator_key is not None:
            sql += " AND v.evaluator_key = ?"
            params.append(evaluator_key)
        sql += " ORDER BY c.case_key, v.created_at, v.rowid"
        keys = (
            "case_key",
            "evaluator_key",
            "reviewer",
            "verdict",
            "score",
            "comment",
            "sample",
            "created_at",
            "id",
        )
        with self._lock:
            return [dict(zip(keys, r, strict=True)) for r in self._conn.execute(sql, params)]

    # -- judge checks -------------------------------------------------------------------------

    def save_judge_check(self, row: dict[str, Any]) -> str:
        check_id = str(uuid.uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO judge_checks (id, evaluator_key, fixture_name, fixture_hash, n, "
                "correct, adversarial_n, adversarial_correct, failed, results_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    check_id,
                    row["evaluator_key"],
                    row["fixture_name"],
                    row["fixture_hash"],
                    row["n"],
                    row["correct"],
                    row["adversarial_n"],
                    row["adversarial_correct"],
                    row["failed"],
                    json.dumps(row["results"], allow_nan=False),
                    _now(),
                ),
            )
        return check_id

    def latest_judge_check(self, evaluator_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, fixture_name, fixture_hash, n, correct, adversarial_n, "
                "adversarial_correct, failed, results_json, created_at FROM judge_checks "
                "WHERE evaluator_key = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (evaluator_key,),
            ).fetchone()
        if row is None:
            return None
        keys = ("id", "fixture_name", "fixture_hash", "n", "correct", "adversarial_n",
                "adversarial_correct", "failed", "results", "created_at")  # fmt: skip
        d = dict(zip(keys, row, strict=True))
        d["results"] = json.loads(d["results"])
        return d
