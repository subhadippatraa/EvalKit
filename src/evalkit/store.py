from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from evalkit.models import EvaluationResult, Review

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
  id                   TEXT PRIMARY KEY,
  created_at           TEXT NOT NULL,
  status               TEXT NOT NULL CHECK (status IN ('ok','error')),
  error                TEXT,
  prompt               TEXT NOT NULL,
  model_output         TEXT NOT NULL,
  reference_output     TEXT,
  rubric_json          TEXT NOT NULL,
  rubric_version       TEXT NOT NULL,
  judge_provider       TEXT NOT NULL,
  judge_model          TEXT NOT NULL,
  judge_temperature    REAL NOT NULL,
  judge_prompt_version TEXT NOT NULL,
  scores_json          TEXT,
  overall_score        REAL,
  verdict              TEXT CHECK (verdict IN ('PASS','FAIL')),
  latency_ms           INTEGER,
  metadata_json        TEXT NOT NULL DEFAULT '{}',
  tags_json            TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);

CREATE TABLE IF NOT EXISTS reviews (
  id            TEXT PRIMARY KEY,
  evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
  reviewer      TEXT NOT NULL,
  score         REAL CHECK (score BETWEEN 0 AND 1),
  verdict       TEXT NOT NULL CHECK (verdict IN ('PASS','FAIL')),
  comment       TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_evaluation_id ON reviews(evaluation_id);
"""


class Store(Protocol):
    def save(self, result: EvaluationResult) -> None: ...
    def get(self, evaluation_id: str) -> EvaluationResult | None: ...
    def list(self, tag: str | None = None, limit: int = 20) -> list[EvaluationResult]: ...
    def add_review(self, review: Review) -> None: ...


class SQLiteStore:
    """Not safe for concurrent multi-process writers; single-process multi-thread use is
    fine — every call is serialized through one connection and a lock (ponytail: one lock
    for the whole store, a WAL + per-thread connections if this ever becomes a bottleneck).
    """

    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + our own lock: Evaluator.from_env() commonly gets shared
        # across threads (e.g. a thread pool), unlike a bare sqlite3 connection.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def save(self, result: EvaluationResult) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO evaluations VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.id,
                    result.created_at.isoformat(),
                    result.status,
                    result.error,
                    result.prompt,
                    result.model_output,
                    result.reference_output,
                    result.rubric.model_dump_json(),
                    result.rubric_version,
                    result.judge_provider,
                    result.judge_model,
                    result.judge_temperature,
                    result.judge_prompt_version,
                    json.dumps({k: v.model_dump() for k, v in result.scores.items()})
                    if result.scores
                    else None,
                    result.overall_score,
                    result.verdict,
                    result.latency_ms,
                    json.dumps(result.metadata),
                    json.dumps(result.tags),
                ),
            )

    def get(self, evaluation_id: str) -> EvaluationResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
            if row is None:
                return None
            result = _to_result(row)
            result.reviews = self._reviews_for([evaluation_id]).get(evaluation_id, [])
            return result

    def list(self, tag: str | None = None, limit: int = 20) -> list[EvaluationResult]:
        sql, params = "SELECT * FROM evaluations", []
        if tag is not None:
            sql += " WHERE EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
            params.append(tag)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            results = [_to_result(r) for r in self._conn.execute(sql, params).fetchall()]
            reviews_by_eval = self._reviews_for([r.id for r in results])
        for r in results:
            r.reviews = reviews_by_eval.get(r.id, [])
        return results

    def _reviews_for(self, evaluation_ids: list[str]) -> dict[str, list[Review]]:
        """Must be called with self._lock held."""
        if not evaluation_ids:
            return {}
        placeholders = ",".join("?" * len(evaluation_ids))
        rows = self._conn.execute(
            f"SELECT * FROM reviews WHERE evaluation_id IN ({placeholders}) "
            "ORDER BY created_at, rowid",
            evaluation_ids,
        ).fetchall()
        reviews_by_eval: dict[str, list[Review]] = {}
        for row in rows:
            reviews_by_eval.setdefault(row["evaluation_id"], []).append(
                Review.model_validate(dict(row))
            )
        return reviews_by_eval

    def add_review(self, review: Review) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    review.id,
                    review.evaluation_id,
                    review.reviewer,
                    review.score,
                    review.verdict,
                    review.comment,
                    review.created_at.isoformat(),
                ),
            )


def _to_result(row: sqlite3.Row) -> EvaluationResult:
    d = dict(row)
    d["rubric"] = json.loads(d.pop("rubric_json"))
    d["scores"] = json.loads(d.pop("scores_json") or "{}")
    d["metadata"] = json.loads(d.pop("metadata_json"))
    d["tags"] = json.loads(d.pop("tags_json"))
    return EvaluationResult.model_validate(d)
