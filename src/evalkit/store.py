from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from evalkit.analysis_store import AnalysisStoreMixin
from evalkit.dataset_store import DatasetStoreMixin
from evalkit.errors import EvalKitError, RubricError
from evalkit.limits import MAX_LIST_LIMIT
from evalkit.migrations import migrate
from evalkit.models import EvaluationResult, Review, format_validation_error
from evalkit.run_store import RunStoreMixin

log = logging.getLogger("evalkit.store")

BUSY_TIMEOUT_S = 10.0
DEFAULT_DB_PATH = "./evalkit.db"
# stay under SQLite's bound-variable cap (999 on old builds) when loading reviews for a page
_IN_CHUNK = 500


class Store(Protocol):
    """Persistence contract. Optional capabilities the Evaluator uses when present:
    `register_rubric(version, content_hash)`, `spill_result(result)`, `recover_spilled()`,
    `list_page(tag, limit, cursor)`."""

    def save(self, result: EvaluationResult) -> None: ...
    def get(self, evaluation_id: str) -> EvaluationResult | None: ...
    def list(self, tag: str | None = None, limit: int = 20) -> list[EvaluationResult]: ...
    def add_review(self, review: Review) -> None: ...


@dataclass
class Recovery:
    recovered: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # spill file name -> reason


def default_db_path() -> str:
    return os.environ.get("EVALKIT_DB_PATH") or DEFAULT_DB_PATH


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL. Changing the journal mode needs an exclusive lock that SQLite's busy timeout
    does not cover, so concurrent first-openers of a new file would fail with "database is
    locked": skip if already WAL, otherwise retry until the busy-timeout budget is spent."""
    deadline = time.monotonic() + BUSY_TIMEOUT_S
    while conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _seeded_order(seed: int, case_key: str) -> int:
    """Processing order of a case: the first 8 bytes of sha256(seed || case_key) as a non-negative
    63-bit integer. Deterministic and uniform; a sample of N is a prefix of a sample of M."""
    digest = hashlib.sha256(f"{seed}\x00{case_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _check_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise EvalKitError(
            f"limit must be an integer between 1 and {MAX_LIST_LIMIT}, got {limit!r}"
        )
    return limit


def _encode_cursor(created_at: str, rowid: int) -> str:
    raw = json.dumps([created_at, rowid]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, int]:
    try:
        created_at, rowid = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if isinstance(created_at, str) and isinstance(rowid, int) and not isinstance(rowid, bool):
            return created_at, rowid
    except (ValueError, TypeError, binascii.Error):
        pass
    raise EvalKitError(f"invalid cursor {cursor!r}")


class SQLiteStore(DatasetStoreMixin, RunStoreMixin, AnalysisStoreMixin):
    """Every call is serialized through one connection and a lock, so a store may be shared
    across threads. Multiple processes may open the same file (WAL + a busy timeout); that is
    tolerated, not tuned for (ponytail: one lock for the whole store, per-thread connections if
    this ever becomes a bottleneck).

    On open, the schema is migrated to the latest version (see evalkit.migrations); a database
    that is unrecognized or newer than this evalkit raises MigrationError. New database files are
    created owner-only (0600).
    """

    def __init__(self, path: str | Path, *, backup: bool = True):
        self.path: Path | None = None if str(path) == ":memory:" else Path(path)
        created = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            created = not self.path.exists()
        # check_same_thread=False + our own lock: Evaluator.from_env() commonly gets shared
        # across threads (e.g. a thread pool), unlike a bare sqlite3 connection.
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=BUSY_TIMEOUT_S)
        self._lock = threading.Lock()
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.create_function("evalkit_ord", 2, _seeded_order, deterministic=True)
            if self.path is not None:
                _enable_wal(self._conn)
            migrate(self._conn, path=self.path, backup=backup)
        except BaseException:
            self._conn.close()
            raise
        if created and self.path is not None:
            os.chmod(self.path, 0o600)
        if self.spill_dir is not None and any(self.spill_dir.glob("*.json")):
            log.warning(
                "unsaved evaluation results are waiting in %s; call recover_spilled()",
                self.spill_dir,
            )

    @property
    def spill_dir(self) -> Path | None:
        return None if self.path is None else self.path.with_name(f"{self.path.name}.spill")

    def close(self) -> None:
        self._conn.close()

    # -- rubric identity ------------------------------------------------------------------

    def register_rubric(self, version: str, content_hash: str, first_seen_at: str | None = None):
        """Bind a rubric version label to its content, or verify the existing binding.

        Raises RubricError if the label was already used for different content.
        """
        with self._lock, self._conn:
            self._register_rubric(version, content_hash, first_seen_at)

    def _register_rubric(self, version: str, content_hash: str, first_seen_at: str | None) -> None:
        """Must be called with self._lock held, inside a transaction."""
        when = first_seen_at or datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO rubric_versions (version, content_hash, first_seen_at) "
            "VALUES (?, ?, ?)",
            (version, content_hash, when),
        )
        (stored,) = self._conn.execute(
            "SELECT content_hash FROM rubric_versions WHERE version = ?", (version,)
        ).fetchone()
        if stored != content_hash:
            raise RubricError(
                f"rubric version {version!r} was already used for different rubric content "
                f"(stored {stored[:12]}, this rubric {content_hash[:12]}); "
                "give the changed rubric a new version, or omit `version` to use its content hash"
            )

    # -- evaluations ----------------------------------------------------------------------

    def save(self, result: EvaluationResult) -> None:
        params = {
            "id": result.id,
            "created_at": result.created_at.isoformat(),
            "status": result.status,
            "error": result.error,
            "prompt": result.prompt,
            "model_output": result.model_output,
            "reference_output": result.reference_output,
            "context": result.context,
            "rubric_json": result.rubric.model_dump_json(),
            "rubric_version": result.rubric_version,
            "judge_provider": result.judge_provider,
            "judge_model": result.judge_model,
            "judge_temperature": result.judge_temperature,
            "judge_prompt_version": result.judge_prompt_version,
            "scores_json": json.dumps({k: v.model_dump() for k, v in result.scores.items()})
            if result.scores
            else None,
            "overall_score": result.overall_score,
            "verdict": result.verdict,
            "latency_ms": result.latency_ms,
            "attempts_json": json.dumps([a.model_dump(mode="json") for a in result.attempts]),
            "metadata_json": json.dumps(result.metadata, allow_nan=False),
            "tags_json": json.dumps(result.tags),
        }
        columns = ", ".join(params)
        placeholders = ", ".join(f":{k}" for k in params)
        with self._lock, self._conn:
            # defense in depth: Evaluator already registers before the judge call
            self._register_rubric(
                result.rubric_version, result.rubric.content_hash, params["created_at"]
            )
            self._conn.execute(
                f"INSERT INTO evaluations ({columns}) VALUES ({placeholders})", params
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
        return self.list_page(tag=tag, limit=limit)[0]

    def list_page(
        self, tag: str | None = None, limit: int = 20, cursor: str | None = None
    ) -> tuple[list[EvaluationResult], str | None]:
        """Newest first. Returns (results, next_cursor); pass next_cursor back to continue.

        Keyset pagination on (created_at, rowid): stable while rows are added, and cheap at any
        depth. (rowid is not preserved by VACUUM, so a cursor should not outlive a VACUUM.)
        """
        limit = _check_limit(limit)
        where: list[str] = []
        params: list[object] = []
        if tag is not None:
            where.append("EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)")
            params.append(tag)
        if cursor is not None:
            created_at, rowid = _decode_cursor(cursor)
            where.append("(created_at, rowid) < (?, ?)")
            params += [created_at, rowid]
        sql = "SELECT rowid AS _rowid, * FROM evaluations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit + 1)  # one extra row tells us whether another page exists
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            page = rows[:limit]
            results = [_to_result(r) for r in page]
            reviews_by_eval = self._reviews_for([r.id for r in results])
        for r in results:
            r.reviews = reviews_by_eval.get(r.id, [])
        next_cursor = None
        if len(rows) > limit:
            last = page[-1]
            next_cursor = _encode_cursor(last["created_at"], last["_rowid"])
        return results, next_cursor

    def _reviews_for(self, evaluation_ids: list[str]) -> dict[str, list[Review]]:
        """Must be called with self._lock held."""
        reviews_by_eval: dict[str, list[Review]] = {}
        for i in range(0, len(evaluation_ids), _IN_CHUNK):
            chunk = evaluation_ids[i : i + _IN_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT * FROM reviews WHERE evaluation_id IN ({placeholders}) "
                "ORDER BY created_at, rowid",
                chunk,
            ).fetchall()
            for row in rows:
                reviews_by_eval.setdefault(row["evaluation_id"], []).append(
                    Review.model_validate(dict(row))
                )
        return reviews_by_eval

    def add_review(self, review: Review) -> None:
        params = {
            "id": review.id,
            "evaluation_id": review.evaluation_id,
            "reviewer": review.reviewer,
            "score": review.score,
            "verdict": review.verdict,
            "comment": review.comment,
            "created_at": review.created_at.isoformat(),
        }
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO reviews ({', '.join(params)}) "
                f"VALUES ({', '.join(f':{k}' for k in params)})",
                params,
            )

    # -- spill: a paid result must never be lost to a storage failure ---------------------

    def spill_result(self, result: EvaluationResult) -> Path:
        """Write `result` to <db>.spill/<id>.json (atomic, owner-only). Independent of the
        database connection, so it works when the database itself is what failed."""
        if self.spill_dir is None:
            raise EvalKitError("an in-memory store has no spill directory")
        self.spill_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        final = self.spill_dir / f"{_safe_name(result.id)}.json"
        tmp = final.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        return final

    def recover_spilled(self) -> Recovery:
        """Replay spilled results into the database. A file is deleted only once its result is
        safely stored (or already was); anything that cannot be recovered stays, with a reason."""
        report = Recovery()
        if self.spill_dir is None or not self.spill_dir.is_dir():
            return report
        for file in sorted(self.spill_dir.glob("*.json")):
            try:
                result = EvaluationResult.model_validate_json(file.read_text(encoding="utf-8"))
                try:
                    self.save(result)
                except sqlite3.IntegrityError:
                    if self.get(result.id) is None:
                        raise
                report.recovered.append(result.id)
                file.unlink()
            except (OSError, ValueError, EvalKitError, sqlite3.Error) as e:
                report.failed[file.name] = f"{type(e).__name__}: {e}"
        return report


def _safe_name(evaluation_id: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", evaluation_id):
        return evaluation_id
    return hashlib.sha256(evaluation_id.encode("utf-8", "replace")).hexdigest()[:32]


def _to_result(row: sqlite3.Row) -> EvaluationResult:
    d = dict(row)
    d.pop("_rowid", None)
    d["rubric"] = json.loads(d.pop("rubric_json"))
    d["scores"] = json.loads(d.pop("scores_json") or "{}")
    d["attempts"] = json.loads(d.pop("attempts_json", None) or "[]")
    d["metadata"] = json.loads(d.pop("metadata_json"))
    d["tags"] = json.loads(d.pop("tags_json"))
    try:
        return EvaluationResult.model_validate(d)
    except ValidationError as e:
        raise EvalKitError(
            f"stored evaluation {d.get('id')!r} is corrupt or incoherent: "
            f"{format_validation_error(e)}"
        ) from e
