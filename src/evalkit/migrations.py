"""Versioned, forward-only SQLite migrations.

Every database -- new, pre-migration ("legacy"), or already migrated -- goes through `migrate()`:

* Applied versions are recorded in `schema_migrations` with a checksum of the migration text.
* Each migration runs in its own `BEGIN IMMEDIATE` transaction (SQLite DDL is transactional), so a
  failure leaves the database exactly as it was, and concurrent openers serialize on the write
  lock and re-check what is already applied (idempotent).
* A database with an `evaluations` table but no `schema_migrations` is a legacy database. Its
  shape is verified (v0 = before `context` existed, v1 = the 0.1.0 shape) and adopted; anything
  else is refused untouched, never guessed at.
* A database newer than this code, or with a modified applied migration, is refused.
* Before changing a non-empty file database, a consistent copy is written next to it.
"""

import hashlib
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evalkit.errors import MigrationError

log = logging.getLogger("evalkit.migrations")

BASELINE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
  id                   TEXT PRIMARY KEY,
  created_at           TEXT NOT NULL,
  status               TEXT NOT NULL CHECK (status IN ('ok','error')),
  error                TEXT,
  prompt               TEXT NOT NULL,
  model_output         TEXT NOT NULL,
  reference_output     TEXT,
  context              TEXT,
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

# P0 hardening. The trigger is the DB-level half of the score/verdict coherence invariant: SQLite
# stores NaN as NULL (which is how a NaN score once became `status=ok, overall_score=NULL`), so
# an "ok" row must carry a real score in [0, 1] and a verdict, and an "error" row neither.
# A trigger rather than a CHECK because a CHECK cannot be added to an existing table without a
# rebuild. INSERT only: rows are immutable and old rows are never re-validated.
P0_SQL = """
ALTER TABLE evaluations ADD COLUMN attempts_json TEXT;

CREATE TABLE rubric_versions (
  version       TEXT PRIMARY KEY,
  content_hash  TEXT NOT NULL,
  first_seen_at TEXT NOT NULL
);

CREATE TRIGGER evaluations_coherent_insert BEFORE INSERT ON evaluations
WHEN NOT (
  (NEW.status = 'ok'
     AND NEW.error IS NULL
     AND NEW.overall_score IS NOT NULL AND NEW.overall_score >= 0 AND NEW.overall_score <= 1
     AND NEW.verdict IS NOT NULL
     AND NEW.scores_json IS NOT NULL)
  OR
  (NEW.status = 'error'
     AND NEW.error IS NOT NULL
     AND NEW.overall_score IS NULL
     AND NEW.verdict IS NULL)
)
BEGIN
  SELECT RAISE(ABORT, 'evaluations: status/score/verdict are incoherent');
END;
"""


def _backfill_rubric_versions(conn: sqlite3.Connection) -> None:
    """Register the rubric version -> content hash of every existing row (earliest first wins).

    A legacy database may already contain one version label used for different rubric
    content (the very problem the registry prevents). Those rows are left untouched and
    logged; the later variant will be refused when next used, with instructions.
    """
    from evalkit.models import Rubric  # local: models must not be needed to import this module

    rows = conn.execute(
        "SELECT rubric_version, rubric_json, MIN(created_at) FROM evaluations "
        "GROUP BY rubric_version, rubric_json ORDER BY MIN(created_at), MIN(rowid)"
    ).fetchall()
    conflicts = unreadable = 0
    for version, rubric_json, first_seen in rows:
        try:
            content_hash = Rubric.model_validate_json(rubric_json).content_hash
        except ValueError:
            unreadable += 1
            continue
        conn.execute(
            "INSERT OR IGNORE INTO rubric_versions (version, content_hash, first_seen_at) "
            "VALUES (?, ?, ?)",
            (version, content_hash, first_seen),
        )
        (stored,) = conn.execute(
            "SELECT content_hash FROM rubric_versions WHERE version = ?", (version,)
        ).fetchone()
        conflicts += stored != content_hash
    if conflicts or unreadable:
        log.warning(
            "rubric version backfill: %d version label(s) already used for different rubric "
            "content, %d stored rubric(s) unreadable; existing rows were left unchanged",
            conflicts,
            unreadable,
        )


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    hook: Callable[[sqlite3.Connection], None] | None = None

    @property
    def checksum(self) -> str:
        text = self.sql + (self.hook.__name__ if self.hook else "")
        return hashlib.sha256(text.encode()).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", BASELINE_SQL),
    Migration(2, "p0_hardening", P0_SQL, _backfill_rubric_versions),
)

_V1_EVALUATION_COLUMNS = frozenset(
    "id created_at status error prompt model_output reference_output context rubric_json "
    "rubric_version judge_provider judge_model judge_temperature judge_prompt_version "
    "scores_json overall_score verdict latency_ms metadata_json tags_json".split()
)
_V0_EVALUATION_COLUMNS = _V1_EVALUATION_COLUMNS - {"context"}
_REVIEW_COLUMNS = frozenset("id evaluation_id reviewer score verdict comment created_at".split())


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _statements(sql: str) -> Iterator[str]:
    """Split a script into statements; complete_statement() understands trigger bodies."""
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                yield buffer
            buffer = ""
    if buffer.strip():
        raise MigrationError(f"incomplete SQL statement in migration: {buffer.strip()[:80]!r}")


def _detect_legacy(conn: sqlite3.Connection) -> str:
    """'v0' (no `context` column) or 'v1'; MigrationError for anything unrecognized."""
    evaluations = _columns(conn, "evaluations")
    if evaluations == _V1_EVALUATION_COLUMNS:
        shape = "v1"
    elif evaluations == _V0_EVALUATION_COLUMNS:
        shape = "v0"
    else:
        raise MigrationError(
            "unrecognized 'evaluations' table (columns: "
            f"{sorted(evaluations)}): not an evalkit database, or an unsupported version. "
            "Refusing to modify it."
        )
    if not _table_exists(conn, "reviews") or _columns(conn, "reviews") != _REVIEW_COLUMNS:
        raise MigrationError(
            "unrecognized or missing 'reviews' table next to a valid 'evaluations' table. "
            "Refusing to modify the database."
        )
    return shape


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, "
    "applied_at TEXT NOT NULL)"
)


def _applied(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    if not _table_exists(conn, "schema_migrations"):
        return {}
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _record(conn: sqlite3.Connection, m: Migration) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
        (m.version, m.name, m.checksum, datetime.now(UTC).isoformat()),
    )


def _backup(conn: sqlite3.Connection, path: Path, from_version: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = path.with_name(f"{path.name}.bak-v{from_version}-{stamp}")
    target = sqlite3.connect(dest)
    try:
        conn.backup(target)
    finally:
        target.close()
    os.chmod(dest, 0o600)
    log.info("backed up %s to %s before migrating", path, dest)
    return dest


def migrate(
    conn: sqlite3.Connection,
    *,
    path: Path | None = None,
    backup: bool = True,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> None:
    """Bring `conn`'s database to the latest schema. Safe to call repeatedly and concurrently."""
    latest = migrations[-1].version
    by_version = {m.version: m for m in migrations}
    previous_isolation = conn.isolation_level
    conn.isolation_level = None  # explicit BEGIN/COMMIT below
    try:
        applied = _applied(conn)
        if applied and max(applied) > latest:
            raise MigrationError(
                f"database schema version {max(applied)} is newer than this evalkit supports "
                f"({latest}); upgrade evalkit"
            )
        for version, (name, checksum) in applied.items():
            if version in by_version and by_version[version].checksum != checksum:
                raise MigrationError(
                    f"migration {version} ({name}) was modified after it was applied to this "
                    "database; refusing to continue"
                )

        legacy_shape = None
        if not applied and _table_exists(conn, "evaluations"):
            legacy_shape = _detect_legacy(conn)

        pending = [m for m in migrations if m.version not in applied]
        adopt = pending[0] if legacy_shape is not None else None  # the baseline, already present
        if adopt is not None:
            pending = pending[1:]
        if not pending and adopt is None:
            return

        if backup and path is not None and (applied or legacy_shape is not None):
            _backup(conn, path, max(applied, default=0))

        if adopt is not None:
            with _transaction(conn):
                conn.execute(_CREATE_TABLE)
                if not conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (adopt.version,)
                ).fetchone():
                    if legacy_shape == "v0":
                        conn.execute("ALTER TABLE evaluations ADD COLUMN context TEXT")
                    _record(conn, adopt)
        for m in pending:
            with _transaction(conn):
                conn.execute(_CREATE_TABLE)
                # another process may have applied it while we waited for the write lock
                if conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (m.version,)
                ).fetchone():
                    continue
                try:
                    for statement in _statements(m.sql):
                        conn.execute(statement)
                    if m.hook is not None:
                        m.hook(conn)
                except sqlite3.Error as e:
                    raise MigrationError(f"migration {m.version} ({m.name}) failed: {e}") from e
                _record(conn, m)
    finally:
        conn.isolation_level = previous_isolation
