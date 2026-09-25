"""F-3: a versioned, transactional, idempotent migration runner; legacy databases upgrade."""

import logging
import os
import sqlite3
import stat
import threading

import pytest
from conftest import CRITERIA, FakeJudge, judged
from legacy_schemas import V0_SCHEMA, V1_SCHEMA

from evalkit import (
    EvalKitError,
    Evaluator,
    MigrationError,
    Review,
    Rubric,
    RubricError,
)
from evalkit.migrations import MIGRATIONS, Migration, migrate
from evalkit.store import SQLiteStore

RUBRIC = Rubric.from_dict({"a": "A?"})
LATEST = MIGRATIONS[-1].version
ALL = [m.version for m in MIGRATIONS]


def make_db(path, schema, rows=()):
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    for sql, params in rows:
        conn.execute(sql, params)
    conn.commit()
    conn.close()


def legacy_row(schema_has_context, *, id="old-1", rubric=RUBRIC, ok=True):
    cols = [
        "id", "created_at", "status", "error", "prompt", "model_output", "reference_output",
        "rubric_json", "rubric_version", "judge_provider", "judge_model", "judge_temperature",
        "judge_prompt_version", "scores_json", "overall_score", "verdict", "latency_ms",
        "metadata_json", "tags_json",
    ]  # fmt: skip
    values = [
        id, "2026-01-01T00:00:00+00:00", "ok" if ok else "error", None if ok else "JudgeError: x",
        "old prompt", "old output", None, rubric.model_dump_json(), rubric.version,
        "bedrock", "old-model", 0.0, "oldpv",
        '{"a": {"reasoning": "fine", "score": 4, "label": null}}' if ok else None,
        0.75 if ok else None, "PASS" if ok else None, 12, '{"app": "legacy"}', '["old"]',
    ]  # fmt: skip
    if schema_has_context:
        cols.append("context")
        values.append("old context")
    marks = ", ".join("?" * len(cols))
    return f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({marks})", values


def applied(path):
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    finally:
        conn.close()


def columns(path, table="evaluations"):
    conn = sqlite3.connect(path)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


# --- fresh databases -------------------------------------------------------------------------


def test_fresh_database_is_migrated_to_latest(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    assert applied(path) == [m.version for m in MIGRATIONS]
    assert {"context", "attempts_json"} <= columns(path)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_new_database_file_is_owner_only(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_in_memory_database_is_migrated_and_never_backed_up():
    store = SQLiteStore(":memory:")
    assert store._conn.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == LATEST


def test_reopening_is_idempotent_and_does_not_back_up(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    before = conn.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
    conn.close()
    for _ in range(3):
        SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall() == before
    conn.close()
    assert list(tmp_path.glob("*.bak-*")) == []  # nothing to migrate -> nothing to back up


# --- legacy databases: the current (v1) and original (v0) schemas ----------------------------


def test_v1_database_is_upgraded_in_place_and_keeps_its_data(tmp_path):
    path = tmp_path / "v1.db"
    make_db(
        path,
        V1_SCHEMA,
        [
            legacy_row(True, id="old-ok"),
            legacy_row(True, id="old-err", ok=False),
            (
                "INSERT INTO reviews (id, evaluation_id, reviewer, score, verdict, comment,"
                " created_at) VALUES ('r1', 'old-ok', 'alice', 0.9, 'PASS', 'c',"
                " '2026-01-02T00:00:00+00:00')",
                (),
            ),
        ],
    )
    store = SQLiteStore(path)

    old = store.get("old-ok")
    assert (old.prompt, old.context, old.verdict, old.tags) == (
        "old prompt",
        "old context",
        "PASS",
        ["old"],
    )
    assert old.attempts == []  # predates attempt evidence: unknown, not "no attempts happened"
    assert [r.reviewer for r in old.reviews] == ["alice"]
    assert store.get("old-err").status == "error"
    assert applied(path) == ALL

    # the upgraded database accepts new writes, alongside the old rows
    result = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store).evaluate(
        "p", "o", criteria=CRITERIA
    )
    assert store.get(result.id) == result
    assert {r.id for r in store.list(limit=10)} == {"old-ok", "old-err", result.id}


def test_v0_database_gets_the_context_column_and_can_be_written_again(tmp_path):
    """Regression for F-3: 19-column table + 20-value positional INSERT => every write failed."""
    path = tmp_path / "v0.db"
    make_db(path, V0_SCHEMA, [legacy_row(False)])
    assert "context" not in columns(path)

    store = SQLiteStore(path)
    assert "context" in columns(path)
    assert store.get("old-1").context is None  # old row: NULL context
    result = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store).evaluate(
        "p", "o", context="new context", criteria=CRITERIA
    )
    assert store.get(result.id).context == "new context"
    assert applied(path) == ALL


@pytest.mark.parametrize("schema", [V0_SCHEMA, V1_SCHEMA])
def test_legacy_upgrade_is_idempotent(tmp_path, schema):
    path = tmp_path / "legacy.db"
    make_db(path, schema, [legacy_row(schema is V1_SCHEMA)])
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    snapshot = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    conn.close()
    for _ in range(3):
        SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    assert (
        conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall() == snapshot
    )
    conn.close()
    assert len(list(tmp_path.glob("*.bak-*"))) == 1  # only the first (real) migration backed up


def test_upgrade_writes_a_consistent_backup_of_the_original(tmp_path):
    path = tmp_path / "v0.db"
    make_db(path, V0_SCHEMA, [legacy_row(False)])
    SQLiteStore(path).close()

    (backup,) = tmp_path.glob("v0.db.bak-v0-*")
    conn = sqlite3.connect(backup)
    assert "context" not in {r[1] for r in conn.execute("PRAGMA table_info(evaluations)")}
    assert conn.execute("SELECT id FROM evaluations").fetchall() == [("old-1",)]
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='schema_migrations'").fetchone()
    conn.close()
    if os.name == "posix":
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_backup_can_be_disabled(tmp_path):
    path = tmp_path / "v1.db"
    make_db(path, V1_SCHEMA)
    SQLiteStore(path, backup=False).close()
    assert list(tmp_path.glob("*.bak-*")) == []


def test_legacy_rows_are_registered_and_the_coherence_trigger_is_installed(tmp_path):
    path = tmp_path / "v1.db"
    make_db(path, V1_SCHEMA, [legacy_row(True)])
    store = SQLiteStore(path)
    row = store._conn.execute("SELECT * FROM rubric_versions").fetchall()
    assert [(r["version"], r["content_hash"]) for r in row] == [
        (RUBRIC.version, RUBRIC.content_hash)
    ]
    with pytest.raises(sqlite3.IntegrityError, match="incoherent"):
        store._conn.execute(
            "INSERT INTO evaluations (id, created_at, status, prompt, model_output, rubric_json,"
            " rubric_version, judge_provider, judge_model, judge_temperature, judge_prompt_version)"
            " VALUES ('bad', 't', 'ok', 'p', 'o', '{}', 'v', 'p', 'm', 0, 'pv')"
        )


def test_backfill_keeps_first_seen_and_reports_conflicting_versions(tmp_path, caplog):
    """A legacy DB may already contain one version label used for two different rubrics."""
    first = Rubric(criteria=RUBRIC.criteria, version="shared")
    second = Rubric.model_validate(
        {"criteria": [{"name": "a", "description": "A completely different question?"}]}
        | {"version": "shared"}
    )
    assert first.content_hash != second.content_hash
    path = tmp_path / "v1.db"
    make_db(
        path,
        V1_SCHEMA,
        [
            legacy_row(True, id="first", rubric=first),
            legacy_row(True, id="second", rubric=second),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="evalkit.migrations"):
        store = SQLiteStore(path)
    assert "already used for different rubric content" in caplog.text
    assert store.get("second").rubric == second  # existing rows are never modified or dropped
    (row,) = store._conn.execute("SELECT content_hash FROM rubric_versions WHERE version='shared'")
    assert row[0] == first.content_hash  # earliest wins
    with pytest.raises(RubricError, match="shared"):
        store.register_rubric("shared", second.content_hash)


# --- refusing what we do not understand ------------------------------------------------------


def test_unrecognized_evaluations_table_is_refused_untouched(tmp_path):
    path = tmp_path / "other.db"
    make_db(path, "CREATE TABLE evaluations (id TEXT PRIMARY KEY, something_else TEXT);")
    with pytest.raises(MigrationError, match="unrecognized 'evaluations' table"):
        SQLiteStore(path)
    assert columns(path) == {"id", "something_else"}  # nothing modified
    conn = sqlite3.connect(path)
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='schema_migrations'").fetchone()
    conn.close()
    assert list(tmp_path.glob("*.bak-*")) == []


def test_evaluations_without_a_matching_reviews_table_is_refused(tmp_path):
    path = tmp_path / "half.db"
    make_db(path, V1_SCHEMA.split("CREATE TABLE IF NOT EXISTS reviews")[0])
    with pytest.raises(MigrationError, match="reviews"):
        SQLiteStore(path)


def test_a_database_with_only_unrelated_tables_is_treated_as_new(tmp_path):
    path = tmp_path / "other.db"
    make_db(path, "CREATE TABLE unrelated (x INTEGER); INSERT INTO unrelated VALUES (7);")
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT x FROM unrelated").fetchall() == [(7,)]
    conn.close()
    assert applied(path) == ALL


def test_a_database_newer_than_this_code_is_refused(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO schema_migrations VALUES (?, 'from_the_future', 'x', 't')", (LATEST + 1,)
    )
    conn.commit()
    conn.close()
    with pytest.raises(MigrationError, match="newer than this evalkit"):
        SQLiteStore(path)


def test_a_modified_applied_migration_is_refused(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_migrations SET checksum = 'tampered' WHERE version = 1")
    conn.commit()
    conn.close()
    with pytest.raises(MigrationError, match="modified after it was applied"):
        SQLiteStore(path)


def test_migration_error_is_an_evalkit_error():
    assert issubclass(MigrationError, EvalKitError)


# --- transactional: a failing migration changes nothing --------------------------------------


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_a_failing_migration_is_rolled_back_completely_then_can_be_retried(tmp_path):
    path = tmp_path / "t.db"
    SQLiteStore(path).close()
    bad = Migration(
        LATEST + 1,
        "bad",
        "CREATE TABLE half_done (x INTEGER); INSERT INTO no_such_table VALUES (1);",
    )
    conn = sqlite3.connect(path)
    with pytest.raises(MigrationError, match="failed"):
        migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS, bad))
    assert "half_done" not in _tables(conn)  # DDL from the failed migration was rolled back
    assert applied(path) == [m.version for m in MIGRATIONS]  # not recorded
    assert not conn.in_transaction

    good = Migration(LATEST + 1, "good", "CREATE TABLE half_done (x INTEGER);")
    migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS, good))
    assert "half_done" in _tables(conn) and applied(path)[-1] == LATEST + 1
    migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS, good))  # idempotent
    conn.close()


def test_a_failing_hook_rolls_back_the_migration_sql_too(tmp_path):
    def hook(conn):
        conn.execute("INSERT INTO half_done VALUES (1)")
        conn.execute("SELECT * FROM missing_table")

    path = tmp_path / "t.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    with pytest.raises(MigrationError):
        migrate(
            conn,
            path=path,
            backup=False,
            migrations=(
                *MIGRATIONS,
                Migration(LATEST + 1, "h", "CREATE TABLE half_done (x);", hook),
            ),
        )
    assert "half_done" not in _tables(conn)
    conn.close()


def test_a_failed_legacy_upgrade_leaves_the_original_database_intact(tmp_path, monkeypatch):
    path = tmp_path / "v0.db"
    make_db(path, V0_SCHEMA, [legacy_row(False)])
    broken = (
        MIGRATIONS[0],
        Migration(
            2,
            "p0_hardening",
            "ALTER TABLE evaluations ADD COLUMN attempts_json TEXT; SELECT nope FROM nope;",
        ),
    )
    conn = sqlite3.connect(path)
    with pytest.raises(MigrationError):
        migrate(conn, path=path, backup=False, migrations=broken)
    conn.close()
    # v0 adoption (its own transaction) may have completed; the failed step must not have
    assert "attempts_json" not in columns(path)
    assert applied(path) == [1]
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT id FROM evaluations").fetchall() == [("old-1",)]
    conn.close()
    SQLiteStore(path).close()  # and the real migrations still complete afterwards
    assert applied(path) == ALL


def test_concurrent_openers_of_a_new_database_all_succeed_and_migrate_once(tmp_path):
    path = tmp_path / "race.db"
    barrier = threading.Barrier(6)
    errors: list[BaseException] = []

    def open_store():
        try:
            barrier.wait()
            SQLiteStore(path).close()
        except BaseException as e:  # noqa: BLE001 - collecting failures across threads
            errors.append(e)

    threads = [threading.Thread(target=open_store) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert applied(path) == [m.version for m in MIGRATIONS]


def test_concurrent_openers_of_a_legacy_database_migrate_it_once(tmp_path):
    path = tmp_path / "race-v0.db"
    make_db(path, V0_SCHEMA, [legacy_row(False)])
    barrier = threading.Barrier(5)
    errors: list[BaseException] = []

    def open_store():
        try:
            barrier.wait()
            SQLiteStore(path, backup=False).close()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=open_store) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert applied(path) == ALL and "context" in columns(path)


# --- named-column writes ---------------------------------------------------------------------


def test_writes_do_not_depend_on_column_order(tmp_path):
    """v0 databases get `context` appended at the END of the table; the old positional INSERT
    put the wrong values in the wrong columns. Round-trip every field to prove named columns."""
    path = tmp_path / "v0.db"
    make_db(path, V0_SCHEMA)
    store = SQLiteStore(path)
    evaluator = Evaluator(FakeJudge(judged(correctness=3, clarity=5)), store)
    result = evaluator.evaluate(
        "the prompt", "the output", reference_output="the ref", context="the ctx",
        criteria=CRITERIA, metadata={"k": [1, 2]}, tags=["t1", "t2"],
    )  # fmt: skip
    got = store.get(result.id)
    assert got == result
    assert (got.prompt, got.model_output, got.reference_output, got.context) == (
        "the prompt", "the output", "the ref", "the ctx",
    )  # fmt: skip
    review = Review(evaluation_id=result.id, reviewer="alice", verdict="PASS", comment="ok")
    store.add_review(review)
    assert store.get(result.id).reviews == [review]


def test_an_unreadable_stored_rubric_does_not_block_the_upgrade(tmp_path, caplog):
    path = tmp_path / "v1.db"
    make_db(path, V1_SCHEMA, [legacy_row(True, id="fine")])
    conn = sqlite3.connect(path)
    conn.execute("UPDATE evaluations SET rubric_json = '{not a rubric}' WHERE id = 'fine'")
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING, logger="evalkit.migrations"):
        SQLiteStore(path).close()
    assert "1 stored rubric(s) unreadable" in caplog.text
    assert applied(path) == ALL


def test_the_statement_splitter_keeps_trigger_bodies_whole_and_rejects_truncated_sql():
    from evalkit.migrations import P0_SQL, _statements

    statements = list(_statements(P0_SQL))
    assert len(statements) == 3 and statements[2].lstrip().startswith("CREATE TRIGGER")
    assert statements[2].rstrip().endswith("END;")  # the `;` inside the body did not split it
    with pytest.raises(MigrationError, match="incomplete SQL"):
        list(_statements("SELECT 1; SELECT"))


def test_switching_to_wal_is_retried_when_the_database_is_briefly_locked(monkeypatch):
    """Concurrent first-openers hit 'database is locked' on the journal-mode switch, which the
    busy timeout does not cover; the store must retry rather than fail."""
    from evalkit import store as store_module

    class Cursor:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class Conn:
        mode, failures_left = "delete", 3

        def execute(self, sql):
            if sql == "PRAGMA journal_mode":
                return Cursor((self.mode,))
            if self.failures_left:
                self.failures_left -= 1
                raise sqlite3.OperationalError("database is locked")
            self.mode = "wal"
            return Cursor(("wal",))

    conn = Conn()
    monkeypatch.setattr(store_module.time, "sleep", lambda s: None)
    store_module._enable_wal(conn)
    assert conn.mode == "wal" and conn.failures_left == 0

    permanent = Conn()
    permanent.failures_left = 10**9
    monkeypatch.setattr(store_module, "BUSY_TIMEOUT_S", -1.0)  # budget already spent
    with pytest.raises(sqlite3.OperationalError):
        store_module._enable_wal(permanent)
