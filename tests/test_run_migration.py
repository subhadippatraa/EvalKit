"""Migration 4 (runs and results): checksum-protected, transactional, backward compatible."""

import inspect
import os
import sqlite3
import stat
import threading

import pytest
from conftest import CRITERIA, EM, FakeJudge, case, complete, judged, make_run, ok
from legacy_schemas import V0_SCHEMA, V1_SCHEMA
from test_migrations import applied, columns, legacy_row, make_db

from evalkit import EvalKit, Evaluator, MigrationError
from evalkit.migrations import MIGRATIONS, RUNS_SQL, Migration, migrate
from evalkit.store import SQLiteStore

ALL = [m.version for m in MIGRATIONS]
NEW_TABLES = {"runs", "run_evaluators", "case_results", "evaluator_results", "metrics", "attempts"}


def names(path, kind):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type=?", (kind,))}
    finally:
        conn.close()


def upto(path, n):
    """A database as an older evalkit left it: migrations 1..n applied."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:n])
    return conn


def test_released_migrations_are_never_edited_and_this_one_is_pinned():
    assert [m.checksum for m in MIGRATIONS] == [
        "704418f12ddbdf1be87617438514db423a948abd967f3853b03c9ee28b489896",  # 1 baseline
        "80a7ff182a5a0bf61f109dabdcb2501f40a64cd095ed27081c5970534d84e9ba",  # 2 p0_hardening
        "7deb3512a5db0e0ea827ee131deb02a4df73244b0ca00edc15e2009f53495bba",  # 3 datasets
        "75809807771611ba08018eb50753cfdddfba91e4411c0af742e6dba5026bb613",  # 4 runs_and_results
        "55838dc14c3feac064f5406dff88e922cf17cf153edb7ce05d5c254d162eb5ed",  # 5 analysis
        "4017ccad2462e3df52d14970d901ae46117f609d54cd56ac20358a0b2d13d995",  # 6 p1_1_hardening
        "eea0a364ee2ff18dc0460a43ef4dde352df1ce586671c0d3535659a91bab6180",  # 7 p2_1_operations
    ]
    assert [m.version for m in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))
    assert MIGRATIONS[3].name == "runs_and_results"


def test_a_fresh_database_gets_the_run_schema(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    assert applied(path) == ALL
    assert NEW_TABLES <= {
        r[0]
        for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert columns(path, "runs") == {
        "id", "dataset_version_id", "name", "status", "stop_reason", "config_json",
        "identity_hash", "exec_hash", "evaluator_count", "environment_json", "idempotency_key",
        "created_at", "started_at", "finished_at",
    }  # fmt: skip
    assert columns(path, "case_results") == {
        "id", "run_id", "case_id", "status", "output", "retrieved_json", "failure_class",
        "failure_kind", "failure_message", "retryable", "created_at", "started_at", "finished_at",
        "duration_ms", "ord", "meta_json", "retry_round",
    }  # fmt: skip
    assert columns(path, "evaluator_results") == {
        "id", "case_result_id", "run_id", "evaluator_key", "status", "verdict", "detail_json",
        "failure_class", "failure_kind", "failure_message", "retryable", "created_at",
        "duration_ms", "retry_round",
    }  # fmt: skip
    assert columns(path, "metrics") == {
        "evaluator_result_id",
        "run_id",
        "evaluator_key",
        "name",
        "value",
    }
    assert {"id", "case_result_id", "evaluator_result_id", "n", "raw_payload"} <= columns(
        path, "attempts"
    )


def test_the_expected_triggers_and_indexes_exist(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    triggers = names(path, "trigger")
    for expected in (
        "runs_insert", "runs_config_immutable", "runs_legal_transition",
        "runs_succeeded_needs_every_case", "runs_no_delete", "run_evaluators_insert",
        "case_results_same_dataset_version", "case_results_run_accepts_insert",
        "case_results_write_once", "case_results_no_delete", "evaluator_results_run_matches",
        "evaluator_results_fit_case_result", "evaluator_results_no_update",
        "metrics_need_ok_result",
        "metrics_no_update", "attempts_owner_running", "attempts_class_fits_owner",
        "attempts_no_update", "attempts_no_delete",
    ):  # fmt: skip
        assert expected in triggers, expected
    assert {
        "idx_case_results_claim", "idx_case_results_failures", "idx_evaluator_results_key",
        "idx_evaluator_results_failures", "idx_metrics_aggregate", "ux_attempts_case_result",
        "ux_attempts_evaluator_result", "idx_runs_dataset_version", "idx_runs_identity",
    } <= names(path, "index")  # fmt: skip


def test_the_hot_queries_use_their_indexes(tmp_path):
    """Guard the access patterns of design 11.3 against an accidental full scan."""
    path = tmp_path / "new.db"
    conn = SQLiteStore(path)._conn

    def plan(sql):
        return " ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))

    assert "idx_case_results_claim" in plan(
        "SELECT * FROM case_results WHERE run_id = 'r' AND status = 'pending'"
    )
    assert "idx_metrics_aggregate" in plan(
        "SELECT AVG(value) FROM metrics WHERE run_id = 'r' AND evaluator_key = 'k' AND name = 's'"
    )
    assert "idx_case_results_failures" in plan(
        "SELECT * FROM case_results WHERE run_id = 'r' AND status = 'failed' "
        "AND failure_class = 'target'"
    )
    assert "idx_evaluator_results_failures" in plan(
        "SELECT * FROM evaluator_results WHERE run_id = 'r' AND status = 'failed' "
        "AND failure_class = 'infrastructure'"
    )
    assert "USING INDEX" in plan("SELECT * FROM case_results WHERE run_id = 'r' AND case_id = 'c'")
    assert "USING INDEX" in plan("SELECT * FROM attempts WHERE case_result_id = 'c'")
    assert "USING INDEX" in plan("SELECT * FROM evaluator_results WHERE case_result_id = 'c'")


def test_reopening_is_idempotent_and_makes_no_backup(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    for _ in range(3):
        SQLiteStore(path).close()
    assert applied(path) == ALL and list(tmp_path.glob("*.bak-*")) == []


# --- upgrades --------------------------------------------------------------------------------


def seed_dataset_at_migration_3(path, name, cases):
    """A database exactly as the dataset-foundation release left it (migrations 1-3), holding one
    sealed dataset version, written with the SQL of that release (no `input_hash`)."""
    import json
    import uuid

    from evalkit.datasets import EvaluationCase
    from evalkit.hashing import dataset_hash

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:3])
    dataset_id, version_id = str(uuid.uuid4()), str(uuid.uuid4())
    conn.execute(
        "INSERT INTO datasets VALUES (?, ?, NULL, '2026-01-01T00:00:00+00:00')", (dataset_id, name)
    )
    conn.execute(
        "INSERT INTO dataset_versions (id, dataset_id, version_no, created_at) "
        "VALUES (?,?,1,'2026-01-01T00:00:00+00:00')",
        (version_id, dataset_id),
    )
    pairs = []
    for raw in sorted(cases, key=lambda c: c["case_key"]):
        c = EvaluationCase(**raw)
        conn.execute(
            "INSERT INTO cases (id, dataset_version_id, case_key, content_hash, prompt, output, "
            "reference, context, retrieved_json, relevance_json, metadata_json, tags_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), version_id, c.case_key, c.content_hash, c.prompt, c.output,
             c.reference, c.context, json.dumps(c.retrieved) if c.retrieved is not None else None,
                json.dumps(c.relevance) if c.relevance is not None else None,
                json.dumps(c.metadata), json.dumps(c.tags)),
        )  # fmt: skip
        pairs.append((c.case_key, c.content_hash))
    content_hash = dataset_hash(pairs)
    conn.execute(
        "UPDATE dataset_versions SET content_hash = ?, case_count = ?, sealed = 1 WHERE id = ?",
        (content_hash, len(pairs), version_id),
    )
    conn.commit()
    conn.close()
    return content_hash


def test_a_dataset_foundation_database_upgrades_with_its_datasets_intact(tmp_path):
    """The database exactly as the previous release left it: real datasets, no runs tables."""
    path = tmp_path / "p1a.db"
    content_hash = seed_dataset_at_migration_3(
        path, "qa", [case("a", output="x"), case("b", output="y")]
    )
    assert applied(path) == [1, 2, 3]
    assert not NEW_TABLES & {
        r[0]
        for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    store = SQLiteStore(path)  # opening with the current release upgrades it
    assert applied(path) == ALL
    kit = EvalKit(store)
    assert kit.datasets.verify("qa").ok  # hashes still verify: nothing about datasets moved
    assert kit.datasets.resolve("qa").content_hash == content_hash
    assert len(list(tmp_path.glob("p1a.db.bak-v3-*"))) == 1  # backed up before the change
    run = make_run(kit, EM, start=True)  # runs work on the pre-existing dataset
    kit.runs.record_evaluator_result(complete(kit, run, "a").id, ok(EM.key))
    assert kit.runs.verify(run.id).ok


@pytest.mark.parametrize("schema", [V0_SCHEMA, V1_SCHEMA], ids=["v0-legacy", "v1-0.1.0"])
def test_legacy_databases_upgrade_through_every_migration_to_runs(tmp_path, schema):
    path = tmp_path / "legacy.db"
    make_db(path, schema, [legacy_row(schema is V1_SCHEMA)])
    store = SQLiteStore(path)
    assert applied(path) == ALL
    assert store.get("old-1").prompt == "old prompt"
    kit = EvalKit(store)
    kit.datasets.import_cases("qa", [case("a", output="x")])
    run = kit.runs.create("qa", {"evaluators": [{"kind": "exact_match", "name": "em"}]})
    assert kit.runs.start(run.id).status == "running"


def test_a_p0_database_upgrades_and_both_worlds_coexist(tmp_path):
    path = tmp_path / "p0.db"
    conn = upto(path, 2)
    sql, params = legacy_row(True, id="p0-row")
    conn.execute(sql, params)
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    assert applied(path) == ALL
    kit = EvalKit(store)
    new = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store).evaluate(
        "p", "o", criteria=CRITERIA
    )
    kit.datasets.import_cases("qa", [case("a", output="x")])
    make_run(kit, EM, start=True)
    assert {r.id for r in store.list()} == {"p0-row", new.id}
    assert store._conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 2
    assert store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_evaluation_rows_and_reviews_are_untouched_by_runs(tmp_path):
    path = tmp_path / "t.db"
    store = SQLiteStore(path)
    ev = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store)
    result = ev.evaluate("p", "o", criteria=CRITERIA)
    review = ev.review(result.id, "alice", "PASS")
    before = [tuple(r) for r in store._conn.execute("SELECT * FROM evaluations")]
    kit = EvalKit(store)
    run = make_run(kit, EM, start=True)
    complete(kit, run, "a")
    assert [tuple(r) for r in store._conn.execute("SELECT * FROM evaluations")] == before
    assert ev.get(result.id).reviews == [review]


# --- failure, concurrency, guards ------------------------------------------------------------


def test_a_failing_run_migration_rolls_back_completely(tmp_path):
    path = tmp_path / "t.db"
    conn = upto(path, 3)
    conn.execute(
        "INSERT INTO datasets VALUES ('d','qa',NULL,'t')"
    )  # existing data must survive too
    conn.commit()
    bad = Migration(4, "runs_and_results", RUNS_SQL + "\nSELECT nope FROM nope;")
    with pytest.raises(MigrationError, match="failed"):
        migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS[:3], bad))
    conn.close()
    assert applied(path) == [1, 2, 3]
    assert not NEW_TABLES & {
        r[0]
        for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not {
        n for n in names(path, "trigger") if n.startswith(("runs_", "case_results_", "attempts_"))
    }
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1
    SQLiteStore(path).close()  # the real migration then succeeds on the same file
    assert applied(path) == ALL


def test_concurrent_first_openers_of_a_previous_release_database_apply_it_once(tmp_path):
    path = tmp_path / "shared.db"
    conn = upto(path, 3)
    conn.close()
    errors, stores = [], []

    def open_it():
        try:
            stores.append(SQLiteStore(path, backup=False))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=open_it) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and applied(path) == ALL
    for s in stores:
        s.close()


def test_a_database_from_a_newer_evalkit_is_refused_by_this_one(tmp_path):
    path = tmp_path / "n.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_migrations VALUES (?, 'future', 'x', 't')", (ALL[-1] + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(MigrationError, match="newer than this evalkit"):
        SQLiteStore(path)


def test_the_previous_release_refuses_a_database_that_has_runs(tmp_path):
    """Downgrade guard: code that only knows migrations 1-3 will not open a version-4 database."""
    path = tmp_path / "t.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    with pytest.raises(MigrationError, match="newer than this evalkit supports"):
        migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:3])
    conn.close()


def test_editing_migration_4_after_it_was_applied_is_detected(tmp_path):
    path = tmp_path / "t.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    edited = (
        *MIGRATIONS[:3],
        Migration(4, "runs_and_results", RUNS_SQL + "\n-- edited"),
        *MIGRATIONS[4:],
    )
    with pytest.raises(MigrationError, match="was modified after it was applied"):
        migrate(conn, path=path, backup=False, migrations=edited)
    conn.close()


def test_p0_guarantees_wal_permissions_and_api_survive(tmp_path):
    path = tmp_path / "new.db"
    store = SQLiteStore(path)
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(inspect.signature(Evaluator.evaluate).parameters) == [
        "self", "prompt", "model_output", "reference_output", "context", "criteria", "rubric",
        "metadata", "tags",
    ]  # fmt: skip
