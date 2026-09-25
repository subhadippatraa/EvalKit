"""Migration 3 (dataset foundation) and backward compatibility with everything before it."""

import inspect
import logging
import os
import sqlite3
import stat

import pytest
from conftest import CRITERIA, FakeJudge, case, judged
from legacy_schemas import V0_SCHEMA, V1_SCHEMA
from test_migrations import applied, columns, legacy_row, make_db

from evalkit import EvalKit, Evaluator, MigrationError, Rubric
from evalkit.migrations import DATASETS_SQL, MIGRATIONS, Migration, migrate
from evalkit.store import SQLiteStore

ALL = [m.version for m in MIGRATIONS]


def tables(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def names(path, kind):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type=?", (kind,))}
    finally:
        conn.close()


def test_a_released_migration_is_never_edited():
    """Checksums of released migrations are recorded in users' databases; editing one would make
    every existing database refuse to open. New schema changes are NEW migrations."""
    assert [m.checksum for m in MIGRATIONS[:3]] == [
        "704418f12ddbdf1be87617438514db423a948abd967f3853b03c9ee28b489896",  # 1 baseline
        "80a7ff182a5a0bf61f109dabdcb2501f40a64cd095ed27081c5970534d84e9ba",  # 2 p0_hardening
        "7deb3512a5db0e0ea827ee131deb02a4df73244b0ca00edc15e2009f53495bba",  # 3 dataset_foundation
    ]
    assert [m.version for m in MIGRATIONS] == list(range(1, len(MIGRATIONS) + 1))


def test_fresh_database_gets_the_dataset_schema(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    assert applied(path) == ALL
    assert {"datasets", "dataset_versions", "cases"} <= tables(path)
    assert columns(path, "datasets") == {"id", "name", "description", "created_at"}
    assert columns(path, "dataset_versions") == {
        "id", "dataset_id", "version_no", "content_hash", "case_count", "source", "created_at",
        "sealed",
    }  # fmt: skip
    assert columns(path, "cases") == {
        "id", "dataset_version_id", "case_key", "content_hash", "prompt", "output", "reference",
        "context", "retrieved_json", "relevance_json", "metadata_json", "tags_json",
    }  # fmt: skip
    assert {
        "cases_no_update", "cases_no_delete", "cases_insert_only_unsealed",
        "dataset_versions_sealed_no_update", "dataset_versions_sealed_no_delete",
    } <= names(path, "trigger")  # fmt: skip
    assert "ux_dataset_versions_hash" in names(path, "index")


def test_reopening_is_idempotent(tmp_path):
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
    assert list(tmp_path.glob("*.bak-*")) == []


def test_a_p0_database_is_upgraded_and_its_data_is_untouched(tmp_path):
    """A database as P0 (0.2.0) left it: migrations 1-2 applied, real evaluation data in it."""
    path = tmp_path / "p0.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:2])
    sql, params = legacy_row(True, id="p0-row")
    conn.execute(sql, params)
    conn.commit()
    conn.close()
    assert applied(path) == [1, 2] and "datasets" not in tables(path)

    store = SQLiteStore(path)  # opening upgrades to the latest schema
    assert applied(path) == ALL and "cases" in tables(path)
    old = store.get("p0-row")
    assert (old.prompt, old.context, old.tags, old.attempts) == (
        "old prompt",
        "old context",
        ["old"],
        [],
    )
    assert len(list(path.parent.glob("p0.db.bak-v2-*"))) == 1  # backed up before the change

    kit = EvalKit(store)
    assert kit.datasets.import_cases("qa", [case("a")]).created
    new = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store).evaluate(
        "p", "o", criteria=CRITERIA
    )
    assert {r.id for r in store.list()} == {"p0-row", new.id}  # both worlds coexist in one file


@pytest.mark.parametrize("schema", [V0_SCHEMA, V1_SCHEMA], ids=["v0-legacy", "v1-0.1.0"])
def test_legacy_databases_upgrade_through_every_migration_to_datasets(tmp_path, schema):
    path = tmp_path / "legacy.db"
    make_db(path, schema, [legacy_row(schema is V1_SCHEMA)])
    store = SQLiteStore(path)
    assert applied(path) == ALL
    assert store.get("old-1").prompt == "old prompt"  # legacy row survives
    kit = EvalKit(store)
    v = kit.datasets.import_cases("qa", [case("a")]).version
    assert v.version_no == 1 and kit.datasets.verify("qa").ok


def test_a_failing_dataset_migration_rolls_back_completely(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:2])
    bad = Migration(3, "dataset_foundation", DATASETS_SQL + "\nSELECT nope FROM nope;")
    with pytest.raises(MigrationError, match="failed"):
        migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS[:2], bad))
    conn.close()
    assert applied(path) == [1, 2]
    assert not ({"datasets", "dataset_versions", "cases"} & tables(path))  # not even half-created
    assert not {n for n in names(path, "trigger") if n.startswith(("cases_", "dataset_versions_"))}
    SQLiteStore(path).close()  # and the real migration then succeeds
    assert applied(path) == ALL


def test_a_newer_database_is_still_refused(tmp_path):
    path = tmp_path / "n.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_migrations VALUES (?, 'future', 'x', 't')", (ALL[-1] + 1,))
    conn.commit()
    conn.close()
    with pytest.raises(MigrationError, match="newer than this evalkit"):
        SQLiteStore(path)


# --- P0 guarantees still hold with datasets in the mix ---------------------------------------


def test_wal_and_owner_only_permissions_are_preserved(tmp_path):
    path = tmp_path / "new.db"
    store = SQLiteStore(path)
    EvalKit(store).datasets.import_cases("qa", [case("a")])
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_public_evaluator_api_is_unchanged(store):
    params = list(inspect.signature(Evaluator.evaluate).parameters)
    assert params == [
        "self", "prompt", "model_output", "reference_output", "context", "criteria", "rubric",
        "metadata", "tags",
    ]  # fmt: skip
    ev = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store)
    result = ev.evaluate("p", "o", criteria=CRITERIA, tags=["t"])
    review = ev.review(result.id, "alice", "PASS")
    assert ev.get(result.id).reviews == [review] and [r.id for r in ev.list(tag="t")] == [result.id]
    rows, cursor = ev.list_page(limit=5)
    assert len(rows) == 1 and cursor is None


def test_the_p0_top_level_imports_still_work():
    import evalkit

    for name in (
        "Evaluator", "Rubric", "Criterion", "EvaluationResult", "Review", "Judge", "Attempt",
        "Limits", "EvalKitError", "ConfigError", "RubricError", "JudgeError", "JudgeOutputError",
        "JudgeTimeoutError", "ScoringError", "StoreError", "MigrationError",
    ):  # fmt: skip
        assert hasattr(evalkit, name), name
    assert Rubric.from_dict({"a": "A?"}).version == "89e795732810"  # auto versions unchanged


def test_dataset_import_failure_evidence_is_structured_and_persists_nothing(kit, store, caplog):
    """The dataset analogue of P0's attempt evidence: every problem, with where it is."""
    from evalkit import DatasetError

    with caplog.at_level(logging.WARNING), pytest.raises(DatasetError) as info:
        kit.datasets.import_cases("qa", [case("a"), {"case_key": "b"}, case("a")])
    assert [(i.line, i.case_key) for i in info.value.issues] == [(2, "b"), (None, "a")]
    assert store._conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
