"""The input-side identity of a case (`input_hash`, migration 6): what comparison pairs on."""

import sqlite3

import pytest
from conftest import case
from test_run_migration import seed_dataset_at_migration_3

from evalkit import EvalKit
from evalkit.datasets import EvaluationCase
from evalkit.hashing import case_input_hash
from evalkit.migrations import MIGRATIONS, migrate
from evalkit.store import SQLiteStore

BASE = {"case_key": "a", "prompt": "q", "reference": "r", "context": "c", "relevance": {"d1": 1}}


def h(**over):
    return EvaluationCase(**(BASE | over)).input_hash


def test_the_input_hash_ignores_everything_that_describes_an_answer_or_an_execution():
    reference = h()
    assert h(output="whatever the system said") == reference
    assert h(retrieved=["d1", "d2"]) == reference  # what the system retrieved is its output
    assert h(metadata={"run": 7}) == reference
    assert h(tags=["slice"]) == reference
    assert h(case_key="renamed") == reference  # the key is the pairing key, not part of the hash


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "another question"},
        {"reference": "another gold answer"},
        {"reference": None},
        {"context": "other grounding"},
        {"context": None},
        {"relevance": {"d1": 2}},
        {"relevance": None},
    ],
)
def test_the_input_hash_covers_the_question_and_what_counts_as_right(change):
    assert h(**change) != h()


def test_none_and_empty_are_different_inputs():
    assert h(context="") != h(context=None)
    assert h(relevance={}) != h(relevance=None)


def test_the_hash_follows_its_documented_recipe():
    """sha256("evalkit-case-input-v1\n" + canonical JSON of the four fields): a scheme change must
    be a new domain, so this independent recomputation must never need editing."""
    import hashlib

    canonical = '{"context":"c","prompt":"q","reference":"r","relevance":{"d":1}}'
    expected = hashlib.sha256(b"evalkit-case-input-v1\n" + canonical.encode()).hexdigest()
    assert case_input_hash("q", "r", "c", {"d": 1}) == expected


def test_import_records_the_input_hash_and_verify_checks_it(kit, store):
    kit.datasets.import_cases("qa", [case("a", reference="r", output="x")])
    (stored,) = store._conn.execute("SELECT input_hash FROM cases").fetchone()
    assert stored == EvaluationCase(**case("a", reference="r", output="x")).input_hash
    assert kit.datasets.verify("qa").ok
    # tampering with the recorded hash behind the API's back is detected (triggers lifted, as a
    # raw writer with a broken schema would)
    store._conn.execute("DROP TRIGGER cases_no_update")
    store._conn.execute("UPDATE cases SET input_hash = ?", ("0" * 64,))
    store._conn.commit()
    report = kit.datasets.verify("qa")
    assert not report.ok and any("input hash" in p for p in report.problems)


def test_migration_6_backfills_existing_cases_and_keeps_them_immutable(tmp_path):
    path = tmp_path / "v3.db"
    cases = [
        case("a", reference="r", output="x", relevance={"d": 1}),
        case("b", context="c", output="y", metadata={"m": 1}, tags=["t"]),
    ]
    content_hash = seed_dataset_at_migration_3(path, "qa", cases)
    store = SQLiteStore(path)  # upgrades 3 -> latest
    kit = EvalKit(store)
    by_key = {c.case_key: c for c in kit.datasets.cases("qa")}
    for raw in cases:
        assert by_key[raw["case_key"]].stored_input_hash == EvaluationCase(**raw).input_hash
    assert kit.datasets.resolve("qa").content_hash == content_hash  # nothing else moved
    assert kit.datasets.verify("qa").ok
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):  # the trigger is back
        store._conn.execute("UPDATE cases SET prompt = 'changed'")
    store.close()


def test_migration_6_rolls_back_completely_if_the_backfill_fails(tmp_path, monkeypatch):
    path = tmp_path / "v5.db"
    seed_dataset_at_migration_3(path, "qa", [case("a", output="x")])
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:5])
    conn.close()

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr("evalkit.hashing.case_input_hash", boom)
    with pytest.raises(Exception, match="disk I/O|migration 6"):
        SQLiteStore(path, backup=False).close()
    monkeypatch.undo()
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 5
    assert "input_hash" not in {r[1] for r in conn.execute("PRAGMA table_info(cases)")}
    triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert "cases_no_update" in triggers  # the immutability trigger survived the failed attempt
    conn.close()
    SQLiteStore(path, backup=False).close()  # and the migration completes once the fault is gone
