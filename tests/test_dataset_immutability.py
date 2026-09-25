"""Sealed dataset data is immutable at the database level, independent of the Python code."""

import sqlite3

import pytest
from conftest import case

from evalkit import DatasetError

CASES = [case("a", output="1"), case("b", output="2")]


@pytest.fixture
def sealed(kit, store):
    kit.datasets.import_cases("qa", CASES)
    return store._conn


def raises(match):
    return pytest.raises(sqlite3.IntegrityError, match=match)


def test_sealed_cases_cannot_be_updated_or_deleted(sealed):
    with raises("cases are immutable"):
        sealed.execute("UPDATE cases SET prompt = 'x'")
    with raises("cases are immutable"):
        sealed.execute("UPDATE cases SET case_key = 'other' WHERE case_key = 'a'")
    with raises("cases are immutable"):
        sealed.execute("DELETE FROM cases")
    with raises("cases are immutable"):
        sealed.execute("DELETE FROM cases WHERE case_key = 'a'")
    assert sealed.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 2


def test_cases_cannot_be_added_to_a_sealed_version(sealed):
    (vid,) = sealed.execute("SELECT id FROM dataset_versions").fetchone()
    with raises("sealed"):
        sealed.execute(
            "INSERT INTO cases (id, dataset_version_id, case_key, content_hash, prompt) "
            "VALUES ('x', ?, 'late', ?, 'p')",
            (vid, "0" * 64),
        )


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE dataset_versions SET content_hash = ?",
        "UPDATE dataset_versions SET case_count = 99",
        "UPDATE dataset_versions SET sealed = 0",
        "UPDATE dataset_versions SET version_no = 7",
        "DELETE FROM dataset_versions",
    ],
)
def test_a_sealed_version_cannot_be_modified_unsealed_or_deleted(sealed, sql):
    params = ("f" * 64,) if "?" in sql else ()
    with raises("immutable once sealed"):
        sealed.execute(sql, params)


def test_constraints_reject_malformed_rows(sealed):
    (did,) = sealed.execute("SELECT id FROM datasets").fetchone()
    (vid,) = sealed.execute("SELECT id FROM dataset_versions").fetchone()
    with pytest.raises(sqlite3.IntegrityError):  # sealed without a hash/count
        sealed.execute(
            "INSERT INTO dataset_versions (id, dataset_id, version_no, created_at, sealed) "
            "VALUES ('v', ?, 9, 't', 1)",
            (did,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):  # version numbers are unique
        sealed.execute(
            "INSERT INTO dataset_versions (id, dataset_id, version_no, created_at) "
            "VALUES ('v', ?, 1, 't')",
            (did,),
        )
    with pytest.raises(sqlite3.IntegrityError):  # version_no >= 1
        sealed.execute(
            "INSERT INTO dataset_versions (id, dataset_id, version_no, created_at) "
            "VALUES ('v', ?, 0, 't')",
            (did,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):  # dataset names are unique
        sealed.execute("INSERT INTO datasets (id, name, created_at) VALUES ('d', 'qa', 't')")
    with pytest.raises(sqlite3.IntegrityError):  # case content hash must be a real hash
        sealed.execute(
            "INSERT INTO cases (id, dataset_version_id, case_key, content_hash, prompt) "
            "VALUES ('c', ?, 'z', 'short', 'p')",
            (vid,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        sealed.execute(
            "INSERT INTO cases (id, dataset_version_id, case_key, content_hash, prompt) "
            "VALUES ('c', 'no-such-version', 'z', ?, 'p')",
            ("0" * 64,),
        )


def test_an_unsealed_version_can_never_be_committed_by_the_python_api(kit, store):
    """The import runs in one transaction: whatever happens, only sealed versions persist."""
    kit.datasets.import_cases("qa", CASES)
    with pytest.raises(DatasetError):
        kit.datasets.import_cases("qa", [case("n"), {"case_key": "bad"}])
    unsealed = store._conn.execute("SELECT COUNT(*) FROM dataset_versions WHERE sealed = 0")
    assert unsealed.fetchone()[0] == 0


def test_returned_models_are_copies_not_handles_into_the_store(kit):
    kit.datasets.import_cases("qa", CASES)
    got = kit.datasets.get_case("qa", "a")
    got.prompt = "mutated in memory"
    got.metadata["x"] = 1
    again = kit.datasets.get_case("qa", "a")
    assert again.prompt == "prompt for a" and again.metadata == {}
    assert again.content_hash == again.stored_hash


def test_new_versions_never_rewrite_older_ones(kit, store):
    kit.datasets.import_cases("qa", CASES)
    snapshot = store._conn.execute("SELECT * FROM cases ORDER BY id").fetchall()
    snapshot = [tuple(r) for r in snapshot]
    kit.datasets.import_cases("qa", [case("a", output="CHANGED"), case("b", output="2"), case("c")])
    after = [tuple(r) for r in store._conn.execute("SELECT * FROM cases ORDER BY id")]
    assert all(row in after for row in snapshot)  # every v1 row is byte-identical afterwards
    assert kit.datasets.verify("qa@1").ok and kit.datasets.verify("qa@2").ok
