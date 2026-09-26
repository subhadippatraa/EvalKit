"""Migration 5 (summaries, tags, generalized reviews, judge checks) and the full upgrade chain."""

import sqlite3

import pytest
from conftest import CRITERIA, FakeJudge, case, judged
from legacy_schemas import V0_SCHEMA, V1_SCHEMA
from test_migrations import applied, columns, legacy_row, make_db

from evalkit import EvalKit, Evaluator, MigrationError
from evalkit.migrations import ANALYSIS_SQL, MIGRATIONS, Migration, migrate
from evalkit.store import SQLiteStore

ALL = [m.version for m in MIGRATIONS]
NEW_TABLES = {"run_summaries", "run_tags", "judge_checks"}


def tables(path):
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def upto(path, n):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn, path=path, backup=False, migrations=MIGRATIONS[:n])
    return conn


def test_a_fresh_database_gets_the_analysis_schema(tmp_path):
    path = tmp_path / "new.db"
    SQLiteStore(path).close()
    assert applied(path) == ALL and ALL[-1] == 6 and NEW_TABLES <= tables(path)
    assert columns(path, "reviews") == {
        "id", "evaluation_id", "case_result_id", "evaluator_key", "sample", "reviewer", "score",
        "verdict", "comment", "created_at",
    }  # fmt: skip
    assert columns(path, "run_summaries") == {
        "run_id",
        "computed_at",
        "evalkit_version",
        "summary_json",
    }
    assert columns(path, "run_tags") == {"run_id", "tag", "created_at"}
    assert {"evaluator_key", "fixture_hash", "adversarial_correct", "results_json"} <= columns(
        path, "judge_checks"
    )


def test_the_reviews_rebuild_keeps_every_existing_review_and_the_legacy_api_works(tmp_path):
    """The riskiest step: `reviews` is rebuilt to relax its CHECK. Real reviews written under the
    previous schema must survive byte for byte, and the single-record API must keep working."""
    path = tmp_path / "p1.db"
    conn = upto(path, 4)
    sql, params = legacy_row(True, id="old-1")
    conn.execute(sql, params)
    conn.execute(
        "INSERT INTO reviews (id, evaluation_id, reviewer, score, verdict, comment, created_at) "
        "VALUES ('r1', 'old-1', 'alice', 0.5, 'PASS', 'looks fine', '2026-01-02T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO reviews (id, evaluation_id, reviewer, score, verdict, comment, created_at) "
        "VALUES ('r2', 'old-1', 'bob', NULL, 'FAIL', NULL, '2026-01-03T00:00:00+00:00')"
    )
    conn.commit()
    before = [tuple(r) for r in conn.execute("SELECT * FROM reviews ORDER BY id")]
    conn.close()
    assert applied(path) == [1, 2, 3, 4]

    store = SQLiteStore(path)
    assert applied(path) == ALL
    rows = store._conn.execute(
        "SELECT id, evaluation_id, reviewer, score, verdict, comment, created_at "
        "FROM reviews ORDER BY id"
    ).fetchall()
    assert [tuple(r) for r in rows] == before
    assert store._conn.execute(
        "SELECT DISTINCT sample, case_result_id, evaluator_key FROM reviews"
    ).fetchall()[0][:] == ("targeted", None, None)
    ev = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store)
    assert [r.reviewer for r in ev.get("old-1").reviews] == ["alice", "bob"]
    new = ev.review("old-1", "carol", "PASS")
    assert [r.id for r in ev.get("old-1").reviews][-1] == new.id
    assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert store._conn.execute("PRAGMA integrity_check").fetchall()[0][0] == "ok"
    assert len(list(tmp_path.glob("p1.db.bak-v4-*"))) == 1


def test_run_and_legacy_reviews_coexist_after_the_rebuild(tmp_path):
    store = SQLiteStore(tmp_path / "x.db")
    kit = EvalKit(store)
    ev = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store)
    result = ev.evaluate("p", "o", criteria=CRITERIA)
    ev.review(result.id, "alice", "PASS")
    from conftest import EM, make_run

    kit.datasets.import_cases("qa", [case("a", output="o", reference="r")])
    run = make_run(kit, EM, start=True)
    from evalkit import CaseOutcome

    kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("o"))
    kit.reviews.add(run.id, "a", reviewer="bob", verdict="PASS")
    assert store._conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2
    assert len(ev.get(result.id).reviews) == 1  # the run review is not a legacy review


@pytest.mark.parametrize("schema", [V0_SCHEMA, V1_SCHEMA], ids=["v0-legacy", "v1-0.1.0"])
def test_every_historical_schema_upgrades_through_all_five_migrations(tmp_path, schema):
    path = tmp_path / "legacy.db"
    make_db(path, schema, [legacy_row(schema is V1_SCHEMA)])
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO reviews (id, evaluation_id, reviewer, verdict, created_at) "
        "VALUES ('r', 'old-1', 'a', 'PASS', '2026-01-02T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    store = SQLiteStore(path)
    assert applied(path) == ALL
    assert store.get("old-1").reviews[0].id == "r"
    kit = EvalKit(store)
    kit.datasets.import_cases("qa", [case("a", output="o", reference="r")])
    from conftest import EM, complete, make_run, ok

    run = make_run(kit, EM, start=True)
    kit.runs.record_evaluator_result(complete(kit, run, "a").id, ok(EM.key))
    assert kit.runs.verify(run.id).ok
    assert kit.summarize(run.id).evaluators[EM.key].scored == 1


def test_a_failing_analysis_migration_rolls_back_completely_including_the_reviews_rebuild(tmp_path):
    path = tmp_path / "t.db"
    conn = upto(path, 4)
    sql, params = legacy_row(True, id="old-1")
    conn.execute(sql, params)
    conn.execute(
        "INSERT INTO reviews (id, evaluation_id, reviewer, verdict, created_at) "
        "VALUES ('r', 'old-1', 'a', 'PASS', 't')"
    )
    conn.commit()
    bad = Migration(5, "analysis_foundation", ANALYSIS_SQL + "\nSELECT nope FROM nope;")
    with pytest.raises(MigrationError, match="failed"):
        migrate(conn, path=path, backup=False, migrations=(*MIGRATIONS[:4], bad))
    conn.close()
    assert applied(path) == [1, 2, 3, 4]
    assert not NEW_TABLES & tables(path) and "reviews_new" not in tables(path)
    assert columns(path, "reviews") == {
        "id",
        "evaluation_id",
        "reviewer",
        "score",
        "verdict",
        "comment",
        "created_at",
    }
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
    SQLiteStore(path).close()
    assert applied(path) == ALL


def test_the_newest_migration_is_pinned():
    assert MIGRATIONS[4].name == "analysis_foundation"
    assert (
        MIGRATIONS[4].checksum == "55838dc14c3feac064f5406dff88e922cf17cf153edb7ce05d5c254d162eb5ed"
    )


def test_the_upgraded_and_the_fresh_schema_are_identical(tmp_path):
    """Migrating step by step must land on the same schema as a fresh open (no drift)."""
    fresh = tmp_path / "fresh.db"
    SQLiteStore(fresh).close()
    stepped = tmp_path / "stepped.db"
    conn = upto(stepped, 1)
    conn.close()
    for n in range(2, len(MIGRATIONS) + 1):
        conn = sqlite3.connect(stepped)
        conn.row_factory = sqlite3.Row
        migrate(conn, path=stepped, backup=False, migrations=MIGRATIONS[:n])
        conn.close()

    def schema(path):
        c = sqlite3.connect(path)
        try:
            return sorted(
                (r[0], r[1], " ".join((r[2] or "").split()))
                for r in c.execute(
                    "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                )
            )
        finally:
            c.close()

    assert schema(fresh) == schema(stepped)
