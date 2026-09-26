"""Migration 7: the database itself enforces the retry, reopen and cache rules (raw SQL, no Python
layer), and a v6 database upgrades without losing a row."""

import sqlite3
import uuid
from datetime import UTC, datetime

import pytest
from conftest import EM, complete, make_run, refuses

from evalkit import CaseOutcome, EvalKit, EvaluatorOutcome, Failure, FailureClass
from evalkit.migrations import MIGRATIONS, migrate
from evalkit.store import SQLiteStore

NOW = datetime.now(UTC).isoformat()
UNAVAILABLE = Failure(failure_class=FailureClass.INFRA, kind="provider_unavailable", message="503")


def ex(store, sql, *params):
    store._conn.execute(sql, params)


def history(store, run, cr_id, round_=0, scope="case", ev_id=None, ev_key=None):
    store._conn.execute(
        "INSERT INTO result_history (id, run_id, scope, case_result_id, evaluator_result_id, "
        "evaluator_key, retry_round, failure_class, failure_kind, failure_message, retryable, "
        "superseded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), run.id, scope, cr_id, ev_id, ev_key, round_, "infrastructure",
         "provider_unavailable", "503", 1, NOW),
    )  # fmt: skip
    store._conn.commit()


@pytest.fixture
def failed_case(kit, store):
    """A running run whose case `a` failed retryably (and has its skipped evaluator rows)."""
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    kit.runs.record_case_result(run.id, "a", CaseOutcome.fail(UNAVAILABLE))
    cr = kit.runs.case_result(run.id, "a")
    key = kit.runs.get(run.id).config.evaluators[0].key
    kit.runs.record_evaluator_result(
        cr.id, EvaluatorOutcome(evaluator_key=key, status="skipped", detail={"reason": "t"})
    )
    return run, cr, key


@pytest.fixture
def failed_evaluator(kit, store):
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    complete(kit, run, "a")
    cr = kit.runs.case_result(run.id, "a")
    key = kit.runs.get(run.id).config.evaluators[0].key
    er = kit.runs.record_evaluator_result(
        cr.id, EvaluatorOutcome(evaluator_key=key, status="failed", failure=UNAVAILABLE)
    )
    return run, cr, er, key


REOPEN = (
    "UPDATE case_results SET status = 'pending', failure_class = NULL, failure_kind = NULL, "
    "failure_message = NULL, retryable = NULL, started_at = NULL, finished_at = NULL, "
    "duration_ms = NULL, retry_round = ? WHERE id = ?"
)


def test_a_failed_case_result_is_reopened_only_with_its_failure_archived(store, failed_case):
    run, cr, _ = failed_case
    refuses(store, REOPEN, (1, cr.id), match="write-once")  # no history row
    history(store, run, cr.id, round_=5)  # a history row for another round
    refuses(store, REOPEN, (1, cr.id), match="write-once")
    history(store, run, cr.id, round_=0)
    refuses(store, REOPEN, (2, cr.id), match="write-once")  # the round must advance by exactly one
    refuses(store, REOPEN, (0, cr.id), match="write-once")
    ex(store, REOPEN, 1, cr.id)  # now legal
    store._conn.commit()
    row = store._conn.execute(
        "SELECT status, retry_round FROM case_results WHERE id = ?", (cr.id,)
    ).fetchone()
    assert tuple(row) == ("pending", 1)


def test_a_reopen_cannot_smuggle_other_changes(store, failed_case):
    run, cr, _ = failed_case
    history(store, run, cr.id)
    for change in ("ord = 12345", "created_at = 'x'", "case_id = 'other'", "run_id = 'other'"):
        refuses(store, REOPEN.replace("retry_round = ?", f"retry_round = ?, {change}"), (1, cr.id))


def test_a_complete_case_result_can_never_be_reopened(kit, store):
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    cr = complete(kit, run, "a")
    history(store, run, cr.id)  # even with a forged history row
    refuses(
        store,
        "UPDATE case_results SET status = 'pending', output = NULL, retry_round = 1 WHERE id = ?",
        (cr.id,),
        match="write-once",
    )
    refuses(store, "UPDATE case_results SET output = 'edited' WHERE id = ?", (cr.id,))


def test_reopening_needs_a_running_run(kit, store, failed_case):
    run, cr, _ = failed_case
    history(store, run, cr.id)
    kit.runs.transition(run.id, "partial", stop_reason="x")
    refuses(store, REOPEN, (1, cr.id), match="write-once")


def test_the_skipped_rows_of_a_reopened_case_may_go_and_nothing_else_may(store, failed_case):
    run, cr, key = failed_case
    refuses(
        store,
        "DELETE FROM evaluator_results WHERE case_result_id = ?",
        (cr.id,),
        match="permanent",
    )  # the case result is still failed
    history(store, run, cr.id)
    ex(store, REOPEN, 1, cr.id)
    ex(
        store,
        "DELETE FROM evaluator_results WHERE case_result_id = ? AND status = 'skipped'",
        cr.id,
    )
    store._conn.commit()
    assert store._conn.execute("SELECT COUNT(*) FROM evaluator_results").fetchone()[0] == 0


def test_only_skipped_rows_can_be_deleted(kit, store, failed_evaluator):
    run, cr, er, key = failed_evaluator
    refuses(store, "DELETE FROM evaluator_results WHERE id = ?", (er.id,), match="permanent")
    refuses(store, "DELETE FROM case_results WHERE id = ?", (cr.id,), match="permanent")
    history(store, run, cr.id, 0, "evaluator", er.id, key)
    refuses(store, "DELETE FROM result_history", match="append-only")


UPDATE_EV = (
    "UPDATE evaluator_results SET status = 'ok', verdict = 'PASS', failure_class = NULL, "
    "failure_kind = NULL, failure_message = NULL, retryable = NULL, retry_round = ? WHERE id = ?"
)


def test_a_failed_evaluator_result_takes_a_new_outcome_only_with_history(store, failed_evaluator):
    run, cr, er, key = failed_evaluator
    refuses(store, UPDATE_EV, (1, er.id), match="write-once")  # nothing archived
    history(store, run, cr.id, 0, "evaluator", er.id, key)
    refuses(store, UPDATE_EV, (2, er.id), match="write-once")  # round must be exactly +1
    refuses(store, UPDATE_EV, (0, er.id), match="write-once")
    refuses(store, UPDATE_EV.replace("retry_round = ?", "retry_round = ?, evaluator_key = 'x'"),
            (1, er.id))  # fmt: skip
    refuses(store, UPDATE_EV.replace("retry_round = ?", "retry_round = ?, created_at = 'x'"),
            (1, er.id))  # fmt: skip
    ex(store, UPDATE_EV, 1, er.id)
    store._conn.commit()
    assert store._conn.execute(
        "SELECT status, retry_round FROM evaluator_results WHERE id = ?", (er.id,)
    ).fetchone()[:] == ("ok", 1)


def test_a_successful_evaluator_result_is_never_replaced(kit, store):
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    cr = complete(kit, run, "a")
    key = kit.runs.get(run.id).config.evaluators[0].key
    er = kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(evaluator_key=key, status="ok", verdict="PASS", metrics={"score": 1.0}),
    )
    history(store, run, cr.id, 0, "evaluator", er.id, key)  # even with a forged history row
    refuses(store, "UPDATE evaluator_results SET status = 'failed', retry_round = 1 WHERE id = ?",
            (er.id,), match="write-once")  # fmt: skip
    refuses(store, "UPDATE evaluator_results SET verdict = 'FAIL' WHERE id = ?", (er.id,))
    refuses(store, "UPDATE metrics SET value = 0 WHERE evaluator_result_id = ?", (er.id,))


def test_a_retried_ok_result_can_get_its_metrics(kit, store, failed_evaluator):
    run, cr, er, key = failed_evaluator
    history(store, run, cr.id, 0, "evaluator", er.id, key)
    ex(store, UPDATE_EV, 1, er.id)
    ex(
        store,
        "INSERT INTO metrics (evaluator_result_id, run_id, evaluator_key, name, value) "
        "VALUES (?,?,?,?,?)",
        er.id, run.id, key, "score", 1.0,
    )  # fmt: skip
    store._conn.commit()
    assert kit.runs.evaluator_results(cr.id)[0].score == 1.0


def test_history_is_append_only_unique_per_round_and_needs_a_running_run(kit, store, failed_case):
    run, cr, _ = failed_case
    history(store, run, cr.id, 0)
    with pytest.raises(sqlite3.IntegrityError):
        history(store, run, cr.id, 0)  # one archive per round
    store._conn.rollback()
    refuses(store, "UPDATE result_history SET failure_kind = 'x'", match="append-only")
    kit.runs.transition(run.id, "partial", stop_reason="x")
    with pytest.raises(sqlite3.IntegrityError, match="running"):
        history(store, run, cr.id, 1)
    store._conn.rollback()


def test_history_rows_must_fit_their_scope(store, failed_case):
    run, cr, key = failed_case
    refuses(
        store,
        "INSERT INTO result_history (id, run_id, scope, case_result_id, evaluator_key, "
        "retry_round, failure_class, failure_kind, failure_message, retryable, superseded_at) "
        "VALUES ('h', ?, 'case', ?, 'k', 0, 'a', 'b', 'c', 1, 'now')",
        (run.id, cr.id[0:0] or cr.id),
    )


# ---- reopening a succeeded run ------------------------------------------------------------------


def succeed(kit, store):
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    for k in ("a", "b", "c"):
        complete(kit, run, k)
    from conftest import settle

    settle(kit, run)
    return kit.runs.transition(run.id, "succeeded")


REOPEN_RUN = (
    "UPDATE runs SET status = 'running', stop_reason = NULL, finished_at = NULL WHERE id = ?"
)


def ticket(store, run, finished_at, seq=1):
    store._conn.execute(
        "INSERT INTO run_executions (id, run_id, seq, started_at, status_before, retry_failed, "
        "reopens_finished_at) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), run.id, seq, NOW, "succeeded", 1, finished_at),
    )
    store._conn.commit()


def test_a_succeeded_run_is_reopened_only_by_an_execution_naming_its_finish(kit, store):
    run = succeed(kit, store)
    refuses(store, REOPEN_RUN, (run.id,), match="illegal run status")
    ticket(store, run, "2000-01-01T00:00:00+00:00")  # a ticket for some other finish
    refuses(store, REOPEN_RUN, (run.id,), match="illegal run status")
    ticket(store, run, run.finished_at.isoformat(), seq=2)
    kit.store.reopen_succeeded_run(run.id)
    assert kit.runs.get(run.id).status == "running" and kit.runs.get(run.id).finished_at is None


def test_the_service_still_refuses_to_reopen_a_succeeded_run(kit, store):
    from evalkit import RunError

    run = succeed(kit, store)
    with pytest.raises(RunError, match="cannot become"):
        kit.runs.transition(run.id, "running")


def test_a_ticket_cannot_be_reused_for_a_later_finish(kit, store):
    run = succeed(kit, store)
    ticket(store, run, run.finished_at.isoformat())
    kit.store.reopen_succeeded_run(run.id)
    settle_and_finish = kit.runs.transition(run.id, "succeeded")  # finishes again, later
    assert settle_and_finish.finished_at != run.finished_at
    refuses(store, REOPEN_RUN, (run.id,), match="illegal run status")


def test_execution_records_finish_once_and_are_permanent(kit, store):
    run = succeed(kit, store)
    ticket(store, run, run.finished_at.isoformat())
    finish = "UPDATE run_executions SET finished_at = ?, units = 1 WHERE run_id = ?"
    ex(store, finish, NOW, run.id)
    store._conn.commit()
    refuses(store, finish, (NOW, run.id), match="finished once")
    refuses(store, "UPDATE run_executions SET seq = 9", match="finished once")
    refuses(store, "DELETE FROM run_executions", match="permanent")


# ---- attempts and cache -------------------------------------------------------------------------


def test_attempt_columns_are_constrained(kit, store):
    run = make_run(kit, EM, start=True)
    kit.runs.plan(run.id)
    cr = complete(kit, run, "a")
    base = (
        "INSERT INTO attempts (id, case_result_id, n, started_at, duration_ms, outcome, {col}) "
        "VALUES (?, ?, 1, ?, 1, 'ok', ?)"
    )
    for col, bad in (("cache_hit", 2), ("cost_usd", -1.0), ("retry_round", -1)):
        refuses(store, base.format(col=col), (str(uuid.uuid4()), cr.id, NOW, bad))


def test_the_cache_key_must_be_a_sha256(store):
    refuses(
        store,
        "INSERT INTO llm_cache (key, provider, model, response_json, created_at) "
        "VALUES ('short', 'p', 'm', '{}', 'now')",
    )


def test_cache_entries_are_unique_and_clearable(store):
    store.cache_put("a" * 64, "p", "m", "1")
    store.cache_put("a" * 64, "p", "m", "2")
    assert store.cache_get("a" * 64) == "1"
    assert store.cache_stats()["entries"] == 1
    assert store.cache_clear() == 1 and store.cache_get("a" * 64) is None


# ---- a v6 database upgrades ---------------------------------------------------------------------

DATA_TABLES = (
    "datasets", "dataset_versions", "cases", "runs", "run_evaluators", "case_results",
    "evaluator_results", "metrics", "attempts", "run_summaries", "run_tags",
)  # fmt: skip


def downgrade_copy(src: SQLiteStore, dst_path) -> None:
    """A v6 database holding the same rows as `src` (only the columns v6 has)."""
    conn = sqlite3.connect(dst_path)
    migrate(conn, path=dst_path, backup=False, migrations=MIGRATIONS[:6])
    triggers = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'").fetchall()
    for name, _ in triggers:
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("ATTACH DATABASE ? AS src", (str(src.path),))
    for table in DATA_TABLES:
        cols = [r[1] for r in conn.execute(f"PRAGMA main.table_info({table})")]
        names = ", ".join(cols)
        conn.execute(f"INSERT INTO main.{table} ({names}) SELECT {names} FROM src.{table}")
    conn.commit()
    conn.execute("DETACH DATABASE src")
    for _, sql in triggers:
        conn.execute(sql)
    conn.commit()
    conn.close()


def test_a_v6_database_upgrades_keeps_every_row_and_retries(tmp_path):
    from test_retry_failed import RX, Flaky, dataset
    from test_retry_failed import UNAVAILABLE as UN

    src_path = tmp_path / "src.db"
    kit = EvalKit.open(src_path)
    dataset(kit, 6)
    flaky = Flaky({"c01": 1, "c04": 1}, error=UN)
    run = kit.controller.create(
        "qa",
        target=flaky.target(),
        evaluators=[RX],
        policy={"retry": {"max_attempts": 1, "base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}},
    )
    kit.controller.execute(run.id, target=flaky.target())
    counts_before = {
        t: kit.store._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in DATA_TABLES
    }
    kit.close()

    v6 = tmp_path / "v6.db"
    downgrade_copy(SQLiteStore(src_path), v6)
    conn = sqlite3.connect(v6)
    assert max(r[0] for r in conn.execute("SELECT version FROM schema_migrations")) == 6
    assert "retry_round" not in {r[1] for r in conn.execute("PRAGMA table_info(case_results)")}
    conn.close()

    kit = EvalKit.open(v6)  # migrates
    assert list(tmp_path.glob("v6.db.bak-v6-*")), "a backup is written before migrating"
    conn = kit.store._conn
    assert max(r[0] for r in conn.execute("SELECT version FROM schema_migrations")) == 7
    for table, n in counts_before.items():
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, table
    assert conn.execute("SELECT COUNT(*) FROM result_history").fetchone()[0] == 0
    rows = conn.execute("SELECT DISTINCT retry_round FROM case_results").fetchall()
    assert [tuple(r) for r in rows] == [(0,)]
    rows = conn.execute("SELECT DISTINCT cache_hit, cost_usd FROM attempts").fetchall()
    assert [tuple(r) for r in rows] == [(0, None)]
    migrated = kit.runs.get(run.id)
    assert migrated.status == "succeeded" and kit.runs.verify(run.id).ok
    assert kit.runs.counts(run.id).failed == 2
    assert kit.summarize(run.id).usage["provider_calls"] == 6  # old attempts count as calls

    # the migrated run can be retried: only its two failures are called again
    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert report.status == "succeeded" and report.retry["reopened_cases"] == 2
    assert kit.runs.counts(run.id).failed == 0 and kit.runs.verify(run.id).ok
    assert sorted(k for k, n in flaky.calls.items() if n == 2) == ["c01", "c04"]
    kit.close()


def test_migration_7_is_recorded_once_and_reopening_is_idempotent(tmp_path):
    path = tmp_path / "x.db"
    SQLiteStore(path).close()
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    assert [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")] == list(
        range(1, len(MIGRATIONS) + 1)
    )
    assert MIGRATIONS[-1].name == "p2_1_operations"


def test_verify_notices_history_that_does_not_match_the_retry_rounds(kit, store, failed_case):
    run, cr, _ = failed_case
    assert kit.runs.verify(run.id).ok
    history(store, run, cr.id, 0)  # an archived failure for a result that was never reopened
    report = kit.runs.verify(run.id)
    assert not report.ok and any("retry round" in p for p in report.problems)


def test_verify_accepts_a_retried_run(kit):
    from test_retry_failed import RX, Flaky, dataset

    dataset(kit, 4)
    flaky = Flaky({"c01": 1})
    run = kit.controller.create(
        "qa",
        target=flaky.target(),
        evaluators=[RX],
        policy={"retry": {"max_attempts": 1, "base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}},
    )
    kit.controller.execute(run.id, target=flaky.target())
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert kit.runs.verify(run.id).ok


def test_a_failed_begin_or_reopen_rolls_back_and_leaves_the_store_usable(kit, store):
    with pytest.raises(TypeError):  # an unknown run: nothing to read
        store.begin_execution("no-such-run", retry_failed=False, environment_json="{}")
    assert not store._conn.in_transaction
    run = succeed(kit, store)
    with pytest.raises(sqlite3.IntegrityError, match="illegal run status"):
        store.reopen_succeeded_run(run.id)  # no execution names this finish
    assert not store._conn.in_transaction and kit.runs.get(run.id).status == "succeeded"
    eid, seq, before = store.begin_execution(run.id, retry_failed=True, environment_json="{}")
    assert (seq, before) == (1, "succeeded")
    store.finish_execution(
        eid, reopened_cases=0, reopened_evaluators=0, units=0, elapsed_s=0.0,
        outcome_status="succeeded", stop_reason=None, budget_json="{}", spend_json="{}",
    )  # fmt: skip
    store.finish_execution(  # a second finish is a no-op: the record is finished once
        eid, reopened_cases=9, reopened_evaluators=9, units=9, elapsed_s=9.0,
        outcome_status="partial", stop_reason="x", budget_json="{}", spend_json="{}",
    )  # fmt: skip
    (row,) = store.list_executions(run.id)
    assert row["units"] == 0 and row["outcome_status"] == "succeeded"
