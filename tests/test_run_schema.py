"""Migration 4's constraints and triggers, driven with raw SQL: they hold with no help from the
Python layer. Also `verify`, which catches what a writer that bypasses the triggers could do."""

import pytest
from conftest import EM, JUDGE, RUN_CASES, complete, make_run, ok, refuses

from evalkit import CaseOutcome, EvaluatorOutcome, Failure


def q(store, sql, params=()):
    return [tuple(r) for r in store._conn.execute(sql, params)]


@pytest.fixture
def world(kit, run):
    """A running run with a complete case result 'a' carrying an ok evaluator result."""
    cr = complete(kit, run, "a")
    er = kit.runs.record_evaluator_result(cr.id, ok(EM.key, 0.5))
    return run, cr, er


# --- write-once results ----------------------------------------------------------------------


def test_case_results_cannot_be_rewritten_or_deleted(kit, run, store, world):
    _, cr, _ = world
    for sql in (
        "UPDATE case_results SET output = 'edited'",
        "UPDATE case_results SET status = 'pending', output = NULL, finished_at = NULL",
        "UPDATE case_results SET run_id = 'other'",
        "UPDATE case_results SET case_id = 'other'",
        "UPDATE case_results SET created_at = 'x'",
    ):
        refuses(store, sql, match="write-once")
    refuses(store, "DELETE FROM case_results", match="permanent")
    assert kit.runs.case_result(run.id, "a").output == "out"


def test_a_pending_case_result_can_only_move_forward_and_only_on_a_running_run(kit, run, store):
    kit.runs.plan(run.id)
    refuses(
        store,
        "UPDATE case_results SET status = 'pending' WHERE status = 'pending'",
        match="write-once",
    )  # not a transition
    refuses(
        store,
        "UPDATE case_results SET run_id = 'other', status = 'complete', output = 'x', "
        "finished_at = 't' WHERE case_id IN (SELECT id FROM cases WHERE case_key = 'a')",
        match="write-once",
    )
    kit.runs.transition(run.id, "partial", stop_reason="x")
    refuses(
        store,
        "UPDATE case_results SET status = 'complete', output = 'x', finished_at = 't'",
        match="only on a running run",
    )


def test_evaluator_results_and_metrics_are_write_once(kit, store, world):
    for sql, match in (
        ("UPDATE evaluator_results SET verdict = 'FAIL'", "write-once"),
        ("UPDATE evaluator_results SET status = 'failed'", "write-once"),
        ("DELETE FROM evaluator_results", "permanent"),
        ("UPDATE metrics SET value = 1.0", "write-once"),
        ("UPDATE metrics SET name = 'other'", "write-once"),
        ("DELETE FROM metrics", "permanent"),
    ):
        refuses(store, sql, match=match)
    assert q(store, "SELECT value FROM metrics") == [(0.5,)]


# --- metrics ---------------------------------------------------------------------------------


def test_metrics_only_attach_to_an_ok_result_of_the_same_run_and_key(kit, run, store, world):
    _, cr, er = world
    other = kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=JUDGE.key,
            status="skipped" if False else "not_applicable",
            detail={"reason": "r"},
        ),
    )
    sql = (
        "INSERT INTO metrics (evaluator_result_id, run_id, evaluator_key, name, value) "
        "VALUES (?,?,?,?,?)"
    )
    m = "ok evaluator result of a running run"
    refuses(store, sql, (other.id, run.id, JUDGE.key, "score", 1.0), match=m)  # not an ok result
    refuses(store, sql, (er.id, "other-run", EM.key, "extra", 1.0), match=m)
    refuses(store, sql, (er.id, run.id, JUDGE.key, "extra", 1.0), match=m)  # wrong key
    refuses(store, sql, ("ghost", run.id, EM.key, "extra", 1.0), match=m)
    refuses(
        store, sql, (er.id, run.id, EM.key, "score", 2.0), match="UNIQUE|PRIMARY"
    )  # duplicate name


@pytest.mark.parametrize("value", [None, float("inf"), float("-inf")])
def test_the_database_refuses_null_and_infinite_metric_values(kit, run, store, world, value):
    """SQLite stores NaN as NULL, so NOT NULL is what stops it; the range stops the infinities."""
    _, _, er = world
    refuses(
        store,
        "INSERT INTO metrics (evaluator_result_id, run_id, evaluator_key, name, value) "
        "VALUES (?,?,?,?,?)",
        (er.id, run.id, EM.key, "extra", value),
    )


def test_metrics_accept_the_extremes_of_the_finite_range(kit, run, store):
    cr = complete(kit, run, "a")
    out = EvaluatorOutcome(
        evaluator_key=EM.key,
        status="ok",
        metrics={"big": 1.7976931348623157e308, "small": -1.7976931348623157e308, "tiny": 5e-324},
    )
    er = kit.runs.record_evaluator_result(cr.id, out)
    assert er.metrics == out.metrics


def test_metric_values_round_trip_exactly(kit, run):
    cr = complete(kit, run, "a")
    values = {"a": 0.1 + 0.2, "b": 1 / 3, "c": 1e-300, "d": -0.0, "e": 123456789.123456789}
    er = kit.runs.record_evaluator_result(
        cr.id, EvaluatorOutcome(evaluator_key=EM.key, status="ok", metrics=values)
    )
    assert kit.runs.evaluator_results(cr.id)[0].metrics == er.metrics == values


# --- CHECKs: an impossible combination cannot be stored, NULLs included ----------------------


CASE_INSERT = (
    "INSERT INTO case_results (id, run_id, case_id, status, output, retrieved_json, "
    "failure_class, failure_kind, failure_message, retryable, started_at, finished_at, "
    "duration_ms, created_at) SELECT 'x', ?, c.id, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 't' "
    "FROM cases c WHERE c.case_key = 'b'"
)


@pytest.mark.parametrize(
    "row",
    [
        # status, output, retrieved, class, kind, message, retryable, started, finished, duration
        ("complete", None, None, None, None, None, None, None, "t", None),  # complete, no output
        ("complete", "o", None, "target", "timeout", "m", 1, None, "t", None),  # complete + failure
        ("complete", "o", None, None, None, None, None, None, None, None),  # no finished_at
        ("failed", "o", None, "target", "timeout", "m", 1, None, "t", None),  # failed + output
        ("failed", None, "[]", "target", "timeout", "m", 1, None, "t", None),  # failed + retrieved
        ("failed", None, None, None, "timeout", "m", 1, None, "t", None),  # NULL class
        ("failed", None, None, "bogus", "timeout", "m", 1, None, "t", None),
        ("failed", None, None, "evaluator", "refused", "m", 1, None, "t", None),
        ("failed", None, None, "target", None, "m", 1, None, "t", None),
        ("failed", None, None, "target", "timeout", None, 1, None, "t", None),
        ("failed", None, None, "target", "timeout", "m", None, None, "t", None),
        ("failed", None, None, "target", "timeout", "m", 2, None, "t", None),  # retryable 0/1 only
        ("failed", None, None, "target", "timeout", "m", 1, None, None, None),
        ("pending", "o", None, None, None, None, None, None, None, None),
        ("pending", None, None, None, None, None, None, None, "t", None),
        ("pending", None, None, "target", "timeout", "m", 1, None, None, None),
        ("complete", "o", None, None, None, None, None, None, "t", -1),  # negative duration
        ("done", "o", None, None, None, None, None, None, "t", None),  # unknown status
    ],
)
def test_the_database_refuses_impossible_case_results(kit, run, store, row):
    refuses(store, CASE_INSERT, (run.id, *row))
    assert q(store, "SELECT COUNT(*) FROM case_results") == [(0,)]


def test_the_database_accepts_the_valid_shapes(kit, run, store):
    conn = store._conn
    good = [
        ("complete", "", None, None, None, None, None, None, "t", None),  # empty output is fine
        ("complete", "o", '["d"]', None, None, None, None, "s", "t", 5),
    ]
    for i, row in enumerate(good):
        conn.execute(
            CASE_INSERT.replace("'x'", f"'x{i}'").replace("'b'", f"'{'bc'[i]}'"), (run.id, *row)
        )
    conn.commit()
    assert q(store, "SELECT COUNT(*) FROM case_results") == [(2,)]


EVAL_INSERT = (
    "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, verdict, "
    "failure_class, failure_kind, failure_message, retryable, created_at) "
    "VALUES ('x', ?, ?, ?, ?, ?, ?, ?, ?, ?, 't')"
)


@pytest.mark.parametrize(
    "row",
    [
        # status, verdict, class, kind, message, retryable
        ("ok", "PASS", "evaluator", "refused", "m", 0),  # ok with a failure
        ("ok", "MAYBE", None, None, None, None),  # bad verdict
        ("not_applicable", "PASS", None, None, None, None),  # verdict without a score
        ("failed", "FAIL", "evaluator", "refused", "m", 0),  # failed with a verdict
        ("failed", None, None, "refused", "m", 0),  # NULL class
        ("failed", None, "target", "exception", "m", 0),  # a target failure is not scorable
        ("failed", None, "evaluator", None, "m", 0),
        ("failed", None, "evaluator", "refused", None, 0),
        ("failed", None, "evaluator", "refused", "m", None),
        ("skipped", None, "infrastructure", "cancelled", "m", 0),  # skipped carries no failure
        ("weird", None, None, None, None, None),
    ],
)
def test_the_database_refuses_impossible_evaluator_results(kit, run, store, row):
    cr = kit.runs.record_case_result(
        run.id,
        "b",
        CaseOutcome.fail(Failure(failure_class="target", kind="exception", message="m")),
    )
    done = complete(kit, run, "c")
    target = cr.id if row[0] == "skipped" else done.id
    refuses(store, EVAL_INSERT, (target, run.id, EM.key, *row))


# --- foreign keys and cross-table consistency ------------------------------------------------


def test_foreign_keys_are_enforced_on_the_store_connection(kit, store):
    assert q(store, "PRAGMA foreign_keys") == [(1,)]
    refuses(
        store,
        "INSERT INTO run_evaluators VALUES ('no-run', 'k:n:000000000000', 'k', 'n', 1, '{}')",
    )


def test_a_populated_database_has_no_foreign_key_violations(kit, run, store):
    kit.runs.plan(run.id)
    for key in RUN_CASES:
        cr = complete(kit, run, key)
        kit.runs.record_evaluator_result(cr.id, ok(EM.key))
    assert q(store, "PRAGMA foreign_key_check") == []
    assert q(store, "PRAGMA integrity_check") == [("ok",)]


def test_dataset_data_stays_immutable_beneath_runs(kit, run, store, world):
    refuses(store, "UPDATE cases SET prompt = 'changed'", match="immutable")
    refuses(store, "DELETE FROM dataset_versions", match="immutable once sealed")
    refuses(store, "DELETE FROM cases WHERE 1", match="immutable")


# --- verify: recompute what can be recomputed ------------------------------------------------


def tamper(store, *statements):
    """Modify data the way a writer that bypassed the triggers could: drop them first."""
    conn = store._conn
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    for sql in statements:
        conn.execute(sql)
    conn.commit()


def test_verify_passes_on_a_healthy_run_and_on_an_empty_one(kit, run, world):
    assert kit.runs.verify(run.id).ok
    assert kit.runs.verify(make_run(kit, EM).id) == kit.runs.verify(make_run(kit, EM).id)


def test_verify_detects_a_modified_config(kit, run, store, world):
    tamper(store, "UPDATE runs SET config_json = replace(config_json, 'em', 'zz')")
    report = kit.runs.verify(run.id)
    assert not report.ok and any(
        "identity_hash" in p or "registered evaluators" in p for p in report.problems
    )


def test_verify_detects_a_modified_identity_or_exec_hash_or_evaluator_count(kit, run, store, world):
    tamper(
        store,
        f"UPDATE runs SET identity_hash = '{'0' * 64}', exec_hash = '{'1' * 64}', "
        "evaluator_count = 9",
    )
    problems = " | ".join(kit.runs.verify(run.id).problems)
    assert "identity_hash" in problems and "exec_hash" in problems and "evaluator_count" in problems


def test_verify_detects_a_changed_dataset_underneath(kit, run, store, world):
    tamper(store, "UPDATE dataset_versions SET content_hash = '" + "e" * 64 + "'")
    assert any("identity_hash" in p for p in kit.runs.verify(run.id).problems)


def test_verify_detects_results_pointing_outside_the_dataset_version(kit, run, store, world):
    kit.datasets.import_cases("qa", [{"case_key": "elsewhere", "prompt": "p"}])
    tamper(
        store,
        "UPDATE case_results SET case_id = (SELECT c.id FROM cases c JOIN dataset_versions v "
        "ON v.id = c.dataset_version_id WHERE v.version_no = 2)",
    )
    assert any("outside the run's dataset version" in p for p in kit.runs.verify(run.id).problems)


def test_verify_detects_an_ok_evaluator_result_without_metrics_and_orphan_metrics(
    kit, run, store, world
):
    tamper(store, "DELETE FROM metrics")
    assert any("have no metrics" in p for p in kit.runs.verify(run.id).problems)


def test_verify_detects_a_scored_target_failure(kit, run, store):
    """Someone writing an ok score onto a failed case result would fabricate a judge opinion."""
    kit.runs.record_case_result(
        run.id,
        "a",
        CaseOutcome.fail(Failure(failure_class="target", kind="exception", message="m")),
    )
    tamper(
        store,
        "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
        f"verdict, created_at) SELECT 'fake', id, run_id, '{EM.key}', 'ok', 'PASS', 't' "
        "FROM case_results",
        "INSERT INTO metrics VALUES ('fake', (SELECT run_id FROM case_results), "
        f"'{EM.key}', 'score', 0.0)",
    )
    assert any(
        "do not fit their case result's status" in p for p in kit.runs.verify(run.id).problems
    )


def test_verify_detects_gaps_in_attempt_numbering(kit, run, store):
    from datetime import UTC, datetime

    from evalkit import RunAttempt

    now = datetime.now(UTC)
    cr = kit.runs.record_case_result(
        run.id, "a", CaseOutcome.complete("x"),
        attempts=[RunAttempt.succeeded(1, now, 1), RunAttempt.succeeded(2, now, 1)],
    )  # fmt: skip
    tamper(store, "DELETE FROM attempts WHERE n = 1")
    assert any("attempts that are not numbered" in p for p in kit.runs.verify(run.id).problems)
    assert cr.id


def test_verify_detects_unknown_failure_kinds_and_a_succeeded_run_with_missing_results(
    kit, run, store
):
    complete(kit, run, "a")
    tamper(
        store,
        "UPDATE case_results SET status = 'failed', output = NULL, failure_class = 'target', "
        "failure_kind = 'made_up', failure_message = 'm', retryable = 0",
        "UPDATE runs SET status = 'succeeded', finished_at = 't'",
    )
    problems = " | ".join(kit.runs.verify(run.id).problems)
    assert "unknown failure target/made_up" in problems
    assert "succeeded but 1 of 3 cases have results" in problems


def test_reading_a_corrupted_row_raises_instead_of_returning_nonsense(kit, run, store):
    """A value the CHECKs cannot judge (an unknown failure kind) is caught when it is read."""
    from evalkit import RunError

    kit.runs.record_case_result(
        run.id,
        "a",
        CaseOutcome.fail(Failure(failure_class="target", kind="exception", message="m")),
    )
    tamper(store, "UPDATE case_results SET failure_kind = 'made_up'")
    with pytest.raises(RunError, match="corrupt or incoherent"):
        kit.runs.case_result(run.id, "a")
    with pytest.raises(RunError, match="corrupt or incoherent"):
        kit.runs.failures(run.id)
    with pytest.raises(RunError, match="corrupt or incoherent"):
        list(kit.runs.case_results(run.id))
