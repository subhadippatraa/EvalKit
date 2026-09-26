"""Runs: creation and identity, dataset-version binding, frozen configuration, lifecycle."""

import itertools

import pytest
from conftest import EM, JUDGE, RUN_CASES, case, complete, make_run, refuses, settle
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import (
    CaseOutcome,
    DatasetError,
    EvalFailure,
    RunConfig,
    RunError,
)
from evalkit.runs import TERMINAL_STATUSES, TRANSITIONS

STATUSES = sorted(TRANSITIONS)


def raw(store):
    return store._conn


# --- creation and identity -------------------------------------------------------------------


def test_a_new_run_is_created_with_a_frozen_config_and_a_reproducible_identity(kit):
    run = make_run(kit, EM, JUDGE, name="baseline")
    assert (run.status, run.name, run.dataset_ref) == ("created", "baseline", "qa@1")
    assert run.started_at is None and run.finished_at is None and run.stop_reason is None
    assert run.config.evaluator_keys == sorted([EM.key, JUDGE.key])
    version = kit.datasets.resolve("qa")
    assert run.dataset_version_id == version.id
    assert run.identity_hash == run.config.identity_hash(version.content_hash)
    assert run.exec_hash == run.config.exec_hash
    assert {"evalkit_version", "python", "sqlite"} <= set(run.environment)
    assert kit.runs.get(run.id) == run


def test_identical_measurements_share_an_identity_but_are_distinct_runs(kit):
    a = make_run(kit, EM, JUDGE, name="one")
    b = make_run(kit, JUDGE, EM, name="two")  # same evaluators, other order and name
    assert a.id != b.id and a.identity_hash == b.identity_hash


def test_a_different_execution_policy_keeps_identity_but_changes_exec_hash(kit):
    a = make_run(kit, config=RunConfig(evaluators=[EM], policy={"concurrency": 1}))
    b = make_run(kit, config=RunConfig(evaluators=[EM], policy={"concurrency": 32}))
    assert a.identity_hash == b.identity_hash and a.exec_hash != b.exec_hash


def test_identity_follows_dataset_content_not_its_name_or_version_number(kit):
    cases = [case("a", output="x")]
    kit.datasets.import_cases("one", cases)
    kit.datasets.import_cases("two", cases)
    a = kit.runs.create("one", RunConfig(evaluators=[EM]))
    b = kit.runs.create("two", RunConfig(evaluators=[EM]))
    assert a.identity_hash == b.identity_hash  # same content, measured the same way
    kit.datasets.import_cases("one", [case("a", output="changed")])
    c = kit.runs.create("one", RunConfig(evaluators=[EM]))
    assert c.identity_hash != a.identity_hash and c.dataset_ref == "one@2"


def test_run_environment_records_extras_and_refuses_non_json(kit):
    kit.datasets.import_cases("qa", [case("a")])
    run = kit.runs.create("qa", RunConfig(), environment={"git_sha": "abc123", "region": "eu"})
    assert run.environment["git_sha"] == "abc123"
    with pytest.raises(ValueError):
        kit.runs.create("qa", RunConfig(), environment={"x": float("nan")})


@pytest.mark.parametrize("bad", ["", "x" * 129, "line\nbreak"])
def test_name_and_idempotency_key_are_validated(kit, bad):
    kit.datasets.import_cases("qa", [case("a")])
    with pytest.raises(RunError, match="printable"):
        kit.runs.create("qa", RunConfig(), name=bad)
    with pytest.raises(RunError, match="printable"):
        kit.runs.create("qa", RunConfig(), idempotency_key=bad)


# --- dataset-version binding -----------------------------------------------------------------


def test_a_run_pins_exactly_one_sealed_dataset_version(kit, store):
    run = make_run(kit, EM)
    rows = raw(store).execute(
        "SELECT v.sealed FROM runs r JOIN dataset_versions v ON v.id = r.dataset_version_id"
    )
    assert [tuple(r) for r in rows] == [(1,)]
    assert run.dataset_version_id == kit.datasets.resolve("qa@1").id


def test_latest_is_resolved_once_at_creation_and_the_run_stays_pinned(kit):
    kit.datasets.import_cases("qa", [case("a", output="v1")])
    run = kit.runs.create("qa@latest", RunConfig())
    kit.datasets.import_cases("qa", [case("a", output="v2")])
    assert kit.runs.get(run.id).dataset_ref == "qa@1"
    assert kit.runs.create("qa@latest", RunConfig()).dataset_ref == "qa@2"
    assert kit.runs.create("qa@1", RunConfig()).dataset_version_id == run.dataset_version_id


def test_a_run_of_an_unknown_dataset_or_version_is_refused_and_creates_nothing(kit, store):
    with pytest.raises(DatasetError, match="not found"):
        kit.runs.create("nope", RunConfig())
    kit.datasets.import_cases("qa", [case("a")])
    with pytest.raises(DatasetError, match="no version"):
        kit.runs.create("qa@9", RunConfig())
    assert raw(store).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_the_database_refuses_a_run_on_an_unsealed_or_missing_dataset_version(store):
    conn = raw(store)
    conn.execute("INSERT INTO datasets (id, name, created_at) VALUES ('d', 'x', 't')")
    conn.execute(
        "INSERT INTO dataset_versions (id, dataset_id, version_no, created_at, sealed) "
        "VALUES ('v', 'd', 1, 't', 0)"
    )
    conn.commit()
    insert = (
        "INSERT INTO runs (id, dataset_version_id, status, config_json, identity_hash, "
        "exec_hash, evaluator_count, created_at) VALUES ('r', ?, 'created', '{}', ?, ?, 0, 't')"
    )
    for version_id in ("v", "no-such-version"):
        refuses(store, insert, (version_id, "a" * 64, "b" * 64), match="sealed dataset version")


def test_sealed_dataset_versions_stay_immutable_with_runs_pointing_at_them(kit, store):
    make_run(kit, EM)
    refuses(store, "UPDATE dataset_versions SET case_count = 99", match="immutable once sealed")
    refuses(store, "DELETE FROM cases", match="immutable")


# --- idempotency -----------------------------------------------------------------------------


def test_repeating_a_create_with_the_same_key_returns_the_same_run(kit, store):
    a = make_run(kit, EM, idempotency_key="nightly-1")
    b = kit.runs.create("qa", RunConfig(evaluators=[EM]), idempotency_key="nightly-1")
    assert a.id == b.id
    assert raw(store).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert raw(store).execute("SELECT COUNT(*) FROM run_evaluators").fetchone()[0] == 1


def test_a_key_reused_for_a_different_configuration_is_an_error_not_a_new_run(kit, store):
    make_run(kit, EM, idempotency_key="k")
    with pytest.raises(RunError, match="different dataset version or configuration"):
        kit.runs.create("qa", RunConfig(evaluators=[EM, JUDGE]), idempotency_key="k")
    kit.datasets.import_cases("qa", [case("a", output="new")])
    with pytest.raises(RunError, match="different dataset version or configuration"):
        kit.runs.create("qa@latest", RunConfig(evaluators=[EM]), idempotency_key="k")
    assert raw(store).execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_a_changed_execution_policy_also_conflicts_on_the_same_key(kit):
    make_run(kit, config=RunConfig(policy={"c": 1}), idempotency_key="k")
    with pytest.raises(RunError, match="different"):
        kit.runs.create("qa", RunConfig(policy={"c": 2}), idempotency_key="k")


def test_runs_without_a_key_are_never_deduplicated(kit):
    assert make_run(kit, EM).id != make_run(kit, EM).id


# --- listing ---------------------------------------------------------------------------------


def test_list_is_newest_first_and_filters_by_dataset_version_and_status(kit):
    a = make_run(kit, EM, name="a")
    kit.datasets.import_cases("qa", [case("a", output="v2")])
    b = kit.runs.create("qa@2", RunConfig())
    c = kit.runs.create("qa@2", RunConfig())
    kit.runs.start(c.id)
    assert [r.id for r in kit.runs.list()] == [c.id, b.id, a.id]
    assert [r.id for r in kit.runs.list(dataset_ref="qa@1")] == [a.id]
    assert [r.id for r in kit.runs.list(dataset_ref="qa@2", status="created")] == [b.id]
    assert [r.id for r in kit.runs.list(limit=1)] == [c.id]


@pytest.mark.parametrize("limit", [0, -1, 10_001, True, "5", 1.5])
def test_list_validates_limit_and_status(kit, limit):
    with pytest.raises(RunError, match="limit"):
        kit.runs.list(limit=limit)
    with pytest.raises(RunError, match="unknown run status"):
        kit.runs.list(status="nope")


def test_unknown_run_ids_raise(kit):
    for call in (kit.runs.get, kit.runs.start, kit.runs.plan, kit.runs.counts, kit.runs.verify):
        with pytest.raises(RunError, match="not found"):
            call("nope")


# --- frozen configuration (database level) ---------------------------------------------------


@pytest.mark.parametrize(
    "column,value",
    [
        ("id", "other"),
        ("dataset_version_id", "other"),
        ("name", "renamed"),
        ("config_json", "{}"),
        ("identity_hash", "c" * 64),
        ("exec_hash", "c" * 64),
        ("environment_json", "{}"),
        ("idempotency_key", "other"),
        ("created_at", "2000-01-01"),
    ],
)
def test_a_runs_frozen_columns_cannot_change_even_in_a_legal_transition(kit, store, column, value):
    run = make_run(kit, EM, idempotency_key="k")
    refuses(
        store,
        f"UPDATE runs SET {column} = ?, status = 'running', started_at = 't' WHERE id = ?",
        (value, run.id),
        match="frozen",
    )
    assert kit.runs.get(run.id) == run


def test_started_at_is_set_once_and_survives_a_resume(kit, store):
    run = make_run(kit, EM, start=True)
    first = kit.runs.get(run.id).started_at
    kit.runs.transition(run.id, "partial", stop_reason="budget")
    resumed = kit.runs.start(run.id)
    assert resumed.started_at == first and resumed.finished_at is None
    assert resumed.stop_reason is None
    kit.runs.transition(run.id, "partial", stop_reason="again")
    refuses(
        store,
        "UPDATE runs SET status = 'running', started_at = 'later', finished_at = NULL, "
        "stop_reason = NULL WHERE id = ?",
        (run.id,),
        match="frozen",
    )


def test_the_evaluators_of_a_run_are_immutable_and_fixed_at_creation(kit, store):
    run = make_run(kit, EM)
    extra = "INSERT INTO run_evaluators VALUES (?, 'x:y:000000000000', 'x', 'y', 1, '{}')"
    # the run recorded one evaluator: no second one can be added, even while still `created`
    refuses(store, extra, (run.id,), match="fixed when the run is created")
    refuses(store, "UPDATE run_evaluators SET name = 'z'", match="immutable")
    refuses(store, "DELETE FROM run_evaluators", match="immutable")
    kit.runs.start(run.id)
    refuses(store, extra, (run.id,), match="fixed when the run is created")
    assert kit.runs.verify(run.id).ok


def test_a_run_with_no_evaluators_accepts_none_later(kit, store):
    run = make_run(kit)
    refuses(
        store,
        "INSERT INTO run_evaluators VALUES (?, 'x:y:000000000000', 'x', 'y', 1, '{}')",
        (run.id,),
        match="fixed when the run is created",
    )


def test_runs_cannot_be_deleted(kit, store):
    run = make_run(kit, EM)
    refuses(store, "DELETE FROM runs WHERE id = ?", (run.id,), match="permanent")
    assert kit.runs.get(run.id)


# --- lifecycle -------------------------------------------------------------------------------


def to(kit, run, status):
    kit.runs.plan(run.id)
    kit.runs.start(run.id)
    if status == "succeeded":
        for key in RUN_CASES:
            complete(kit, run, key)
        settle(kit, run)
        return kit.runs.transition(run.id, "succeeded")
    if status == "running":
        return kit.runs.get(run.id)
    return kit.runs.transition(run.id, status, stop_reason="because")


def test_the_lifecycle_of_a_successful_run(kit):
    run = make_run(kit, EM)
    assert kit.runs.plan(run.id) == 3
    running = kit.runs.start(run.id)
    assert running.status == "running" and running.started_at and not running.finished_at
    for key in RUN_CASES:
        complete(kit, run, key)
    settle(kit, run)
    done = kit.runs.transition(run.id, "succeeded")
    assert done.status == "succeeded" and done.finished_at >= done.started_at
    assert done.stop_reason is None


@pytest.mark.parametrize("stop", ["partial", "failed", "cancelled"])
def test_a_stopped_run_says_why_keeps_its_results_and_can_resume(kit, stop):
    run = make_run(kit, EM, start=True)
    complete(kit, run, "a")
    stopped = kit.runs.transition(run.id, stop, stop_reason="budget exhausted")
    assert (stopped.status, stopped.stop_reason) == (stop, "budget exhausted")
    assert stopped.finished_at is not None
    assert kit.runs.case_result(run.id, "a").status == "complete"  # nothing was lost
    resumed = kit.runs.start(run.id)
    assert resumed.status == "running" and resumed.stop_reason is None
    complete(kit, run, "b")  # and it accepts results again
    assert kit.runs.counts(run.id).complete == 2


def test_a_stop_reason_is_scrubbed(kit):
    run = make_run(kit, EM, start=True)
    stopped = kit.runs.transition(
        run.id, "failed", stop_reason="denied sk-abcdefghijklmnopqrstuvwx"
    )
    assert "sk-abcdef" not in stopped.stop_reason


@pytest.mark.parametrize("old,new", list(itertools.product(STATUSES, STATUSES)))
def test_every_status_pair_matches_the_state_machine(kit, store, old, new):
    """Table-driven over all 36 pairs, at the service and at the database."""
    run = make_run(kit, EM)
    if old != "created":
        to(kit, run, old)
    stop = "why" if new in ("partial", "failed", "cancelled") else None
    if new in TRANSITIONS[old]:
        if new == "succeeded":
            for key in RUN_CASES:
                complete(kit, run, key)
            settle(kit, run)
        assert kit.runs.transition(run.id, new, stop_reason=stop).status == new
        return
    with pytest.raises(RunError, match="cannot become"):
        kit.runs.transition(run.id, new, stop_reason=stop)
    # the database refuses the same illegal move, whatever the Python layer says
    now = "2026-01-01T00:00:00+00:00"
    finished = now if new in TERMINAL_STATUSES else None
    refuses(
        store,
        "UPDATE runs SET status = ?, stop_reason = ?, started_at = COALESCE(started_at, ?), "
        "finished_at = ? WHERE id = ?",
        (new, stop, now, finished, run.id),
    )
    assert kit.runs.get(run.id).status == old


def test_succeeded_is_terminal_and_created_cannot_stop_without_running(kit):
    run = make_run(kit, EM)
    for status in ("partial", "failed", "cancelled", "succeeded"):
        with pytest.raises(RunError, match="cannot become"):
            kit.runs.transition(run.id, status, stop_reason="x" if status != "succeeded" else None)


@pytest.mark.parametrize(
    "status,reason",
    [
        ("partial", None),
        ("failed", None),
        ("cancelled", None),
        ("running", "why"),
        ("succeeded", "why"),
    ],
)
def test_a_stop_reason_goes_with_exactly_the_stopped_statuses(kit, status, reason):
    run = make_run(kit, EM, start=True)
    with pytest.raises(RunError, match="stop_reason"):
        kit.runs.transition(run.id, status, stop_reason=reason)
    assert kit.runs.get(run.id).status == "running"


def test_unknown_status_is_refused(kit):
    with pytest.raises(RunError, match="unknown run status"):
        kit.runs.transition(make_run(kit, EM).id, "paused")


def test_a_run_cannot_succeed_while_cases_are_pending_or_unrecorded(kit, store):
    run = make_run(kit, EM, start=True)
    with pytest.raises(RunError, match=r"0 of 3"):
        kit.runs.transition(run.id, "succeeded")
    kit.runs.plan(run.id)
    complete(kit, run, "a")
    with pytest.raises(RunError, match=r"1 of 3"):  # two are still pending
        kit.runs.transition(run.id, "succeeded")
    complete(kit, run, "b")
    complete(kit, run, "c")
    settle(kit, run)
    assert kit.runs.transition(run.id, "succeeded").status == "succeeded"


def test_the_database_also_refuses_success_with_missing_results(kit, store):
    run = make_run(kit, EM, start=True)
    refuses(
        store,
        "UPDATE runs SET status = 'succeeded', finished_at = 't' WHERE id = ?",
        (run.id,),
        match="every case has a terminal result",
    )


def test_target_failures_count_as_terminal_for_success(kit):
    """A run whose target failed on some cases still *succeeded* at running: the failures are
    results, gated separately, not a reason the run did not finish."""
    run = make_run(kit, EM, start=True)
    complete(kit, run, "a")
    complete(kit, run, "b")
    kit.runs.record_case_result(
        run.id, "c", CaseOutcome.fail(EvalFailure("target", "exception", "boom"))
    )
    settle(kit, run)
    assert kit.runs.transition(run.id, "succeeded").status == "succeeded"


def test_check_constraints_tie_timestamps_and_reason_to_status(kit, store):
    """Even a raw UPDATE that names a legal status cannot leave it half-filled."""
    run = make_run(kit, EM, start=True)
    for sql in (
        "UPDATE runs SET status='partial', finished_at='t' WHERE id=?",  # no stop_reason
        "UPDATE runs SET status='partial', stop_reason='x' WHERE id=?",  # no finished_at
        "UPDATE runs SET status='succeeded', stop_reason='x', finished_at='t' WHERE id=?",
        "UPDATE runs SET status='succeeded' WHERE id=?",
    ):
        refuses(store, sql, (run.id,))
    assert kit.runs.get(run.id).status == "running"


# --- property: any sequence of transitions obeys the state machine ---------------------------


@settings(max_examples=60, deadline=None)
@given(steps=st.lists(st.sampled_from(STATUSES), max_size=8))
def test_property_random_transition_sequences_follow_the_state_machine(steps):
    from evalkit import EvalKit

    kit = EvalKit.open(":memory:")
    kit.datasets.import_cases("qa", [case("a", output="x")])
    run = kit.runs.create("qa", RunConfig())
    kit.runs.plan(run.id)
    model = "created"
    for step in steps:
        stop = "why" if step in ("partial", "failed", "cancelled") else None
        if step == "succeeded" and model == "running" and kit.runs.counts(run.id).pending:
            complete(kit, run, "a")  # success needs every case done; everything else is unchanged
        try:
            got = kit.runs.transition(run.id, step, stop_reason=stop).status
        except RunError:
            assert step not in TRANSITIONS[model]
            assert kit.runs.get(run.id).status == model
        else:
            assert step in TRANSITIONS[model] and got == step
            model = step
    kit.close()
