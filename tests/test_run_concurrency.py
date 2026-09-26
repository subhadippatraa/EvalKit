"""Concurrent access: threads sharing one store, and separate connections to one file."""

import threading
from collections import Counter

import pytest
from conftest import EM, JUDGE, case, ok, settle

from evalkit import (
    CaseOutcome,
    DuplicateResultError,
    EvalKit,
    EvaluatorOutcome,
    RunConfig,
    RunError,
)
from evalkit.store import SQLiteStore

N = 16


def race(fns):
    """Run callables at (about) the same moment; returns [(result | None, exception | None)]."""
    barrier = threading.Barrier(len(fns))
    out = [None] * len(fns)

    def go(i, fn):
        barrier.wait()
        try:
            out[i] = (fn(), None)
        except BaseException as e:
            out[i] = (None, e)

    threads = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


@pytest.fixture
def big(kit):
    kit.datasets.import_cases("qa", [case(f"c{i:03}", output="o") for i in range(N)])
    run = kit.runs.create("qa", RunConfig(evaluators=[EM, JUDGE]))
    kit.runs.start(run.id)
    return run


def test_threads_recording_different_cases_lose_nothing(kit, big):
    results = race(
        [
            lambda i=i: kit.runs.record_case_result(
                big.id, f"c{i:03}", CaseOutcome.complete(f"o{i}")
            )
            for i in range(N)
        ]
    )
    assert all(e is None for _, e in results)
    counts = kit.runs.counts(big.id)
    assert (counts.complete, counts.pending, counts.missing) == (N, 0, 0)
    assert {r.output for r in kit.runs.case_results(big.id)} == {f"o{i}" for i in range(N)}
    assert kit.runs.verify(big.id).ok


def test_threads_racing_for_one_case_result_exactly_one_wins(kit, big):
    results = race(
        [
            lambda i=i: kit.runs.record_case_result(big.id, "c000", CaseOutcome.complete(f"w{i}"))
            for i in range(N)
        ]
    )
    winners = [r for r, e in results if e is None]
    errors = [e for _, e in results if e is not None]
    assert len(winners) == 1 and all(isinstance(e, DuplicateResultError) for e in errors)
    assert kit.runs.case_result(big.id, "c000").output == winners[0].output
    assert kit.runs.counts(big.id).complete == 1


def test_threads_racing_for_one_evaluator_result_exactly_one_wins(kit, big):
    cr = kit.runs.record_case_result(big.id, "c000", CaseOutcome.complete("o"))
    results = race(
        [lambda i=i: kit.runs.record_evaluator_result(cr.id, ok(EM.key, i / N)) for i in range(N)]
    )
    assert Counter(type(e).__name__ for _, e in results if e) == {"DuplicateResultError": N - 1}
    (only,) = kit.runs.evaluator_results(cr.id)
    assert [r.score for r, e in results if e is None] == [only.score]


def test_concurrent_creates_with_one_idempotency_key_make_one_run(kit):
    kit.datasets.import_cases("qa", [case("a")])
    results = race(
        [
            lambda: kit.runs.create("qa", RunConfig(evaluators=[EM]), idempotency_key="nightly")
            for _ in range(N)
        ]
    )
    assert all(e is None for _, e in results)
    assert len({r.id for r, _ in results}) == 1
    assert kit.store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_racing_lifecycle_transitions_only_one_finishes_the_run(kit, big):
    kit.runs.plan(big.id)
    for i in range(N):
        kit.runs.record_case_result(big.id, f"c{i:03}", CaseOutcome.complete("o"))
    settle(kit, big)
    results = race([lambda: kit.runs.transition(big.id, "succeeded") for _ in range(8)])
    assert sum(e is None for _, e in results) == 1
    assert all(isinstance(e, RunError) for _, e in results if e)
    assert kit.runs.get(big.id).status == "succeeded"


def test_a_result_racing_with_the_run_ending_is_either_stored_or_refused_never_half(kit, big):
    def stop():
        return kit.runs.transition(big.id, "partial", stop_reason="deadline")

    fns = [stop] + [
        lambda i=i: kit.runs.record_case_result(big.id, f"c{i:03}", CaseOutcome.complete("o"))
        for i in range(N)
    ]
    results = race(fns)
    stored = sum(1 for r, e in results[1:] if e is None)
    assert all(isinstance(e, RunError) for _, e in results[1:] if e)
    assert kit.runs.counts(big.id).complete == stored
    assert kit.runs.get(big.id).status == "partial"
    assert kit.runs.verify(big.id).ok


def test_readers_never_see_a_torn_result(kit, big):
    """A result and its metrics are one transaction: a concurrent reader sees both or neither."""
    kit.runs.plan(big.id)
    for i in range(N):
        kit.runs.record_case_result(big.id, f"c{i:03}", CaseOutcome.complete("o"))
    stop = threading.Event()
    torn = []

    def reader():
        while not stop.is_set():
            rows = kit.store._conn  # same connection, but each store call takes the lock
            with kit.store._lock:
                bad = rows.execute(
                    "SELECT COUNT(*) FROM evaluator_results e WHERE e.status = 'ok' AND NOT EXISTS "
                    "(SELECT 1 FROM metrics m WHERE m.evaluator_result_id = e.id)"
                ).fetchone()[0]
            if bad:
                torn.append(bad)

    t = threading.Thread(target=reader)
    t.start()
    crs = list(kit.runs.case_results(big.id))
    race(
        [
            lambda cr=cr: kit.runs.record_evaluator_result(
                cr.id,
                EvaluatorOutcome(
                    evaluator_key=EM.key, status="ok", metrics={"score": 1.0, "a": 2.0, "b": 3.0}
                ),
            )
            for cr in crs
        ]
    )
    stop.set()
    t.join()
    assert torn == [] and kit.runs.verify(big.id).ok


def test_separate_connections_to_one_file_also_serialize_correctly(tmp_path):
    """Two stores = two connections (like two processes): BEGIN IMMEDIATE + WAL arbitrate."""
    path = tmp_path / "shared.db"
    a, b = EvalKit.open(path), EvalKit.open(path)
    a.datasets.import_cases("qa", [case(f"c{i}", output="o") for i in range(8)])
    run = a.runs.create("qa", RunConfig(evaluators=[EM]))
    a.runs.start(run.id)
    kits = [a, b] * 8
    results = race(
        [
            lambda k=k, i=i: k.runs.record_case_result(
                run.id, f"c{i % 8}", CaseOutcome.complete(f"w{i}")
            )
            for i, k in enumerate(kits)
        ]
    )
    per_case = Counter()
    for i, (_, e) in enumerate(results):
        if e is None:
            per_case[f"c{i % 8}"] += 1
        else:
            assert isinstance(e, DuplicateResultError), e
    assert dict(per_case) == {f"c{i}": 1 for i in range(8)}  # one winner per case, across processes
    assert b.runs.counts(run.id).complete == 8 and b.runs.verify(run.id).ok
    a.close()
    b.close()


def test_two_stores_see_each_others_committed_runs(tmp_path):
    path = tmp_path / "shared.db"
    a, b = SQLiteStore(path), SQLiteStore(path)
    ka, kb = EvalKit(a), EvalKit(b)
    ka.datasets.import_cases("qa", [case("a")])
    run = ka.runs.create("qa", RunConfig())
    assert kb.runs.get(run.id) == run
    kb.runs.start(run.id)
    assert ka.runs.get(run.id).status == "running"
    a.close()
    b.close()
