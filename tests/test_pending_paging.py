"""P1.1 (audit P1-14): paging the pending units of a run must cost the same at any depth.

The unhardened query ordered by `COALESCE(ord, 0), id`, which no index can serve, so every page
sorted every remaining pending row: quadratic over a run, with the store lock held throughout
(18 ms/page at 20K cases, 187 ms/page at 100K)."""

import pytest

from evalkit import EvalKit, EvaluatorSpec
from evalkit.targets import PrecomputedTarget


def make(tmp_path, n):
    kit = EvalKit.open(tmp_path / f"p{n}.db")
    kit.datasets.import_cases(
        "qa",
        (
            {"case_key": f"c{i:06}", "prompt": "p", "output": "o", "reference": "o"}
            for i in range(n)
        ),
    )
    run = kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[EvaluatorSpec(kind="exact_match", name="em")]
    )
    kit.runs.start(run.id)
    return kit, run


def vm_steps(conn, fn):
    """SQLite virtual-machine instructions executed while `fn` runs: a deterministic cost measure
    (wall-clock time would make this test flaky)."""
    steps = [0]

    def tick():
        steps[0] += 1
        return 0

    conn.set_progress_handler(tick, 100)
    try:
        fn()
    finally:
        conn.set_progress_handler(None, 0)
    return steps[0] * 100


def test_the_pending_page_query_is_served_by_the_index_without_a_sort(tmp_path):
    """Explain the query `next_pending` really runs (captured, not restated here)."""
    kit, run = make(tmp_path, 200)
    conn = kit.store._conn
    captured = []
    conn.set_trace_callback(captured.append)
    kit.store.next_pending(run.id, (5, "x"), 256)
    conn.set_trace_callback(None)
    (sql,) = [q for q in captured if "FROM case_results r" in q and "LIMIT" in q]
    plan = " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert "TEMP B-TREE" not in plan and "idx_case_results_claim" in plan, plan
    kit.close()


def measure_page(kit, run, after):
    return vm_steps(kit.store._conn, lambda: kit.store.next_pending(run.id, after, 256))


@pytest.mark.parametrize("position", ["start", "middle"])
def test_a_page_costs_the_same_whatever_the_size_of_the_run(tmp_path, position):
    """4x the pending rows must not mean ~4x the work per page (that is the quadratic bug)."""
    costs = {}
    for n in (2_000, 8_000):
        kit, run = make(tmp_path, n)
        after = None
        if position == "middle":  # page from the middle of the order, like a running run
            for _ in range(n // 256 // 2):
                page = kit.store.next_pending(run.id, after, 256)
                after = (page[-1][0], page[-1][1].result_id)
        costs[n] = measure_page(kit, run, after)
        kit.close()
    assert costs[8_000] < costs[2_000] * 1.5, costs


def test_paging_visits_every_pending_unit_exactly_once_in_order(tmp_path):
    kit, run = make(tmp_path, 1_000)
    after, seen = None, []
    while True:
        page = kit.store.next_pending(run.id, after, 128)
        seen += [(o, u.result_id) for o, u in page]
        if len(page) < 128:
            break
        after = (page[-1][0], page[-1][1].result_id)
    assert len(seen) == len(set(seen)) == 1_000 and seen == sorted(seen)
    kit.close()


def test_a_pending_result_without_an_order_cannot_exist(tmp_path):
    import sqlite3

    kit, run = make(tmp_path, 3)
    conn = kit.store._conn
    case_id = conn.execute("SELECT id FROM cases LIMIT 1").fetchone()[0]
    conn.execute("DELETE FROM case_results WHERE 0")  # (no-op: results are permanent records)
    with pytest.raises(sqlite3.IntegrityError, match="processing order"):
        conn.execute(
            "INSERT INTO case_results (id, run_id, case_id, status, created_at) "
            "VALUES ('z', ?, ?, 'pending', 't')",
            (run.id, case_id),
        )
    conn.rollback()
    kit.close()
