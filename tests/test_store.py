import sqlite3
import threading

import pytest

from evalkit import EvaluationResult, Review, Rubric
from evalkit.models import CriterionScore

RUBRIC = Rubric.from_dict({"a": "A?"})


def result(**overrides) -> EvaluationResult:
    fields = dict(
        status="ok",
        prompt="p",
        model_output="o",
        rubric=RUBRIC,
        rubric_version=RUBRIC.version,
        judge_provider="fake",
        judge_model="m",
        judge_temperature=0.0,
        judge_prompt_version="pv",
        scores={"a": CriterionScore(reasoning="r", score=4)},
        overall_score=0.75,
        verdict="PASS",
        latency_ms=12,
    )
    return EvaluationResult(**(fields | overrides))


def test_save_get_roundtrip(store):
    r = result(reference_output="ref", metadata={"app": "x", "n": 1}, tags=["t1", "t2"])
    store.save(r)
    assert store.get(r.id) == r


def test_context_roundtrips_and_defaults_to_none(store):
    with_context = result(context="supporting material")
    without_context = result()
    store.save(with_context)
    store.save(without_context)
    assert store.get(with_context.id).context == "supporting material"
    assert store.get(without_context.id).context is None


def test_error_row_roundtrip(store):
    r = result(
        status="error", error="JudgeError: boom", scores={}, overall_score=None, verdict=None
    )
    store.save(r)
    got = store.get(r.id)
    assert got.status == "error" and got.error == "JudgeError: boom"
    assert got.scores == {} and got.overall_score is None and got.verdict is None


def test_get_missing(store):
    assert store.get("nope") is None


def test_list_order_limit_and_tag(store):
    rs = [result(tags=["even"] if i % 2 == 0 else ["odd", "x"]) for i in range(5)]
    for r in rs:
        store.save(r)
    assert [r.id for r in store.list()] == [r.id for r in reversed(rs)]
    assert len(store.list(limit=2)) == 2
    assert {r.id for r in store.list(tag="even")} == {rs[0].id, rs[2].id, rs[4].id}
    assert store.list(tag="missing") == []


def test_multiple_reviews(store):
    r = result()
    store.save(r)
    first = Review(evaluation_id=r.id, reviewer="alice", verdict="PASS", score=0.8, comment="ok")
    second = Review(evaluation_id=r.id, reviewer="bob", verdict="FAIL")
    store.add_review(first)
    store.add_review(second)
    assert store.get(r.id).reviews == [first, second]
    assert store.list()[0].reviews == [first, second]  # list() also loads reviews


def test_list_reviews_scoped_per_evaluation(store):
    r1, r2 = result(), result()
    store.save(r1)
    store.save(r2)
    store.add_review(Review(evaluation_id=r1.id, reviewer="alice", verdict="PASS"))
    by_id = {r.id: r for r in store.list()}
    assert len(by_id[r1.id].reviews) == 1
    assert by_id[r2.id].reviews == []


def test_review_fk_backstop(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_review(Review(evaluation_id="missing", reviewer="a", verdict="PASS"))


def test_usable_from_multiple_threads(store):
    # a SQLiteStore is commonly shared across threads via one Evaluator (e.g. a thread pool);
    # a bare sqlite3 connection would raise ProgrammingError on the second thread's call
    results = [result() for _ in range(20)]
    errors = []

    def worker(r):
        try:
            store.save(r)
            fetched = store.get(r.id)
            if fetched is None or fetched.id != r.id:
                errors.append(f"roundtrip mismatch for {r.id}")
        except Exception as e:  # noqa: BLE001 - collecting failures across threads
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(r,)) for r in results]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert {r.id for r in store.list(limit=100)} == {r.id for r in results}


def test_persists_across_connections(tmp_path):
    from evalkit.store import SQLiteStore

    path = tmp_path / "nested" / "db.sqlite"
    s1 = SQLiteStore(path)
    r = result()
    s1.save(r)
    s1.close()
    s2 = SQLiteStore(path)
    assert s2.get(r.id) == r
    s2.close()
