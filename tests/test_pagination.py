"""F-12: `list(limit)` used to crash with "too many SQL variables" above ~32k; limits are now
validated and pages are walked with a keyset cursor."""

import json

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import EvalKitError, Evaluator, Review, Rubric
from evalkit.cli import main
from evalkit.limits import MAX_LIST_LIMIT

R = Rubric.from_dict({"a": "A?"})
_COLUMNS = (
    "id, created_at, status, prompt, model_output, rubric_json, rubric_version, judge_provider, "
    "judge_model, judge_temperature, judge_prompt_version, scores_json, overall_score, verdict, "
    "latency_ms, metadata_json, tags_json"
)


def _timestamp(i):
    return f"2026-01-01T00:{i // 3600 % 60:02d}:{i % 3600 // 60:02d}.{i % 60:02d}0000+00:00"


def bulk_insert(store, n, tag_every=None):
    """n coherent rows with strictly increasing created_at, inserted directly for speed."""
    rows = [
        (
            f"id-{i:06d}", _timestamp(i),
            "ok", "p", "o", R.model_dump_json(), R.version, "fake", "m", 0.0, "pv",
            '{"a": {"reasoning": "r", "score": 5, "label": null}}', 1.0, "PASS", 1, "{}",
            json.dumps(["even"] if tag_every and i % tag_every == 0 else []),
        )
        for i in range(n)
    ]  # fmt: skip
    marks = ",".join("?" * 17)
    with store._conn:
        store._conn.executemany(f"INSERT INTO evaluations ({_COLUMNS}) VALUES ({marks})", rows)


@pytest.mark.parametrize("limit", [0, -1, MAX_LIST_LIMIT + 1, 40_000, 10**9, True, 1.5, "5", None])
def test_invalid_limits_are_refused_with_a_clear_error(store, limit):
    with pytest.raises(EvalKitError, match="limit must be an integer between 1 and"):
        store.list(limit=limit)
    with pytest.raises(EvalKitError, match="limit must be"):
        Evaluator(None, store).list(limit=limit)


def test_the_original_f12_failure_is_now_a_clear_error_not_an_sqlite_crash(store):
    bulk_insert(store, 100)
    # before P0: sqlite3.OperationalError: too many SQL variables (on limit >= ~32k)
    with pytest.raises(EvalKitError, match="between 1 and 10000"):
        store.list(limit=40_000)


def test_the_maximum_limit_works_with_every_row_having_reviews_loaded(store):
    bulk_insert(store, MAX_LIST_LIMIT + 5)
    store.add_review(Review(evaluation_id="id-000000", reviewer="a", verdict="PASS"))
    page = store.list(limit=MAX_LIST_LIMIT)
    assert len(page) == MAX_LIST_LIMIT
    assert len({r.id for r in page}) == MAX_LIST_LIMIT


def test_walking_pages_with_a_cursor_returns_every_row_exactly_once_in_order(store):
    bulk_insert(store, 2500)
    seen, cursor, pages = [], None, 0
    while True:
        rows, cursor = store.list_page(limit=1000, cursor=cursor)
        seen += [r.id for r in rows]
        pages += 1
        if cursor is None:
            break
    assert pages == 3 and len(seen) == len(set(seen)) == 2500
    assert seen == sorted(seen, reverse=True)  # newest first


def test_exact_multiple_of_the_page_size_has_no_phantom_last_page(store):
    bulk_insert(store, 2000)
    rows, cursor = store.list_page(limit=1000)
    assert len(rows) == 1000 and cursor is not None
    rows, cursor = store.list_page(limit=1000, cursor=cursor)
    assert len(rows) == 1000 and cursor is None
    assert store.list_page(limit=5000)[1] is None


def test_an_empty_store_and_a_short_page_have_no_cursor(store):
    assert store.list_page() == ([], None)
    bulk_insert(store, 3)
    rows, cursor = store.list_page(limit=10)
    assert len(rows) == 3 and cursor is None


def test_tag_filter_composes_with_the_cursor(store):
    bulk_insert(store, 100, tag_every=4)
    ids, cursor = [], None
    while True:
        rows, cursor = store.list_page(tag="even", limit=10, cursor=cursor)
        ids += [r.id for r in rows]
        if cursor is None:
            break
    assert ids == [f"id-{i:06d}" for i in range(96, -1, -4)]


def test_new_rows_do_not_shift_or_duplicate_pages_already_being_walked(store):
    bulk_insert(store, 30)
    first, cursor = store.list_page(limit=10)
    Evaluator(FakeJudge(judged(a=5)), store).evaluate(
        "p", "o", criteria={"a": "A?"}
    )  # newest row arrives mid-walk
    second, _ = store.list_page(limit=10, cursor=cursor)
    assert not {r.id for r in first} & {r.id for r in second}
    assert [r.id for r in second] == [f"id-{i:06d}" for i in range(19, 9, -1)]


@pytest.mark.parametrize("cursor", ["", "not-base64!!", "e30", "W10", "WyJ4IiwgInkiXQ", "WzEsMl0"])
def test_malformed_cursors_are_refused(store, cursor):
    with pytest.raises(EvalKitError, match="invalid cursor"):
        store.list_page(cursor=cursor)


def test_reviews_are_attached_correctly_across_the_in_chunk_boundary(store, monkeypatch):
    monkeypatch.setattr("evalkit.store._IN_CHUNK", 3)  # force several IN (...) chunks
    ev = Evaluator(FakeJudge(*[judged(correctness=4, clarity=4)] * 10), store)
    ids = [ev.evaluate("p", "o", criteria=CRITERIA).id for _ in range(10)]
    for i, evaluation_id in enumerate(ids):
        for j in range(i % 3):
            store.add_review(Review(evaluation_id=evaluation_id, reviewer=f"r{j}", verdict="PASS"))
    by_id = {r.id: r for r in store.list(limit=10)}
    for i, evaluation_id in enumerate(ids):
        assert [r.reviewer for r in by_id[evaluation_id].reviews] == [f"r{j}" for j in range(i % 3)]


def test_evaluator_list_page_requires_a_store_that_supports_it(store):
    rows, cursor = Evaluator(None, store).list_page(limit=5)
    assert rows == [] and cursor is None

    class Bare:
        def list(self, tag=None, limit=20):
            return []

    from evalkit import ConfigError

    with pytest.raises(ConfigError, match="pagination"):
        Evaluator(None, Bare()).list_page()


def test_cli_list_pages_with_a_cursor(monkeypatch, store, capsys):
    bulk_insert(store, 5)
    monkeypatch.setattr(
        Evaluator, "from_env", classmethod(lambda cls, with_judge=True: cls(None, store))
    )
    assert main(["list", "--limit", "2"]) == 0
    out, err = capsys.readouterr()
    assert [r["id"] for r in json.loads(out)] == ["id-000004", "id-000003"]
    cursor = err.strip().split("next_cursor: ")[1]

    assert main(["list", "--limit", "2", "--cursor", cursor]) == 0
    out, err = capsys.readouterr()
    assert [r["id"] for r in json.loads(out)] == ["id-000002", "id-000001"]

    assert main(["list", "--cursor", "garbage"]) == 1
    assert "invalid cursor" in capsys.readouterr().err
    assert main(["list", "--limit", "99999"]) == 1
