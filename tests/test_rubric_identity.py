"""F-7: a rubric version label can never silently refer to different rubric content."""

import threading

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import Criterion, Evaluator, Rubric, RubricError
from evalkit.models import CriterionScore, EvaluationResult
from evalkit.store import SQLiteStore


def rubric(description="Is it correct?", version="qa-v1", **kw):
    return Rubric(criteria=[Criterion(name="a", description=description, **kw)], version=version)


def test_auto_versions_are_unchanged_by_p0():
    """Pinned from the pre-P0 code: existing stored version labels must stay valid."""
    assert Rubric.from_dict({"a": "A?"}).version == "89e795732810"
    assert (
        Rubric.from_dict(
            {"correctness": "Is it factually correct?", "clarity": "Is it clear?"}
        ).version
        == "0da0fb9a41a8"
    )
    mixed = Rubric(
        criteria=[
            Criterion(name="f", description="d", scale=(0, 10), weight=2.5),
            Criterion(name="s", description="d2", labels=("BAD", "OK", "GOOD")),
        ],
        threshold=0.8,
    )
    assert mixed.version == "eb83eadd2b01"


def test_auto_version_is_the_prefix_of_the_full_content_hash():
    r = Rubric.from_dict({"a": "A?"})
    assert len(r.content_hash) == 64 and r.content_hash.startswith(r.version)


def test_content_hash_ignores_version_but_reflects_every_content_change():
    base = rubric()
    assert rubric(version="other-label").content_hash == base.content_hash
    variants = [
        rubric(description="Is it correct??"),
        rubric(scale=(0, 5)),
        rubric(weight=2),
        rubric(labels=("BAD", "GOOD")),
        Rubric(criteria=base.criteria, threshold=0.5, version="qa-v1"),
        Rubric(criteria=[*base.criteria, Criterion(name="b", description="B?")], version="qa-v1"),
    ]
    hashes = {v.content_hash for v in variants} | {base.content_hash}
    assert len(hashes) == len(variants) + 1
    # ordinal label order is content: worst -> best
    a = rubric(labels=("BAD", "OK", "GOOD"))
    b = rubric(labels=("GOOD", "OK", "BAD"))
    assert a.content_hash != b.content_hash


def test_same_version_with_different_content_is_refused_before_the_judge_is_called(store):
    ev = Evaluator(FakeJudge(judged(a=5), judged(a=5)), store)
    ev.evaluate("p", "o", rubric=rubric())

    judge = FakeJudge(judged(a=5))
    with pytest.raises(RubricError, match="qa-v1.*different rubric content"):
        Evaluator(judge, store).evaluate("p", "o", rubric=rubric(description="Is it kind?"))
    assert judge.calls == []  # no paid call
    assert len(store.list()) == 1  # nothing persisted for the refused evaluation


def test_same_version_with_same_content_is_fine_and_survives_reopening(tmp_path):
    path = tmp_path / "e.db"
    s1 = SQLiteStore(path)
    Evaluator(FakeJudge(judged(a=5)), s1).evaluate("p", "o", rubric=rubric())
    s1.close()
    s2 = SQLiteStore(path)
    Evaluator(FakeJudge(judged(a=4)), s2).evaluate("p2", "o2", rubric=rubric())
    with pytest.raises(RubricError):
        Evaluator(FakeJudge(judged(a=5)), s2).evaluate("p", "o", rubric=rubric(scale=(0, 10)))
    assert len(s2.list()) == 2


def test_different_versions_may_have_different_content(store):
    ev = Evaluator(FakeJudge(judged(a=5), judged(a=5)), store)
    ev.evaluate("p", "o", rubric=rubric(version="qa-v1"))
    ev.evaluate("p", "o", rubric=rubric(description="Is it kind?", version="qa-v2"))
    assert len(store.list()) == 2


def test_criteria_dicts_get_content_addressed_versions_that_cannot_collide(store):
    ev = Evaluator(FakeJudge(judged(a=5), judged(a=5)), store)
    r1 = ev.evaluate("p", "o", criteria={"a": "Is it correct?"})
    r2 = ev.evaluate("p", "o", criteria={"a": "Is it kind?"})
    assert r1.rubric_version != r2.rubric_version


def test_save_itself_enforces_the_binding_even_when_called_directly(store):
    """Defense in depth: a caller that bypasses Evaluator cannot poison the registry."""

    def result_for(r):
        return EvaluationResult(
            status="ok", prompt="p", model_output="o", rubric=r, rubric_version=r.version,
            judge_provider="f", judge_model="m", judge_temperature=0.0, judge_prompt_version="v",
            scores={"a": CriterionScore(reasoning="r", score=4)},
            overall_score=0.75, verdict="PASS",
        )  # fmt: skip

    store.save(result_for(rubric()))
    conflicting = result_for(rubric(description="Something else"))
    with pytest.raises(RubricError):
        store.save(conflicting)
    assert store.get(conflicting.id) is None  # rolled back: no row
    (stored,) = store._conn.execute("SELECT content_hash FROM rubric_versions")
    assert stored[0] == rubric().content_hash  # registry unchanged


def test_registration_is_atomic_under_concurrent_registration(store):
    hashes = [f"{i:064x}" for i in range(8)]
    outcomes: list[str] = []
    barrier = threading.Barrier(len(hashes))

    def register(content_hash):
        barrier.wait()
        try:
            store.register_rubric("contested", content_hash)
            outcomes.append("won")
        except RubricError:
            outcomes.append("lost")

    threads = [threading.Thread(target=register, args=(h,)) for h in hashes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("won") == 1 and outcomes.count("lost") == len(hashes) - 1


@pytest.mark.parametrize("bad", ["", "x" * 129, "two\nlines"])
def test_version_label_must_be_a_short_printable_line(bad):
    with pytest.raises(ValueError):
        rubric(version=bad)


def test_a_store_without_a_registry_still_works():
    """Custom Store implementations (the documented Protocol) need not know about registries."""

    class MinimalStore:
        def __init__(self):
            self.saved = []

        def save(self, result):
            self.saved.append(result)

        def get(self, evaluation_id):
            return None

        def list(self, tag=None, limit=20):
            return []

        def add_review(self, review):
            pass

    store = MinimalStore()
    result = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store).evaluate(
        "p", "o", criteria=CRITERIA
    )
    assert store.saved == [result]
