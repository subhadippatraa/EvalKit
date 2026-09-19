import pytest
from conftest import judged
from pydantic import ValidationError

from evalkit import Criterion, JudgeOutputError, Review, Rubric


def rubric(*criteria: Criterion, threshold: float = 0.75) -> Rubric:
    return Rubric(criteria=list(criteria), threshold=threshold)


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "description": "d"},
        {"name": "n", "description": ""},
        {"name": "n", "description": "d", "scale": (5, 1)},
        {"name": "n", "description": "d", "scale": (3, 3)},
        {"name": "n", "description": "d", "weight": 0},
        {"name": "n", "description": "d", "weight": -1},
        {"name": "has space", "description": "d"},
        {"name": "has/slash", "description": "d"},
        {"name": "a" * 65, "description": "d"},  # over the 64-char tool-schema name limit
    ],
)
def test_invalid_criterion(kwargs):
    with pytest.raises(ValidationError):
        Criterion(**kwargs)


def test_criterion_name_allows_letters_digits_underscore_hyphen():
    for name in ["a", "correctness", "factual_accuracy", "step-1", "A1_b-2", "a" * 64]:
        assert Criterion(name=name, description="d").name == name


def test_rubric_rejects_empty_duplicates_and_bad_threshold():
    c = Criterion(name="a", description="d")
    with pytest.raises(ValidationError):
        Rubric(criteria=[])
    with pytest.raises(ValidationError, match="duplicate"):
        Rubric(criteria=[c, c])
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            Rubric(criteria=[c], threshold=bad)


def test_from_dict_defaults():
    r = Rubric.from_dict({"a": "A?", "b": "B?"})
    assert [c.name for c in r.criteria] == ["a", "b"]
    assert all(c.scale == (1, 5) and c.weight == 1.0 for c in r.criteria)
    assert r.threshold == 0.75


def test_version_is_content_hash_or_explicit():
    a1, a2 = Rubric.from_dict({"a": "A?"}), Rubric.from_dict({"a": "A?"})
    assert a1.version == a2.version and len(a1.version) == 12
    assert Rubric.from_dict({"a": "changed"}).version != a1.version
    assert Rubric(criteria=a1.criteria, threshold=0.5).version != a1.version
    assert Rubric(criteria=a1.criteria, version="v7").version == "v7"
    # round-trips keep the stored version
    assert Rubric.model_validate(a1.model_dump()).version == a1.version


def test_review_score_range():
    Review(evaluation_id="e", reviewer="r", verdict="PASS", score=0)
    Review(evaluation_id="e", reviewer="r", verdict="FAIL", score=1)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValidationError):
            Review(evaluation_id="e", reviewer="r", verdict="PASS", score=bad)
    with pytest.raises(ValidationError):
        Review(evaluation_id="e", reviewer="r", verdict="MAYBE")


# --- scoring ------------------------------------------------------------------


def test_normalization_equal_weights():
    r = Rubric.from_dict({"a": "A", "b": "B"})
    scores, overall, verdict = r.score(judged(a=5, b=3))  # 1.0 and 0.5
    assert overall == 0.75 and verdict == "PASS"
    assert scores["a"].score == 5 and scores["a"].reasoning


def test_extremes():
    r = Rubric.from_dict({"a": "A"})
    assert r.score(judged(a=1))[1:] == (0.0, "FAIL")
    assert r.score(judged(a=5))[1:] == (1.0, "PASS")


def test_weights():
    r = rubric(
        Criterion(name="a", description="A", weight=3),
        Criterion(name="b", description="B", weight=1),
    )
    _, overall, _ = r.score(judged(a=5, b=1))  # (3*1 + 1*0) / 4
    assert overall == 0.75


def test_mixed_scales():
    r = rubric(
        Criterion(name="binary", description="B", scale=(0, 1)),
        Criterion(name="ten", description="T", scale=(0, 10)),
    )
    _, overall, _ = r.score(judged(binary=1, ten=4))  # (1 + 0.4) / 2
    assert overall == pytest.approx(0.7)


def test_threshold_boundary():
    r = Rubric.from_dict({"a": "A", "b": "B", "c": "C"})
    assert r.score(judged(a=4, b=4, c=4))[1:] == (0.75, "PASS")  # exactly at threshold
    _, overall, verdict = r.score(judged(a=4, b=4, c=3))
    assert overall < 0.75 and verdict == "FAIL"


# --- malformed judge output ---------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        "text",
        judged(a=4),  # missing b
        judged(a=4, b=4, c=4),  # unexpected c
        {"a": {"reasoning": "x", "score": 4}, "b": "not an object"},
        {"a": {"reasoning": "x", "score": 4}, "b": {"score": 4}},  # missing reasoning
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "", "score": 4}},
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x"}},  # missing score
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x", "score": "4"}},
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x", "score": 4.0}},
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x", "score": 4.5}},
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x", "score": True}},
        {"a": {"reasoning": "x", "score": 4}, "b": {"reasoning": "x", "score": 4, "x": 1}},
        judged(a=4, b=0),  # below scale
        judged(a=4, b=6),  # above scale
    ],
)
def test_malformed_output_rejected(raw):
    with pytest.raises(JudgeOutputError):
        Rubric.from_dict({"a": "A", "b": "B"}).score(raw)
