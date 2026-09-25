"""F-1: NaN/Infinity/extreme numbers can never become an "ok" result.

Guarantee under test: Rubric.score() either returns a finite overall score in [0, 1] with a
verdict that follows from it, or raises a typed error. Nothing else escapes.
"""

import math
import sqlite3
from fractions import Fraction

import pytest
from conftest import CRITERIA, FakeJudge, judged
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from evalkit import (
    Criterion,
    EvaluationResult,
    Evaluator,
    JudgeOutputError,
    Review,
    Rubric,
    RubricError,
    ScoringError,
)
from evalkit.limits import MAX_WEIGHT, MIN_WEIGHT
from evalkit.models import CriterionScore

PROPERTY = settings(max_examples=300, deadline=None)

any_float = st.floats(allow_nan=True, allow_infinity=True)  # includes nan, +-inf, subnormals, 1e308
BAD_NUMBERS = [float("nan"), float("inf"), float("-inf")]


# --- construction rejects non-finite / out-of-range numbers -------------------------------


@pytest.mark.parametrize(
    "weight", [*BAD_NUMBERS, 0, -1, 1e-7, MIN_WEIGHT / 10, 1e308, MAX_WEIGHT * 1.0001]
)
def test_criterion_rejects_bad_weights(weight):
    with pytest.raises(ValidationError):
        Criterion(name="a", description="d", weight=weight)


@pytest.mark.parametrize("weight", [MIN_WEIGHT, 1.0, 2.5, MAX_WEIGHT])
def test_criterion_accepts_bounded_weights(weight):
    assert Criterion(name="a", description="d", weight=weight).weight == weight


@pytest.mark.parametrize("threshold", [*BAD_NUMBERS, -0.1, 1.1])
def test_rubric_rejects_bad_threshold(threshold):
    with pytest.raises(ValidationError):
        Rubric(criteria=[Criterion(name="a", description="d")], threshold=threshold)


@pytest.mark.parametrize("score", [*BAD_NUMBERS, -0.1, 1.1])
def test_review_rejects_bad_score(score):
    with pytest.raises(ValidationError):
        Review(evaluation_id="e", reviewer="a", verdict="PASS", score=score)


def test_original_f1_reproduction_is_now_rejected_before_the_judge_is_called(store):
    """Two weights of 1e308 made overall = inf/inf = NaN, persisted as ok/NULL/FAIL (audit F-1)."""
    rubric = {
        "criteria": [
            {"name": "a", "description": "d", "weight": 1e308},
            {"name": "b", "description": "d", "weight": 1e308},
        ]
    }
    judge = FakeJudge(judged(a=5, b=5))
    with pytest.raises(RubricError, match="weight"):
        Evaluator(judge, store).evaluate("p", "o", rubric=rubric)
    assert judge.calls == [] and store.list() == []


def test_metadata_with_nan_is_rejected(store):
    with pytest.raises(RubricError, match="metadata"):
        Evaluator(FakeJudge(), store).evaluate(
            "p", "o", criteria=CRITERIA, metadata={"x": float("nan")}
        )


# --- Rubric.score: finite, in range, or a typed error -------------------------------------


def _payload(rubric, score_for):
    return {c.name: {"reasoning": "r", "score": score_for(c)} for c in rubric.criteria}


def test_score_defends_itself_against_a_non_finite_result(monkeypatch):
    """Unreachable with validated inputs; if a future change reached it, it must raise."""
    rubric = Rubric.from_dict({"a": "A?"})
    monkeypatch.setattr("evalkit.models.math.fsum", lambda _values: float("nan"))
    with pytest.raises(ScoringError, match="finite"):
        rubric.score(_payload(rubric, lambda c: 5))


def test_scoring_error_is_persisted_as_error_row_and_not_retried(store, monkeypatch):
    monkeypatch.setattr("evalkit.models.math.fsum", lambda _values: float("inf"))
    judge = FakeJudge(judged(correctness=5, clarity=5), judged(correctness=5, clarity=5))
    with pytest.raises(ScoringError) as info:
        Evaluator(judge, store).evaluate("p", "o", criteria=CRITERIA)
    assert len(judge.calls) == 1  # an internal fault is not the judge's, so it is not retried
    (row,) = store.list()
    assert row.status == "error" and row.overall_score is None and row.verdict is None
    assert info.value.evaluation_id == row.id
    assert row.attempts[0].outcome == "scoring_error"


@PROPERTY
@given(weights=st.lists(any_float, min_size=1, max_size=6), data=st.data())
def test_property_arbitrary_weights_never_yield_a_bad_score(weights, data):
    try:
        rubric = Rubric(
            criteria=[
                Criterion(name=f"c{i}", description="d", scale=(0, 10), weight=w)
                for i, w in enumerate(weights)
            ]
        )
    except ValidationError:
        # rejection is only ever for a genuinely bad weight
        assert any(not (math.isfinite(w) and MIN_WEIGHT <= w <= MAX_WEIGHT) for w in weights)
        return
    scores = [data.draw(st.integers(0, 10)) for _ in weights]
    raw = {f"c{i}": {"reasoning": "r", "score": s} for i, s in enumerate(scores)}
    _, overall, verdict = rubric.score(raw)

    assert math.isfinite(overall) and 0.0 <= overall <= 1.0
    assert verdict == ("PASS" if overall >= rubric.threshold else "FAIL")
    exact = sum(Fraction(w) * Fraction(s, 10) for w, s in zip(weights, scores, strict=True)) / sum(
        Fraction(w) for w in weights
    )
    assert abs(Fraction(overall) - exact) < Fraction(1, 10**9)  # no silent precision loss


_json_scalar = st.none() | st.booleans() | st.integers() | any_float | st.text(max_size=8)
_json_like = st.recursive(
    _json_scalar,
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=4), inner, max_size=3)
    ),
    max_leaves=12,
)
_MIXED = Rubric(
    criteria=[
        Criterion(name="num", description="d", scale=(1, 5), weight=3),
        Criterion(name="cat", description="d", labels=("BAD", "OK", "GOOD")),
    ],
    threshold=0.6,
)


def _assert_valid_or_typed_error(raw):
    try:
        scores, overall, verdict = _MIXED.score(raw)
    except JudgeOutputError:
        return  # rejected with the documented typed error
    assert math.isfinite(overall) and 0.0 <= overall <= 1.0
    assert verdict == ("PASS" if overall >= _MIXED.threshold else "FAIL")
    assert set(scores) == {"num", "cat"}


@PROPERTY
@given(raw=_json_like)
def test_property_arbitrary_json_is_scored_or_rejected_with_a_typed_error(raw):
    _assert_valid_or_typed_error(raw)


_criterion_value = _json_scalar | st.sampled_from(["BAD", "OK", "GOOD", 1, 5, 4.0])


@PROPERTY
@given(
    num=st.fixed_dictionaries(
        {"reasoning": st.text(max_size=5) | st.none(), "score": _criterion_value}
    ),
    cat=st.fixed_dictionaries({"reasoning": st.just("r"), "label": _criterion_value}),
    extra=st.booleans(),
)
def test_property_structured_payloads_with_hostile_values(num, cat, extra):
    raw = {"num": num, "cat": cat}
    if extra:
        raw["surprise"] = 1
    _assert_valid_or_typed_error(raw)


@PROPERTY
@given(
    lo=st.integers(-(10**6), 10**6),
    span=st.integers(1, 10**6),
    score=st.integers(min_value=-(10**400), max_value=10**400) | st.integers(-(10**7), 10**7),
)
def test_property_huge_integers_never_overflow(lo, span, score):
    try:
        rubric = Rubric(criteria=[Criterion(name="a", description="d", scale=(lo, lo + span))])
    except ValidationError:
        assert max(abs(lo), abs(lo + span)) > 10**6  # rejected only for exceeding the bound
        return
    try:
        _, overall, _ = rubric.score({"a": {"reasoning": "r", "score": score}})
    except JudgeOutputError:
        assert not lo <= score <= lo + span
        return
    assert lo <= score <= lo + span and 0.0 <= overall <= 1.0


# --- coherence of stored/returned results --------------------------------------------------

_RUBRIC = Rubric.from_dict({"a": "A?"})
_OK = dict(
    status="ok",
    prompt="p",
    model_output="o",
    rubric=_RUBRIC,
    rubric_version=_RUBRIC.version,
    judge_provider="f",
    judge_model="m",
    judge_temperature=0.0,
    judge_prompt_version="v",
    scores={"a": CriterionScore(reasoning="r", score=4)},
    overall_score=0.75,
    verdict="PASS",
)


def _result(**overrides):
    return EvaluationResult(**(_OK | overrides))


def test_a_coherent_result_is_accepted():
    assert _result().verdict == "PASS"


@pytest.mark.parametrize(
    "overrides",
    [
        {"overall_score": None},  # the F-1 shape: ok without a score
        {"verdict": None},
        {"overall_score": float("nan")},
        {"overall_score": float("inf")},
        {"overall_score": 1.5},
        {"overall_score": -0.1},
        {"overall_score": 0.75, "verdict": "FAIL"},  # contradicts threshold 0.75
        {"overall_score": 0.7, "verdict": "PASS"},
        {"scores": {}},
        {"scores": {"other": CriterionScore(reasoning="r", score=4)}},
        {"error": "boom"},  # ok with an error
        {"rubric_version": "not-the-rubric-version"},
        {"judge_temperature": float("nan")},
        {"latency_ms": -1},
    ],
)
def test_incoherent_ok_results_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"error": None},
        {"overall_score": 0.5},
        {"verdict": "FAIL"},
        {"scores": {"a": CriterionScore(reasoning="r", score=4)}},
    ],
)
def test_incoherent_error_results_are_rejected(overrides):
    base = dict(
        _OK, status="error", error="JudgeError: x", scores={}, overall_score=None, verdict=None
    )
    with pytest.raises(ValidationError):
        EvaluationResult(**(base | overrides))


# --- the database refuses incoherent rows even if Python were bypassed ---------------------

_ROW = dict(
    id="x",
    created_at="2026-01-01T00:00:00+00:00",
    status="ok",
    error=None,
    prompt="p",
    model_output="o",
    reference_output=None,
    context=None,
    rubric_json=_RUBRIC.model_dump_json(),
    rubric_version=_RUBRIC.version,
    judge_provider="f",
    judge_model="m",
    judge_temperature=0.0,
    judge_prompt_version="v",
    scores_json='{"a": {"reasoning": "r", "score": 4, "label": null}}',
    overall_score=0.75,
    verdict="PASS",
    latency_ms=1,
    metadata_json="{}",
    tags_json="[]",
)


def _insert(store, **overrides):
    row = _ROW | overrides
    store._conn.execute(
        f"INSERT INTO evaluations ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})",
        row,
    )


def test_db_accepts_a_coherent_row(store):
    _insert(store)
    assert store.get("x").verdict == "PASS"


@pytest.mark.parametrize(
    "overrides",
    [
        {"overall_score": None},  # what SQLite makes of NaN -- exactly the F-1 row
        {"overall_score": float("nan")},  # binds as NULL
        {"overall_score": float("inf")},
        {"overall_score": 1.5},
        {"overall_score": -1.0},
        {"verdict": None},
        {"scores_json": None},
        {"error": "boom"},
        {"status": "error", "error": None, "overall_score": None, "verdict": None},
        {"status": "error", "error": "boom"},  # error row that still carries a score
    ],
)
def test_db_rejects_incoherent_rows(store, overrides):
    with pytest.raises(sqlite3.IntegrityError, match="incoherent"):
        _insert(store, **overrides)


def test_db_accepts_a_coherent_error_row(store):
    _insert(
        store,
        status="error",
        error="JudgeError: x",
        overall_score=None,
        verdict=None,
        scores_json=None,
    )
    assert store.get("x").status == "error"


def test_a_corrupt_legacy_row_is_reported_not_silently_returned(store):
    # simulate a row written by the pre-P0 bug, bypassing the trigger by dropping it
    store._conn.execute("DROP TRIGGER evaluations_coherent_insert")
    _insert(store, overall_score=None)
    from evalkit import EvalKitError

    with pytest.raises(EvalKitError, match="corrupt or incoherent"):
        store.get("x")
    with pytest.raises(EvalKitError, match="corrupt or incoherent"):
        store.list()


# --- non-finite configuration --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EVALKIT_JUDGE_TEMPERATURE", "nan"),
        ("EVALKIT_JUDGE_TEMPERATURE", "inf"),
        ("EVALKIT_JUDGE_TEMPERATURE", "-0.5"),
        ("EVALKIT_JUDGE_TIMEOUT", "nan"),
        ("EVALKIT_JUDGE_TIMEOUT", "inf"),
        ("EVALKIT_JUDGE_TIMEOUT", "0"),
        ("EVALKIT_JUDGE_TIMEOUT", "-5"),
    ],
)
def test_from_env_rejects_non_finite_or_out_of_range_numbers(monkeypatch, tmp_path, name, value):
    from evalkit import ConfigError

    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "e.db"))
    monkeypatch.setenv("EVALKIT_JUDGE_MODEL", "m")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        Evaluator.from_env()


def test_an_inconsistent_scoring_result_becomes_a_persisted_scoring_error_not_an_ok_row(
    store, monkeypatch
):
    """Defense in depth: even if scoring returned a verdict that contradicts its own score, the
    paid-for attempt is kept as an error row (with its evidence) and nothing is stored as ok."""
    real_score = Rubric.score

    def contradictory(self, raw):
        scores, overall, _ = real_score(self, raw)
        return scores, overall, "FAIL" if overall >= self.threshold else "PASS"

    monkeypatch.setattr(Rubric, "score", contradictory)
    judge = FakeJudge(judged(correctness=5, clarity=5))
    with pytest.raises(ScoringError, match="inconsistent result") as info:
        Evaluator(judge, store).evaluate("p", "o", criteria=CRITERIA)
    (row,) = store.list()
    assert row.status == "error" and row.verdict is None and row.overall_score is None
    assert info.value.evaluation_id == row.id
    assert [a.outcome for a in row.attempts] == ["ok"]  # the judge call itself did succeed
