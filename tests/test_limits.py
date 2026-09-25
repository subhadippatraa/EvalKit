"""F-10: oversized or malformed input is refused at the boundary, before any paid judge call."""

import json

import pytest
from conftest import CRITERIA, FakeJudge, judged
from pydantic import ValidationError

from evalkit import Criterion, Evaluator, Limits, Review, Rubric, RubricError
from evalkit.cli import main
from evalkit.limits import (
    MAX_CRITERIA,
    MAX_DESCRIPTION_CHARS,
    MAX_INPUT_FILE_BYTES,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_REVIEW_COMMENT_CHARS,
    MAX_REVIEWER_CHARS,
    MAX_SCALE_ABS,
    truncate_utf8,
)


def evaluate(store, limits=None, judge=None, **kwargs):
    judge = judge or FakeJudge(judged(correctness=5, clarity=5))
    ev = Evaluator(judge, store, limits)
    return judge, ev.evaluate(
        **({"prompt": "p", "model_output": "o", "criteria": CRITERIA} | kwargs)
    )


def assert_refused_for_free(store, judge, **kwargs):
    """Refused with RubricError, the judge never called, and the store unchanged."""
    before = len(store.list(limit=100))
    with pytest.raises(RubricError):
        Evaluator(judge, store, kwargs.pop("limits", None)).evaluate(
            **({"prompt": "p", "model_output": "o", "criteria": CRITERIA} | kwargs)
        )
    assert judge.calls == [] and len(store.list(limit=100)) == before


# --- text fields -------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["prompt", "model_output", "reference_output", "context"])
def test_each_text_field_is_limited_in_utf8_bytes(store, field):
    limits = Limits(max_field_bytes=100)
    _, ok = evaluate(store, limits, **{field: "x" * 100})  # exactly at the limit
    assert getattr(ok, field) == "x" * 100

    judge = FakeJudge(judged(correctness=5, clarity=5))
    kwargs = {"prompt": "p", "model_output": "o", "criteria": CRITERIA, field: "x" * 101}
    with pytest.raises(RubricError, match=f"`{field}` is too large"):
        Evaluator(judge, store, limits).evaluate(**kwargs)
    assert judge.calls == []


def test_the_limit_counts_bytes_not_characters(store):
    limits = Limits(max_field_bytes=100)
    judge = FakeJudge()
    assert_refused_for_free(
        store, judge, limits=limits, model_output="日" * 40
    )  # 40 chars, 120 bytes
    _, ok = evaluate(store, limits, model_output="日" * 33)  # 99 bytes


def test_default_limit_is_256_kib(store):
    _, ok = evaluate(store, model_output="x" * (256 * 1024))
    assert len(ok.model_output) == 256 * 1024
    assert_refused_for_free(store, FakeJudge(), model_output="x" * (256 * 1024 + 1))


@pytest.mark.parametrize("field", ["prompt", "model_output", "reference_output", "context"])
def test_unpaired_surrogates_are_refused_up_front(store, field):
    """They cannot be UTF-8 encoded: they used to fail inside store.save, after the paid call."""
    kwargs = {"prompt": "p", "model_output": "o", "criteria": CRITERIA, field: "bad \ud800 text"}
    with pytest.raises(RubricError, match=f"`{field}` is not valid Unicode"):
        Evaluator(FakeJudge(), store).evaluate(**kwargs)


def test_metadata_and_tag_limits(store):
    limits = Limits(max_metadata_bytes=50, max_tags=3)
    assert_refused_for_free(store, FakeJudge(), limits=limits, metadata={"k": "v" * 100})
    assert_refused_for_free(store, FakeJudge(), limits=limits, tags=["a", "b", "c", "d"])
    assert_refused_for_free(store, FakeJudge(), tags=["t" * 129])
    _, ok = evaluate(store, limits, metadata={"k": "v"}, tags=["a", "b", "c"])
    assert ok.tags == ["a", "b", "c"]


# --- rubric shape --------------------------------------------------------------------------------


def test_rubric_shape_limits():
    def c(i=0, **kw):
        return Criterion(name=f"c{i}", description="d", **kw)

    Rubric(criteria=[c(i) for i in range(MAX_CRITERIA)])
    with pytest.raises(ValidationError):
        Rubric(criteria=[c(i) for i in range(MAX_CRITERIA + 1)])
    Criterion(name="a", description="d" * MAX_DESCRIPTION_CHARS)
    with pytest.raises(ValidationError):
        Criterion(name="a", description="d" * (MAX_DESCRIPTION_CHARS + 1))
    labels = tuple(f"L{i}" for i in range(MAX_LABELS))
    Criterion(name="a", description="d", labels=labels)
    with pytest.raises(ValidationError):
        Criterion(name="a", description="d", labels=(*labels, "one-too-many"))
    with pytest.raises(ValidationError):
        Criterion(name="a", description="d", labels=("ok", "x" * (MAX_LABEL_CHARS + 1)))
    Criterion(name="a", description="d", scale=(-MAX_SCALE_ABS, MAX_SCALE_ABS))
    with pytest.raises(ValidationError):
        Criterion(name="a", description="d", scale=(0, MAX_SCALE_ABS + 1))


def test_an_oversized_rubric_is_refused_before_the_judge(store):
    too_many = {f"c{i}": "d" for i in range(MAX_CRITERIA + 1)}
    assert_refused_for_free(store, FakeJudge(), criteria=too_many)


def test_review_field_limits(store):
    with pytest.raises(ValidationError):
        Review(evaluation_id="e", reviewer="a" * (MAX_REVIEWER_CHARS + 1), verdict="PASS")
    with pytest.raises(ValidationError):
        Review(
            evaluation_id="e",
            reviewer="a",
            verdict="PASS",
            comment="c" * (MAX_REVIEW_COMMENT_CHARS + 1),
        )


def test_truncate_utf8_never_splits_a_character_and_never_exceeds_the_budget():
    text, cut = truncate_utf8("日" * 10, 10)
    assert cut and len(text.encode()) <= 10 and text == "日" * 3
    assert truncate_utf8("short", 100) == ("short", False)


# --- CLI input file -----------------------------------------------------------------------------


@pytest.fixture
def cli_file(tmp_path, monkeypatch, store):
    monkeypatch.setattr(
        Evaluator,
        "from_env",
        classmethod(lambda cls, with_judge=True: cls(FakeJudge() if with_judge else None, store)),
    )

    def write(content, binary=False):
        path = tmp_path / "in.json"
        path.write_bytes(content) if binary else path.write_text(content)
        return path

    return write


_BODY = '{"prompt": "p", "model_output": "o", "criteria": {"a": "A"}, "metadata": {"x": %s}}'


@pytest.mark.parametrize(
    "content",
    [
        *(_BODY % constant for constant in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")),
        '{"prompt": "p", "prompt": "again", "model_output": "o", "criteria": {"a": "A"}}',
        "[" * 200_000,  # RecursionError inside the JSON parser
    ],
)
def test_cli_refuses_non_finite_numbers_duplicate_keys_and_pathological_nesting(
    cli_file, content, capsys
):
    assert main(["run", "--input", str(cli_file(content))]) == 2
    assert "cannot read input file" in capsys.readouterr().err


def test_cli_refuses_invalid_utf8(cli_file, capsys):
    assert main(["run", "--input", str(cli_file(b'{"prompt": "\xff\xfe"}', binary=True))]) == 2


def test_cli_refuses_a_file_over_the_size_cap_without_parsing_it(cli_file, capsys):
    big = b'{"prompt": "' + b"x" * MAX_INPUT_FILE_BYTES + b'"}'
    assert main(["run", "--input", str(cli_file(big, binary=True))]) == 2
    assert "exceeds" in capsys.readouterr().err


def test_cli_missing_required_key_is_a_usage_error_not_an_internal_error(cli_file, capsys):
    assert main(["run", "--input", str(cli_file(json.dumps({"prompt": "p"})))]) == 2
    assert "invalid input file" in capsys.readouterr().err


def test_a_typeerror_inside_evaluate_is_a_bug_and_is_not_reported_as_bad_input(
    cli_file, monkeypatch
):
    def broken(self, **kwargs):
        raise TypeError("internal bug")

    monkeypatch.setattr(Evaluator, "evaluate", broken)
    path = cli_file(json.dumps({"prompt": "p", "model_output": "o", "criteria": {"a": "A"}}))
    with pytest.raises(TypeError, match="internal bug"):  # propagates with a traceback, not exit 2
        main(["run", "--input", str(path)])


def test_cli_input_over_field_limit_exits_1_with_a_clear_message(cli_file, capsys):
    content = json.dumps(
        {"prompt": "p", "model_output": "x" * (256 * 1024 + 1), "criteria": {"a": "A"}}
    )
    assert main(["run", "--input", str(cli_file(content))]) == 1
    assert "too large" in capsys.readouterr().err
