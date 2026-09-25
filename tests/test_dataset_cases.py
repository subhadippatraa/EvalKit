"""EvaluationCase: strict structure, bounded size, lossless content."""

import json

import pytest
from conftest import case
from pydantic import ValidationError

from evalkit import EvaluationCase, Limits
from evalkit.datasets import IssueCollector, check_case_limits, validate_cases
from evalkit.limits import MAX_DOC_IDS, MAX_JSON_DEPTH, MAX_TAG_CHARS


def make(**fields):
    return EvaluationCase(**case(**fields))


def issues_for(raw, limits=None):
    collector = IssueCollector()
    valid = list(validate_cases([(1, raw)], limits or Limits(), collector))
    return valid, collector.issues


# --- shape -----------------------------------------------------------------------------------


def test_minimal_case_has_documented_defaults():
    c = EvaluationCase(case_key="k", prompt="p")
    assert (c.output, c.reference, c.context, c.retrieved, c.relevance) == (None,) * 5
    assert c.metadata == {} and c.tags == []


def test_unknown_fields_are_refused_including_the_legacy_reference_name():
    with pytest.raises(ValidationError, match="reference_output"):
        EvaluationCase(case_key="k", prompt="p", reference_output="x")
    with pytest.raises(ValidationError):
        EvaluationCase(case_key="k", prompt="p", surprise=1)


@pytest.mark.parametrize(
    "key", ["", " a", "a b", "é", "a" * 129, "-x", ".x", "a\n", "a\tb", "a@1", "a,b"]
)
def test_invalid_case_keys(key):
    with pytest.raises(ValidationError):
        EvaluationCase(case_key=key, prompt="p")


@pytest.mark.parametrize("key", ["a", "Q1", "a.b-c_d:e/f", "0", "x" * 128])
def test_valid_case_keys(key):
    assert EvaluationCase(case_key=key, prompt="p").case_key == key


def test_prompt_is_required_and_non_empty_but_output_may_be_empty():
    with pytest.raises(ValidationError):
        EvaluationCase(case_key="k", prompt="")
    with pytest.raises(ValidationError):
        EvaluationCase(case_key="k")
    assert (
        EvaluationCase(case_key="k", prompt="p", output="").output == ""
    )  # an empty answer is data


def test_tags_are_deduplicated_in_order_and_bounded():
    assert make(tags=["b", "a", "b", "c", "a"]).tags == ["b", "a", "c"]
    for bad in ([""], ["t" * (MAX_TAG_CHARS + 1)]):
        with pytest.raises(ValidationError):
            make(tags=bad)


@pytest.mark.parametrize("field", ["prompt", "output", "reference", "context"])
def test_unpaired_surrogates_are_refused_in_every_text_field(field):
    with pytest.raises(ValidationError, match="[Uu]nicode"):  # ours, or pydantic's own check
        make(**{field: "bad \ud800 text"})


def test_unpaired_surrogates_are_refused_in_ids_tags_and_metadata():
    for bad in (
        {"tags": ["\ud800"]},
        {"retrieved": ["\ud800"]},
        {"relevance": {"\ud800": 1}},
        {"metadata": {"k": "\ud800"}},
        {"metadata": {"\ud800": 1}},
        {"metadata": {"k": ["ok", {"deep": "\ud800"}]}},
    ):
        with pytest.raises(ValidationError, match="Unicode"):
            make(**bad)


# --- retrieval labels ------------------------------------------------------------------------


def test_retrieval_fields_round_trip_and_keep_order():
    c = make(retrieved=["d3", "d1", "d2"], relevance={"d1": 3, "d9": 0})
    assert c.retrieved == ["d3", "d1", "d2"] and c.relevance == {"d1": 3, "d9": 0}


@pytest.mark.parametrize(
    "bad",
    [
        {"retrieved": [""]},
        {"retrieved": ["x" * 257]},
        {"retrieved": [f"d{i}" for i in range(MAX_DOC_IDS + 1)]},
        {"relevance": {"": 1}},
        {"relevance": {"d": -1}},
        {"relevance": {"d": 101}},
        {"relevance": {f"d{i}": 1 for i in range(MAX_DOC_IDS + 1)}},
    ],
)
def test_invalid_retrieval_fields(bad):
    with pytest.raises(ValidationError):
        make(**bad)


# --- metadata --------------------------------------------------------------------------------


def nested(depth):
    value = "leaf"
    for _ in range(depth):
        value = {"k": value}
    return value


def test_metadata_depth_is_bounded():
    assert make(metadata=nested(MAX_JSON_DEPTH - 1)).metadata
    with pytest.raises(ValidationError, match="nested deeper"):
        make(metadata=nested(MAX_JSON_DEPTH + 2))


@pytest.mark.parametrize(
    "bad",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": {1, 2}},
        {"x": b"bytes"},
        {"x": object()},
        {1: "int key"},
        {"x": {"nested": {2: "int key"}}},
        {"x": 10**5000},  # cannot be rendered as JSON
    ],
)
def test_metadata_must_be_plain_finite_json(bad):
    with pytest.raises(ValidationError):
        make(metadata=bad)


def test_metadata_is_normalized_to_exactly_its_stored_json_form():
    c = make(metadata={"t": (1, 2), "n": {"a": 1}})
    assert c.metadata == {"t": [1, 2], "n": {"a": 1}}  # what is hashed is what is stored


# --- strictness on the import path (no coercion) ---------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"relevance": {"d": "3"}},
        {"relevance": {"d": True}},
        {"relevance": {"d": 1.0}},
        {"tags": "single"},
        {"retrieved": "d1"},
        {"prompt": 5},
        {"output": 5},
        {"metadata": [1, 2]},
        {"case_key": 7},
    ],
)
def test_import_validation_does_not_coerce_types(bad):
    valid, issues = issues_for(case(**bad))
    assert valid == [] and len(issues) == 1


def test_non_object_items_are_reported():
    for raw in (42, "text", None, [1, 2], 3.5):
        valid, issues = issues_for(raw)
        assert valid == [] and "expected a JSON object" in issues[0].message


# --- size limits -----------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["prompt", "output", "reference", "context"])
def test_field_limit_counts_utf8_bytes(field):
    limits = Limits(max_field_bytes=100)
    valid, issues = issues_for(case(**{field: "x" * 100}), limits)
    assert len(valid) == 1 and issues == []
    valid, issues = issues_for(case(**{field: "x" * 101}), limits)
    assert valid == [] and f"`{field}` is too large" in issues[0].message
    valid, issues = issues_for(case(**{field: "日" * 40}), limits)  # 40 chars = 120 bytes
    assert valid == [] and "too large" in issues[0].message


def test_metadata_tag_and_whole_case_limits():
    assert issues_for(case(metadata={"k": "v" * 100}), Limits(max_metadata_bytes=50))[1]
    assert issues_for(case(tags=["a", "b", "c"]), Limits(max_tags=2))[1]
    big = case(prompt="p" * 400, output="o" * 400)
    assert issues_for(big, Limits(max_case_bytes=500))[1]
    assert not issues_for(big, Limits(max_case_bytes=5000))[1]


def test_check_case_limits_counts_ids_and_tags_toward_the_case_total():
    c = make(retrieved=["d" * 200] * 5, tags=["t" * 100] * 3)
    with pytest.raises(ValueError, match="case is too large"):
        check_case_limits(c, Limits(max_case_bytes=1000))


# --- the JSONL form --------------------------------------------------------------------------


def test_to_record_omits_unset_fields_and_round_trips():
    minimal = EvaluationCase(case_key="k", prompt="p")
    assert minimal.to_record() == {"case_key": "k", "prompt": "p"}
    full = make(
        output="",
        reference="r",
        context="c",
        retrieved=[],
        relevance={},
        metadata={"a": 1},
        tags=["t"],
    )
    record = json.loads(json.dumps(full.to_record()))
    assert list(record) == [
        "case_key", "prompt", "output", "reference", "context", "retrieved", "relevance",
        "metadata", "tags",
    ]  # fmt: skip
    assert EvaluationCase(**record) == full  # "" and [] survive: they are not the same as absent


# --- issue collection ------------------------------------------------------------------------


def test_the_collector_caps_what_it_records_regardless_of_the_caller():
    collector = IssueCollector(cap=3)
    for i in range(10):
        collector.add(i, "problem")
    assert len(collector.issues) == 3 and collector.full
    with pytest.raises(Exception, match="stopped after the first 3"):
        collector.raise_if_any("invalid dataset")


def test_a_flood_of_duplicate_keys_is_bounded_too(kit, store):
    from evalkit import DatasetError
    from evalkit.limits import MAX_ISSUES

    rows = [case("same") for _ in range(MAX_ISSUES * 5)]
    with pytest.raises(DatasetError) as info:
        kit.datasets.import_cases("qa", rows)
    assert len(info.value.issues) == MAX_ISSUES
    assert store._conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 0
