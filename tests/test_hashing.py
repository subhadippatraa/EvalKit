"""Canonical JSON and the hashes built on it: stable, unambiguous, domain-separated."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import EvaluationCase
from evalkit.hashing import canonical_json, dataset_hash, stable_hash

BASE = dict(
    case_key="k1",
    prompt="What is 2+2?",
    output="4",
    reference="4",
    retrieved=["d2", "d1"],
    relevance={"d1": 2, "d2": 0},
    metadata={"b": 1, "a": [1.5, None]},
    tags=["z", "a"],
)


def h(**changes):
    return EvaluationCase(**(BASE | changes)).content_hash


# --- canonical JSON --------------------------------------------------------------------------


def test_canonical_json_is_sorted_compact_and_ascii():
    assert canonical_json({"b": 1, "a": [1.0, None, "é"]}) == '{"a":[1.0,null,"\\u00e9"],"b":1}'


def test_key_order_and_tuples_do_not_matter():
    assert canonical_json({"a": 1, "b": (1, 2)}) == canonical_json({"b": [1, 2], "a": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {1: "x"},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": {1, 2}},
        {"a": b"bytes"},
        {"a": object()},
        {"a": {"nested": {2: 3}}},
    ],
)
def test_ambiguous_or_non_json_values_are_refused(bad):
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_types_that_would_blur_together_stay_distinct():
    assert canonical_json({"a": 1}) != canonical_json({"a": 1.0})
    assert canonical_json({"a": True}) != canonical_json({"a": 1})
    assert canonical_json({"a": None}) != canonical_json({"a": ""})
    assert canonical_json({"a": "1"}) != canonical_json({"a": 1})


def test_lone_surrogates_cannot_break_hashing():
    assert "\\ud800" in canonical_json(
        {"a": "x\ud800"}
    )  # ASCII escapes: never a UnicodeEncodeError


def test_stable_hash_is_domain_separated_and_pinned():
    assert stable_hash("a", {"x": 1}) != stable_hash("b", {"x": 1})
    assert (
        stable_hash("d", {"x": 1})
        == "1a31446afe71f8b3301b338d9f25d6f875f075e3076afc9ecf1af8ae72f8e3fe"
    )


# --- case content hash -----------------------------------------------------------------------


def test_case_hash_is_pinned():
    """Characterization pin: a change here means every stored dataset hash would change. If the
    canonicalization must change, introduce a new domain (evalkit-case-v2), never edit v1."""
    assert h() == "397d27ffeeb5c4b37a6aab7de5576f23c6c2b85017fee76704057d73365dfb05"
    assert EvaluationCase(case_key="k", prompt="p").content_hash == (
        "94caa5ef8d76daa608feea0d575bd744e1ae92f72ba18aa827f097ff970014b3"
    )


def test_case_hash_excludes_the_key_and_ignores_metadata_key_order_and_tag_order():
    assert h(case_key="a-completely-different-key") == h()
    assert h(metadata={"a": [1.5, None], "b": 1}) == h()
    assert h(tags=["a", "z"]) == h()
    assert h(tags=["a", "z", "a"]) == h()  # duplicates are removed on validation


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "What is 2+3?"},
        {"output": "5"},
        {"output": None},
        {"reference": "four"},
        {"reference": None},
        {"context": "ctx"},
        {"retrieved": ["d1", "d2"]},  # a ranking: order is content
        {"retrieved": None},
        {"relevance": {"d1": 2, "d2": 1}},
        {"relevance": None},
        {"metadata": {"b": 1, "a": [1.5, 0]}},
        {"metadata": {"b": 1.0, "a": [1.5, None]}},  # 1 vs 1.0
        {"tags": ["z"]},
    ],
)
def test_every_content_field_changes_the_hash(change):
    assert h(**change) != h()


def test_none_and_empty_string_are_different_content():
    assert h(reference=None) != h(reference="")
    assert h(context=None) != h(context="")
    assert h(retrieved=None) != h(retrieved=[])


def test_no_unicode_normalization_or_whitespace_folding_is_applied():
    assert h(prompt="café") != h(prompt="café")  # NFC vs NFD are different bytes
    assert h(prompt="a b") != h(prompt="a  b")
    assert h(prompt="a\n") != h(prompt="a\r\n")


# --- dataset hash ----------------------------------------------------------------------------


def test_dataset_hash_is_pinned_and_sensitive_to_keys_hashes_and_order():
    a, b = ("a", "1" * 64), ("b", "2" * 64)
    assert (
        dataset_hash([a, b]) == "8e3053f6fd064a734e3081ebf3b4d96b5a18bcf08e2f9eca95df7cc923290d56"
    )
    assert dataset_hash([b, a]) != dataset_hash([a, b])  # callers supply ascending case_key order
    assert dataset_hash([a]) != dataset_hash([a, b])
    assert dataset_hash([("a", "3" * 64), b]) != dataset_hash([a, b])
    assert dataset_hash([("c", "1" * 64), b]) != dataset_hash([a, b])


def test_dataset_hash_encoding_is_unambiguous():
    assert dataset_hash([("ab", "c")]) != dataset_hash([("a", "bc")])
    assert dataset_hash([]) != dataset_hash([("", "")])


@settings(max_examples=200, deadline=None)
@given(
    prompt=st.text(min_size=1, max_size=20),
    tags=st.lists(st.text(min_size=1, max_size=5), max_size=4),
    key=st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,20}", fullmatch=True),
)
def test_property_hash_is_deterministic_and_key_independent(prompt, tags, key):
    a = EvaluationCase(case_key=key, prompt=prompt, tags=tags)
    b = EvaluationCase(case_key="other", prompt=prompt, tags=list(reversed(tags)))
    assert (
        a.content_hash
        == b.content_hash
        == EvaluationCase(case_key=key, prompt=prompt, tags=tags).content_hash
    )
