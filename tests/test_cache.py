"""The response cache's identity, codec and concurrency (the executor-level behaviour is in
test_cache_runs.py)."""

import threading
from dataclasses import replace

import pytest

from evalkit.cache import CachePolicy, ResponseCache, cache_key, decode_response, encode_response
from evalkit.llm import LLMRequest, LLMResponse, ToolSpec, Usage
from evalkit.store import SQLiteStore

TOOL = ToolSpec("submit", "d", {"type": "object", "properties": {"a": {"type": "string"}}})
REQ = LLMRequest(system="sys", user="user", tool=TOOL, temperature=0.0, max_tokens=100)
ID = {"provider": "p", "model": "m", "endpoint": "us-east-1", "scope": "judge:abc"}


def key(req=REQ, **over):
    return cache_key(req, **(ID | over))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: replace(r, system="sys2"),
        lambda r: replace(r, user="user2"),
        lambda r: replace(r, temperature=0.5),
        lambda r: replace(r, max_tokens=101),
        lambda r: replace(r, sample_index=1),
        lambda r: replace(r, role="target"),
        lambda r: replace(r, tool=None),
        lambda r: replace(r, tool=replace(TOOL, name="other")),
        lambda r: replace(r, tool=replace(TOOL, description="other")),
        lambda r: replace(r, tool=replace(TOOL, schema={"type": "object"})),
    ],
)
def test_every_request_field_that_can_change_the_answer_changes_the_key(change):
    assert key(change(REQ)) != key()


@pytest.mark.parametrize(
    "over",
    [
        {"provider": "p2"},
        {"model": "m2"},
        {"endpoint": "eu-west-1"},
        {"endpoint": None},
        {"scope": "judge:def"},
    ],
)
def test_provider_model_endpoint_and_evaluator_scope_are_part_of_the_key(over):
    assert key(**over) != key()


def test_the_timeout_is_not_part_of_the_key_and_the_key_is_stable():
    assert key(replace(REQ, timeout_s=1.0)) == key(replace(REQ, timeout_s=99.0)) == key()
    assert len(key()) == 64 and key() == key()


def test_a_new_request_field_would_join_the_key_automatically():
    """The key is built from every field of the request, minus an explicit allow-list."""
    from evalkit.cache import KEY_IGNORES

    assert KEY_IGNORES == {"timeout_s"}


def test_responses_round_trip_and_a_corrupt_one_is_none():
    resp = LLMResponse(
        payload={"a": 1}, text=None, usage=Usage(3, 4), request_id="r1", provider_latency_ms=12,
        stop_reason="tool_use", provider_stop="stopReason=tool_use",
    )  # fmt: skip
    back = decode_response(encode_response(resp))
    assert back == replace(resp, raw=None)
    for bad in ("", "not json", "[]", '{"v": 99}', '{"v": 1, "stop_reason": 5}'):
        assert decode_response(bad) is None


def test_evidence_is_never_cached():
    resp = LLMResponse(text="x", raw={"secret": "evidence"})
    assert decode_response(encode_response(resp)).raw is None


def test_policy_validation():
    assert CachePolicy().mode == "off"
    assert CachePolicy.from_mapping({"mode": "readwrite", "targets": True}).targets is True
    for bad in (
        {"mode": "on"}, {"targets": "yes"}, {"max_temperature": -1}, {"x": 1},
        {"max_temperature": True},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            CachePolicy.from_mapping(bad)


def test_eligibility_follows_mode_role_and_temperature():
    on = ResponseCache(None, CachePolicy("readwrite"))
    assert on.eligible(REQ)
    assert not on.eligible(replace(REQ, temperature=0.1))  # non-deterministic: never cached
    assert not on.eligible(replace(REQ, role="target"))  # targets are opt-in
    assert ResponseCache(None, CachePolicy("readwrite", targets=True)).eligible(
        replace(REQ, role="target")
    )
    assert ResponseCache(None, CachePolicy("readwrite", max_temperature=0.3)).eligible(
        replace(REQ, temperature=0.2)
    )
    assert not ResponseCache(None, CachePolicy("off")).eligible(REQ)
    assert ResponseCache(None, CachePolicy("replay")).eligible(REQ)


def test_put_then_get_and_first_writer_wins(tmp_path):
    store = SQLiteStore(tmp_path / "c.db")
    cache = ResponseCache(store, CachePolicy("readwrite"))
    assert cache.get("a" * 64) is None
    cache.put("a" * 64, "p", "m", LLMResponse(text="one", stop_reason="end_turn"))
    cache.put("a" * 64, "p", "m", LLMResponse(text="two", stop_reason="end_turn"))
    assert cache.get("a" * 64).text == "one"


def test_a_corrupt_stored_entry_is_a_miss(tmp_path):
    store = SQLiteStore(tmp_path / "c.db")
    store.cache_put("b" * 64, "p", "m", "garbage")
    cache = ResponseCache(store, CachePolicy("readwrite"))
    assert cache.get("b" * 64) is None
    cache.put("b" * 64, "p", "m", LLMResponse(text="fresh", stop_reason="end_turn"))
    assert cache.get("b" * 64).text == "fresh"  # and the bad entry did not block the good one


def test_the_key_lock_serializes_the_same_key_only():
    cache = ResponseCache(None, CachePolicy("readwrite"))
    inside, order = threading.Event(), []

    def first():
        with cache.key_lock("k"):
            inside.set()
            threading.Event().wait(0.15)
            order.append("first")

    t = threading.Thread(target=first)
    t.start()
    inside.wait()
    with cache.key_lock("other"):  # a different key is not blocked
        order.append("other")
    with cache.key_lock("k"):  # the same key waits for the first
        order.append("second")
    t.join()
    assert order == ["other", "first", "second"]
    assert not cache._locks  # no lock leaks


def test_replay_and_off_never_write(tmp_path):
    store = SQLiteStore(tmp_path / "c.db")
    resp = LLMResponse(text="x", stop_reason="end_turn")
    for mode in ("replay", "off"):
        ResponseCache(store, CachePolicy(mode)).put("c" * 64, "p", "m", resp)
    assert store.cache_stats()["entries"] == 0
