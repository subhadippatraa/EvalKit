"""P2.1 in the call layer: cache, token/cost capture, budget reservation, events."""

import io
import json
import threading
from dataclasses import replace

import pytest

from evalkit import EvalFailure, FailureClass, events
from evalkit.cache import CachePolicy, ResponseCache
from evalkit.calls import (
    Budget,
    CallRunner,
    RetryPolicy,
    RunGuard,
    SpendTotals,
    UnitAbandoned,
    estimate_call,
)
from evalkit.llm import LLMRequest, LLMResponse, ToolSpec, Usage
from evalkit.pricing import NO_PRICING, PriceTable
from evalkit.store import SQLiteStore

FAST = RetryPolicy(base_s=0.001, cap_s=0.002)
REQ = LLMRequest(system="s", user="hello", temperature=0.0, max_tokens=100, role="evaluator")
PRICES = PriceTable.from_mapping(
    {
        "version": "v1",
        "models": [
            {"provider": "p", "model": "m", "input_per_mtok": 1000.0, "output_per_mtok": 2000.0}
        ],
    }
)


class Client:
    provider, model, endpoint = "p", "m", "here"

    def __init__(self, usage=None):
        self.calls, self.lock, self.usage = 0, threading.Lock(), usage or Usage(10, 5)

    def call(self, req):
        with self.lock:
            self.calls += 1
        return LLMResponse(text="ans", usage=self.usage, request_id="rq", stop_reason="end_turn")


def make(store=None, mode="readwrite", *, budget=None, pricing=PRICES, prior=None, **kw):
    cache = ResponseCache(store or SQLiteStore(":memory:"), CachePolicy(mode, **kw))
    guard = RunGuard(budget or Budget(), prior=prior)
    return CallRunner(FAST, guard=guard, cache=cache, pricing=pricing), guard, cache


def ask(runner, client, req=REQ, scope="ev1", validate=None, **kw):
    calls = runner.unit(scope=scope, **kw)
    from evalkit.calls import describe_llm_response

    value = calls.run(
        lambda fb: client.call(req),
        validate=validate,
        describe=describe_llm_response,
        provider=client.provider,
        model=client.model,
        request=lambda fb: req,
        endpoint=client.endpoint,
    )
    return value, calls


# ---- cache ----------------------------------------------------------------------------------


def test_an_exact_repeat_is_served_from_the_cache_with_no_provider_call_and_no_spend():
    runner, guard, _ = make()
    c = Client()
    first, calls1 = ask(runner, c)
    second, calls2 = ask(runner, c)
    assert c.calls == 1 and second.text == first.text == "ans"
    a1, a2 = calls1.attempts[0], calls2.attempts[0]
    assert (a1.cache_hit, a1.input_tokens, a1.output_tokens) == (False, 10, 5)
    assert a1.cache_key == a2.cache_key and a1.cache_key is not None
    assert (a2.cache_hit, a2.input_tokens, a2.output_tokens, a2.cost_usd) == (True, None, None, 0.0)
    assert a2.outcome == "ok" and a2.request_id is None
    assert (guard.usage.calls, guard.usage.cache_hits) == (1, 1)  # a hit is not a provider call
    assert guard.usage.input_tokens == 10  # ...and adds no tokens


def test_a_different_scope_model_or_request_misses():
    runner, _, _ = make()
    c = Client()
    ask(runner, c)
    ask(runner, c, scope="ev2")
    ask(runner, c, req=replace(REQ, user="other"))
    ask(runner, c, req=replace(REQ, max_tokens=101))
    other = Client()
    other.model = "m2"
    ask(runner, other)
    assert c.calls == 4 and other.calls == 1  # each variant paid once


def test_cache_off_never_reads_or_writes_and_records_no_key():
    runner, guard, _ = make(mode="off")
    c = Client()
    _, calls = ask(runner, c)
    ask(runner, c)
    assert c.calls == 2 and calls.attempts[0].cache_key is None and guard.usage.cache_hits == 0


def test_non_deterministic_requests_are_never_cached():
    runner, _, _ = make()
    c = Client()
    hot = replace(REQ, temperature=0.7)
    ask(runner, c, req=hot)
    ask(runner, c, req=hot)
    assert c.calls == 2


def test_targets_are_cached_only_when_opted_in():
    tgt = replace(REQ, role="target")
    for opt, expected in ((False, 2), (True, 1)):
        runner, _, _ = make(targets=opt)
        c = Client()
        ask(runner, c, req=tgt)
        ask(runner, c, req=tgt)
        assert c.calls == expected


def test_only_validated_responses_are_cached():
    runner, _, _ = make()
    c = Client()

    def reject(resp):
        raise EvalFailure(FailureClass.EVALUATOR, "truncated", "bad")

    for _ in range(2):
        with pytest.raises(EvalFailure):
            ask(runner, c, validate=reject)
    assert c.calls == 2  # a failure is never replayed
    ask(runner, c)  # ...and did not poison the entry
    ask(runner, c)
    assert c.calls == 3


def test_a_cached_response_that_no_longer_validates_is_dropped_and_refetched():
    runner, _, cache = make()
    c = Client()
    ask(runner, c)
    seen = []

    def validate(resp):
        seen.append(resp)
        if len(seen) == 1:  # the *cached* copy fails today's validation
            raise EvalFailure(FailureClass.EVALUATOR, "invalid_output", "stale")
        return resp

    value, calls = ask(runner, c, validate=validate)
    assert c.calls == 2 and value.text == "ans" and calls.attempts[-1].cache_hit is False


def test_replay_mode_hits_but_a_miss_never_calls_the_provider():
    store = SQLiteStore(":memory:")
    warm, _, _ = make(store)
    c = Client()
    ask(warm, c)
    replay, guard, _ = make(store, mode="replay")
    value, calls = ask(replay, c)
    assert value.text == "ans" and c.calls == 1 and calls.attempts[0].cache_hit
    with pytest.raises(UnitAbandoned) as e:
        ask(replay, c, req=replace(REQ, user="never seen"))
    assert e.value.reason == "cache_miss" and c.calls == 1
    assert guard.blocked and guard.blocked[0] == "cache_miss"


def test_replay_never_writes():
    store = SQLiteStore(":memory:")
    replay, _, _ = make(store, mode="replay")
    with pytest.raises(UnitAbandoned):
        ask(replay, Client())
    assert store.cache_stats()["entries"] == 0


def test_concurrent_identical_requests_call_the_provider_once():
    runner, _, _ = make()
    c = Client()
    slow = c.call

    def call(req):
        threading.Event().wait(0.05)
        return slow(req)

    c.call = call
    out = []
    threads = [
        threading.Thread(target=lambda: out.append(ask(runner, c)[0].text)) for _ in range(8)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert out == ["ans"] * 8 and c.calls == 1


# ---- tokens and cost --------------------------------------------------------------------------


def test_cost_is_computed_from_reported_tokens_with_the_price_version():
    runner, guard, _ = make(mode="off")
    _, calls = ask(runner, Client(Usage(1000, 500)))
    a = calls.attempts[0]
    assert a.cost_usd == pytest.approx(1000 * 1000 / 1e6 + 500 * 2000 / 1e6)
    assert a.price_version == "v1"
    assert guard.usage.cost_usd == pytest.approx(a.cost_usd)


def test_unpriced_model_and_missing_usage_are_unknown_not_zero():
    runner, guard, _ = make(mode="off", pricing=NO_PRICING)
    _, calls = ask(runner, Client())
    a = calls.attempts[0]
    assert (a.cost_usd, a.price_version, a.input_tokens) == (None, None, 10)
    assert guard.usage.unpriced_calls == 1 and guard.usage.cost_usd == 0.0
    runner, guard, _ = make(mode="off")
    _, calls = ask(runner, Client(Usage(None, None)))
    assert calls.attempts[0].cost_usd is None and calls.attempts[0].input_tokens is None
    assert guard.usage.unpriced_calls == 1


def test_a_paid_response_that_fails_validation_still_records_its_tokens_and_cost():
    runner, guard, _ = make(mode="off")

    def reject(resp):
        raise EvalFailure(FailureClass.EVALUATOR, "truncated", "cut")

    calls = runner.unit()
    with pytest.raises(EvalFailure):
        c = Client(Usage(100, 50))
        from evalkit.calls import describe_llm_response

        calls.run(
            lambda fb: c.call(REQ), validate=reject, describe=describe_llm_response,
            provider="p", model="m",
        )  # fmt: skip
    a = calls.attempts[0]
    assert a.outcome == "failed" and (a.input_tokens, a.output_tokens) == (100, 50)
    assert a.cost_usd is not None and a.request_id == "rq"
    assert guard.usage.tokens == 150


def test_retry_round_is_stamped_on_every_attempt():
    runner, _, _ = make(mode="off")
    _, calls = ask(runner, Client(), retry_round=2)
    assert calls.attempts[0].retry_round == 2


# ---- budgets ----------------------------------------------------------------------------------


def test_the_estimate_is_an_upper_bound_on_input_plus_the_output_cap():
    est = estimate_call(REQ, PRICES, "p", "m")
    assert est.tokens >= len("s") + len("hello") + REQ.max_tokens
    assert est.cost_usd == pytest.approx(
        PRICES.upper_bound("p", "m", est.tokens - REQ.max_tokens, REQ.max_tokens)
    )
    assert estimate_call(REQ, NO_PRICING, "p", "m").cost_usd is None
    tool = ToolSpec("t", "d", {"type": "object", "big": "x" * 500})
    assert estimate_call(replace(REQ, tool=tool), PRICES, "p", "m").tokens > est.tokens + 500


def test_a_call_that_cannot_fit_the_token_budget_is_never_made():
    runner, guard, _ = make(mode="off", budget=Budget(max_tokens=50))  # cap alone is 100
    c = Client()
    with pytest.raises(UnitAbandoned) as e:
        ask(runner, c)
    assert e.value.reason == "budget" and c.calls == 0
    assert guard.budget_stop() == "budget"


def test_the_usd_budget_gates_on_the_priced_worst_case():
    est = estimate_call(REQ, PRICES, "p", "m").cost_usd
    runner, guard, _ = make(mode="off", budget=Budget(max_cost_usd=est * 1.5))
    c = Client(Usage(1, 1))  # tiny actual spend
    ask(runner, c)  # worst case fits once...
    ask(runner, c)  # ...and again: the actual spend was tiny, so the headroom is still there
    runner2, _, _ = make(mode="off", budget=Budget(max_cost_usd=est * 0.5))
    with pytest.raises(UnitAbandoned):
        ask(runner2, Client())


def test_a_budget_counts_what_earlier_executions_of_the_run_spent():
    # tokens and USD are per run: an earlier execution's spend counts...
    prior = SpendTotals(input_tokens=990)
    runner, guard, _ = make(mode="off", budget=Budget(max_tokens=1000), prior=prior)
    with pytest.raises(UnitAbandoned):
        ask(runner, Client())
    _, g_usd, _ = make(mode="off", budget=Budget(max_cost_usd=1.0), prior=SpendTotals(cost_usd=1.0))
    assert g_usd.budget_stop() == "budget"
    _, g2, _ = make(mode="off", budget=Budget(max_tokens=1000), prior=SpendTotals(input_tokens=999))
    assert g2.budget_stop() is None
    # ...max_calls is per execution, as it was before P2.1
    runner, guard, _ = make(mode="off", budget=Budget(max_calls=3), prior=SpendTotals(calls=3))
    ask(runner, Client())


def test_cache_hits_do_not_consume_the_call_budget():
    runner, guard, _ = make(budget=Budget(max_calls=1))
    c = Client()
    for _ in range(5):
        ask(runner, c)
    assert c.calls == 1 and guard.budget_stop() == "budget"  # spent by the one real call


def test_concurrent_workers_can_never_overspend_the_budget():
    est = estimate_call(REQ, PRICES, "p", "m")
    limit = est.tokens * 5  # room for five worst cases at once
    runner, guard, _ = make(mode="off", budget=Budget(max_tokens=limit))
    c = Client(
        Usage(est.tokens - REQ.max_tokens, REQ.max_tokens)
    )  # each call spends its worst case
    made, abandoned = [], []

    def worker():
        try:
            ask(runner, c)
            made.append(1)
        except UnitAbandoned:
            abandoned.append(1)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert c.calls == len(made) == 5 and len(abandoned) == 35
    assert guard.usage.tokens <= limit


def test_no_provider_call_happens_after_the_budget_is_exhausted():
    runner, guard, _ = make(mode="off", budget=Budget(max_calls=3))
    c = Client()
    for _ in range(3):
        ask(runner, c)
    for _ in range(5):
        with pytest.raises(UnitAbandoned):
            ask(runner, c)
    assert c.calls == 3


def test_an_unreported_usage_is_charged_at_its_reservation_under_a_budget():
    est = estimate_call(REQ, PRICES, "p", "m")
    runner, guard, _ = make(mode="off", budget=Budget(max_tokens=est.tokens * 2 - 1))
    c = Client(Usage(None, None))  # the provider does not say what it used
    ask(runner, c)
    with pytest.raises(UnitAbandoned):  # fail closed: the first call is assumed to have cost it all
        ask(runner, c)
    assert guard.usage.assumed_tokens == est.tokens


def test_a_budget_refusal_after_a_failed_attempt_ends_with_budget_exceeded():
    est = estimate_call(REQ, PRICES, "p", "m")
    runner, guard, _ = make(mode="off", budget=Budget(max_calls=1))
    boom = EvalFailure(FailureClass.INFRA, "rate_limited", "429")
    c = Client()
    c.call = lambda req: (_ for _ in ()).throw(boom)
    calls = runner.unit()
    with pytest.raises(EvalFailure) as e:
        calls.run(
            lambda fb: c.call(REQ), provider="p", model="m", request=lambda fb: REQ,
        )  # fmt: skip
    assert (e.value.failure_class, e.value.kind) == (FailureClass.INFRA, "budget_exceeded")
    assert "rate_limited" in str(e.value) or "429" in str(e.value)
    assert len(calls.attempts) == 1 and calls.attempts[0].error_kind == "rate_limited"
    assert est.tokens > 0


def test_budget_validation_accepts_a_cost_limit():
    assert Budget(max_cost_usd=1.5).max_cost_usd == 1.5
    for bad in (0, -1, float("inf"), True):
        with pytest.raises(ValueError):
            Budget(max_cost_usd=bad)
    assert Budget.from_mapping({"max_cost_usd": 2}).max_cost_usd == 2


# ---- events -----------------------------------------------------------------------------------


@pytest.fixture
def log_stream():
    stream = io.StringIO()
    events.configure_logging("DEBUG", "json", stream=stream)
    yield stream
    events.reset_logging()


def records(stream):
    return [json.loads(x) for x in stream.getvalue().splitlines()]


def test_every_attempt_emits_an_event_with_ids_and_no_content(log_stream):
    runner, _, _ = make()
    c = Client()
    ask(runner, c, ctx={"run_id": "r1", "case_key": "case-1", "evaluator": "ev1"})
    ask(runner, c, ctx={"run_id": "r1", "case_key": "case-2", "evaluator": "ev1"})
    recs = records(log_stream)
    calls = [r for r in recs if r["event"] == "provider.call"]
    assert [r["cache_hit"] for r in calls] == [False, True]
    assert calls[0]["run_id"] == "r1" and calls[0]["case_key"] == "case-1"
    assert calls[0]["attempt"] == 1 and "duration_ms" in calls[0] and calls[0]["request_id"] == "rq"
    assert calls[0]["input_tokens"] == 10 and calls[0]["cost_usd"] is not None
    assert {r["event"] for r in recs} >= {"cache.miss", "cache.hit"}
    text = log_stream.getvalue()
    assert "hello" not in text and "ans" not in text  # neither the prompt nor the output


def test_a_retry_and_a_final_failure_are_events(log_stream):
    runner, _, _ = make(mode="off")
    boom = EvalFailure(
        FailureClass.INFRA, "rate_limited", "SECRET-PROVIDER-TEXT sk-abcdefghijklmnopqrstuv",
        request_id="rq-7",
    )  # fmt: skip
    seq = iter([boom, boom, boom, boom])

    def send(fb):
        raise next(seq)

    calls = runner.unit(ctx={"run_id": "r1", "correlation_id": "c1"})
    with pytest.raises(EvalFailure):
        calls.run(send, provider="p", model="m")
    recs = records(log_stream)
    names = [r["event"] for r in recs]
    assert names.count("provider.retry") == 3 and names.count("provider.failed") == 1
    assert all(r["failure_class"] == "infrastructure" for r in recs if "failure_class" in r)
    retry = next(r for r in recs if r["event"] == "provider.retry")
    assert retry["failure_kind"] == "rate_limited" and retry["request_id"] == "rq-7"
    assert retry["run_id"] == "r1" and retry["correlation_id"] == "c1" and "backoff_ms" in retry
    text = log_stream.getvalue()
    assert "SECRET-PROVIDER-TEXT" not in text and "sk-abcdefghij" not in text
