"""P2.1 through the executor: cache, tokens and cost, budgets, executions, events."""

import io
import json
import threading

import pytest
from conftest import judged

from evalkit import EvaluatorSpec, Rubric, RunError, events
from evalkit.calls import estimate_call
from evalkit.engine import PreflightError
from evalkit.evaluators import llm_judge_spec
from evalkit.llm import LLMRequest, LLMResponse, Usage
from evalkit.pricing import NO_PRICING
from evalkit.targets import CallableTarget, ModelTarget, PrecomputedTarget

FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}
RUBRIC = Rubric.from_dict({"quality": "Good?"})
PRICING = {
    "version": "test-2026",
    "models": [
        {
            "provider": "fake",
            "model": "judge-1",
            "input_per_mtok": 1000.0,
            "output_per_mtok": 2000.0,
        }
    ],
}


class Judge:
    provider, model, endpoint = "fake", "judge-1", "test-region"

    def __init__(self, usage=None, delay=0.0):
        usage = usage or Usage(100, 20)
        self.calls, self.lock, self.usage, self.delay = 0, threading.Lock(), usage, delay

    def call(self, req):
        if self.delay:
            threading.Event().wait(self.delay)
        with self.lock:
            self.calls += 1
            n = self.calls
        return LLMResponse(
            payload=judged(quality=5), stop_reason="tool_use", usage=self.usage, request_id=f"r{n}"
        )


def dataset(kit, n=6, name="qa", same_prompt=False):
    cases = [
        {
            "case_key": f"c{i:03}",
            "prompt": "the same question" if same_prompt else f"question {i}",
            "output": "the same answer" if same_prompt else f"answer {i}",
        }
        for i in range(n)
    ]
    kit.datasets.import_cases(name, cases)


def make_run(kit, judge, policy=None, rubric=RUBRIC, ref="qa", name=None):
    return kit.controller.create(
        ref,
        target=PrecomputedTarget(),
        evaluators=[llm_judge_spec("quality", rubric, judge)],
        policy={**FAST, **(policy or {})},
        name=name,
    )


def attempts_of(kit, run):
    out = []
    for cr in kit.runs.case_results(run.id):
        for er in kit.runs.evaluator_results(cr.id):
            out += er.attempts
    return out


# ---- cache ----------------------------------------------------------------------------------


def test_a_rerun_with_the_cache_makes_no_provider_calls_and_scores_the_same(kit):
    dataset(kit)
    judge = Judge()
    policy = {"cache": {"mode": "readwrite"}, "pricing": PRICING}
    first = make_run(kit, judge, policy)
    kit.controller.execute(first.id, clients=[judge])
    assert judge.calls == 6
    second = make_run(kit, judge, policy)
    report = kit.controller.execute(second.id, clients=[judge])
    assert report.status == "succeeded" and judge.calls == 6  # not one more call
    assert (report.spend.calls, report.spend.cache_hits) == (0, 6)
    assert report.spend.tokens == 0 and report.spend.cost_usd == 0.0
    hits = attempts_of(kit, second)
    assert len(hits) == 6 and all(a.cache_hit and a.cost_usd == 0.0 for a in hits)
    a = kit.summarize(first.id)
    b = kit.summarize(second.id)
    key = next(iter(a.evaluators))
    assert a.evaluators[key].metrics["score"].value == b.evaluators[key].metrics["score"].value
    assert b.usage["cache"]["hits"] == 6 and b.usage["cache"]["hit_rate"] == 1.0
    assert a.usage["cache"]["hits"] == 0 and a.usage["cache"]["misses"] == 6


def test_the_cache_is_off_unless_the_policy_turns_it_on(kit):
    dataset(kit, 3)
    judge = Judge()
    for _ in range(2):
        r = make_run(kit, judge)
        kit.controller.execute(r.id, clients=[judge])
    assert judge.calls == 6


def test_a_changed_rubric_or_judge_setting_never_reuses_an_entry(kit):
    dataset(kit, 3)
    judge = Judge()
    policy = {"cache": {"mode": "readwrite"}}
    kit.controller.execute(make_run(kit, judge, policy).id, clients=[judge])
    other = Rubric.from_dict({"quality": "Is it good, really?"})
    kit.controller.execute(make_run(kit, judge, policy, rubric=other).id, clients=[judge])
    assert judge.calls == 6  # 3 + 3: nothing crossed


def test_identical_requests_running_concurrently_are_paid_for_once(kit):
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i}", "prompt": "same", "output": "same"} for i in range(8)]
    )
    judge = Judge(delay=0.03)
    run = make_run(kit, judge, {"cache": {"mode": "readwrite"}, "concurrency": 8})
    report = kit.controller.execute(run.id, clients=[judge])
    assert report.status == "succeeded" and judge.calls == 1
    assert (report.spend.calls, report.spend.cache_hits) == (1, 7)


def test_replay_reproduces_a_run_from_the_cache_alone(kit):
    dataset(kit, 4)
    judge = Judge()
    live = make_run(kit, judge, {"cache": {"mode": "readwrite"}})
    kit.controller.execute(live.id, clients=[judge])
    replay = make_run(kit, judge, {"cache": {"mode": "replay"}})
    report = kit.controller.execute(replay.id, clients=[judge])
    assert report.status == "succeeded" and judge.calls == 4
    assert all(a.cache_hit for a in attempts_of(kit, replay))


def test_replay_with_a_cold_cache_calls_nothing_and_stops_resumably(kit):
    dataset(kit, 4)
    judge = Judge()
    run = make_run(kit, judge, {"cache": {"mode": "replay"}})
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "cache_miss")
    assert judge.calls == 0 and kit.runs.counts(run.id).pending == 4
    assert "replay" in report.stop_detail
    # switching this execution to readwrite finishes it
    done = kit.controller.execute(
        run.id, clients=[judge], overrides={"cache": {"mode": "readwrite"}}
    )
    assert done.status == "succeeded" and judge.calls == 4


def test_a_model_target_is_cached_only_when_asked(kit):
    dataset(kit, 3)

    class Model:
        provider, model = "fake", "gen-1"
        calls = 0

        def call(self, req: LLMRequest):
            Model.calls += 1
            return LLMResponse(text="hi", stop_reason="end_turn", usage=Usage(3, 1))

    def run_with(cache):
        target = ModelTarget(Model(), "{prompt}")
        run = kit.controller.create(
            "qa",
            target=target,
            evaluators=[EvaluatorSpec(kind="regex", name="any", params={"pattern": "."})],
            policy={**FAST, "cache": cache},
        )
        kit.controller.execute(run.id, target=target)

    run_with({"mode": "readwrite"})
    run_with({"mode": "readwrite"})
    assert Model.calls == 6  # targets are not cached by default
    Model.calls = 0
    run_with({"mode": "readwrite", "targets": True})
    run_with({"mode": "readwrite", "targets": True})
    assert Model.calls == 3  # cached: the second run cost nothing


# ---- tokens and cost ----------------------------------------------------------------------------


def test_tokens_and_estimated_cost_are_recorded_with_the_price_version(kit):
    dataset(kit, 4)
    judge = Judge(Usage(1000, 500))
    run = make_run(kit, judge, {"pricing": PRICING})
    report = kit.controller.execute(run.id, clients=[judge])
    per_call = 1000 * 1000 / 1e6 + 500 * 2000 / 1e6
    assert report.spend.cost_usd == pytest.approx(4 * per_call)
    assert report.run_spend.cost_usd == pytest.approx(4 * per_call)
    a = attempts_of(kit, run)[0]
    assert (a.input_tokens, a.output_tokens, a.price_version) == (1000, 500, "test-2026")
    s = kit.summarize(run.id).usage
    assert s["cost_usd_estimate"] == pytest.approx(4 * per_call)
    assert s["unpriced_attempts"] == 0 and s["price_versions"] == ["test-2026"]


def test_without_prices_cost_is_unknown_not_zero(kit):
    dataset(kit, 3)
    judge = Judge()
    run = make_run(kit, judge)
    report = kit.controller.execute(run.id, clients=[judge])
    assert report.spend.unpriced_calls == 3 and attempts_of(kit, run)[0].cost_usd is None
    s = kit.summarize(run.id).usage
    assert s["cost_usd_estimate"] is None and s["unpriced_attempts"] == 3
    assert s["input_tokens"] == 300  # tokens are still exact


def test_a_provider_that_reports_no_usage_is_unknown_usage(kit):
    dataset(kit, 2)
    judge = Judge(Usage(None, None))
    run = make_run(kit, judge, {"pricing": PRICING})
    kit.controller.execute(run.id, clients=[judge])
    a = attempts_of(kit, run)[0]
    assert a.input_tokens is None and a.cost_usd is None
    s = kit.summarize(run.id).usage
    assert s["cost_usd_estimate"] is None and s["unknown_usage_attempts"] == 2


# ---- budgets ------------------------------------------------------------------------------------


def worst_case_judge(kit):
    """A judge that reports exactly the worst case the executor reserves for one of its calls
    (input upper bound + the output cap), so a budget binds exactly."""
    from evalkit.judge_eval import build_request

    probe = make_run(kit, Judge())
    spec = probe.config.evaluators[0]
    case = next(iter(kit.datasets.cases("qa")))
    req = build_request(
        case.prompt, case.output, case.reference, case.context, RUBRIC,
        temperature=0.0, timeout_s=1.0, max_tokens=spec.params["max_tokens"],
    )  # fmt: skip
    est = estimate_call(req, NO_PRICING, "fake", "judge-1").tokens
    out = spec.params["max_tokens"]
    # every case's prompt differs by a few bytes only: use the largest so no request exceeds it
    return Judge(Usage(est - out + 64, out)), est + 64


def test_a_token_budget_never_overspends_and_makes_no_call_after_exhaustion(kit):
    dataset(kit, 40)
    judge, worst = worst_case_judge(kit)
    run = make_run(kit, judge, {"concurrency": 8, "budget": {"max_tokens": worst * 3}})
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "budget")
    assert judge.calls == report.spend.calls == 3  # exactly what fits: three worst cases
    assert report.run_spend.tokens <= worst * 3
    assert "max_tokens" in report.stop_detail
    assert kit.runs.counts(run.id).pending == 37
    again = kit.controller.execute(run.id, clients=[judge])
    assert again.stop_reason == "budget" and judge.calls == 3  # no call after exhaustion


def test_the_budget_is_per_run_so_a_resume_cannot_spend_it_again(kit):
    dataset(kit, 30)
    judge, worst = worst_case_judge(kit)
    run = make_run(kit, judge, {"concurrency": 1, "budget": {"max_tokens": worst * 2}})
    first = kit.controller.execute(run.id, clients=[judge])
    assert first.stop_reason == "budget" and judge.calls == 2
    second = kit.controller.execute(run.id, clients=[judge])
    assert second.stop_reason == "budget" and judge.calls == 2  # the budget is spent, not renewed
    total = kit.store.spend_totals(run.id)
    assert total["input_tokens"] + total["output_tokens"] <= worst * 2
    raised = kit.controller.execute(
        run.id, clients=[judge], overrides={"budget": {"max_tokens": 10**9}}
    )
    assert raised.status == "succeeded" and kit.runs.counts(run.id).pending == 0
    assert judge.calls == 30  # each case judged exactly once across the three executions


def test_a_usd_budget_stops_the_run_on_the_priced_worst_case(kit):
    dataset(kit, 20)
    judge, worst = worst_case_judge(kit)
    per_call = (worst - 4096) * 1000 / 1e6 + 4096 * 2000 / 1e6
    policy = {"pricing": PRICING, "concurrency": 4, "budget": {"max_cost_usd": per_call * 4.5}}
    run = make_run(kit, judge, policy)
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "budget")
    assert judge.calls == 4 and report.run_spend.cost_usd <= per_call * 4.5
    assert "max_cost_usd" in report.stop_detail


def test_a_usd_budget_needs_a_price_for_every_paid_model(kit):
    dataset(kit, 3)
    judge = Judge()
    with pytest.raises(PreflightError, match="budget_unpriced|price"):
        make_run(kit, judge, {"budget": {"max_cost_usd": 1.0}})
    run = make_run(kit, judge)  # no budget at creation...
    with pytest.raises(RunError, match="price"):  # ...and one added at execute is refused too
        kit.controller.execute(run.id, clients=[judge], overrides={"budget": {"max_cost_usd": 1.0}})
    assert judge.calls == 0 and kit.runs.get(run.id).status == "created"


def test_a_callable_target_is_not_priced_so_a_usd_budget_does_not_need_it(kit):
    dataset(kit, 3)
    target = CallableTarget(lambda i: "x", name="t", fingerprint="1")
    kit.controller.create(
        "qa",
        target=target,
        evaluators=[EvaluatorSpec(kind="regex", name="r", params={"pattern": "x"})],
        policy={**FAST, "budget": {"max_cost_usd": 1.0}},
    )


# ---- executions ---------------------------------------------------------------------------------


def test_every_execution_is_recorded_with_its_environment_and_spend(kit):
    dataset(kit, 4)
    judge = Judge()
    run = make_run(kit, judge, {"budget": {"max_tokens": 10**9}, "cache": {"mode": "readwrite"}})
    kit.controller.execute(run.id, clients=[judge])
    (ex,) = kit.store.list_executions(run.id)
    assert ex["seq"] == 1 and ex["outcome_status"] == "succeeded" and ex["units"] == 4
    assert ex["environment"]["endpoints"] == {"fake/judge-1": "test-region"}
    assert ex["environment"]["cache_mode"] == "readwrite"
    assert ex["spend"]["calls"] == 4 and ex["budget"]["limits"] == {"max_tokens": 10**9}
    assert ex["finished_at"] is not None and ex["retry_failed"] == 0


# ---- events -------------------------------------------------------------------------------------


def test_a_full_run_narrates_itself_without_leaking_content(kit):
    events_out = io.StringIO()
    events.configure_logging("DEBUG", "json", stream=events_out)
    try:
        kit.datasets.import_cases(
            "qa", [{"case_key": "k1", "prompt": "SECRET-PROMPT", "output": "SECRET-OUTPUT"}]
        )
        judge = Judge()
        run = make_run(kit, judge, {"cache": {"mode": "readwrite"}})
        kit.controller.execute(run.id, clients=[judge])
    finally:
        events.reset_logging()
    text = events_out.getvalue()
    recs = [json.loads(line) for line in text.splitlines()]
    names = [r["event"] for r in recs]
    assert names[0] == "run.started" and names[-1] == "run.finished"
    assert {"provider.call", "evaluator.finished", "cache.miss"} <= set(names)
    call = next(r for r in recs if r["event"] == "provider.call")
    assert call["run_id"] == run.id and call["case_key"] == "k1" and call["attempt"] == 1
    assert call["evaluator"] == run.config.evaluators[0].key and call["correlation_id"]
    assert call["request_id"] == "r1" and call["duration_ms"] >= 0
    assert "SECRET" not in text and "Good?" not in text
    finished = recs[-1]
    assert finished["status"] == "succeeded" and finished["calls"] == 1


def test_a_failure_event_names_its_class_and_kind_but_not_its_message(kit):
    stream = io.StringIO()
    events.configure_logging("WARNING", "json", stream=stream)
    try:
        kit.datasets.import_cases("qa", [{"case_key": "k1", "prompt": "p", "output": "o"}])

        def boom(inp):
            raise ValueError("SECRET exception text with sk-abcdefghijklmnopqrstuvwxyz")

        target = CallableTarget(boom, name="t", fingerprint="1")
        run = kit.controller.create(
            "qa",
            target=target,
            evaluators=[EvaluatorSpec(kind="regex", name="r", params={"pattern": "x"})],
            policy=FAST,
        )
        kit.controller.execute(run.id, target=target)
    finally:
        events.reset_logging()
    recs = [json.loads(line) for line in stream.getvalue().splitlines()]
    failed = [r for r in recs if r["event"] == "target.failed"]
    assert (
        failed
        and failed[0]["failure_class"] == "target"
        and failed[0]["failure_kind"] == "exception"
    )
    assert "SECRET" not in stream.getvalue() and "sk-abcdef" not in stream.getvalue()
