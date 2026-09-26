"""The three workflows EvalKit exists for, end to end (the regression workflow has its own file,
test_regression_e2e.py):

  2. Reliability: run -> failure and interruption -> resume / retry -> no duplicated successful
     work -> final results identical to a run that never failed.
  3. Production operations: LLM calls -> cache -> token and cost tracking -> budget enforcement ->
     structured events -> run and report summary.
"""

import io
import json
import threading
from collections import Counter

from conftest import judged

from evalkit import EvalFailure, FailureClass, Rubric, events
from evalkit.calls import CancelToken
from evalkit.evaluators import llm_judge_spec
from evalkit.llm import LLMResponse, Usage
from evalkit.operations import run_status
from evalkit.report import render_report
from evalkit.targets import PrecomputedTarget

N = 30
RUBRIC = Rubric.from_dict({"quality": "Good?"})
POLICY = {
    "concurrency": 4,
    "retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0, "max_attempts": 1},
    "pricing": {
        "version": "e2e-1",
        "models": [
            {"provider": "fake", "model": "j", "input_per_mtok": 3.0, "output_per_mtok": 15.0}
        ],
    },
}


class Judge:
    """Scores by the output's number so every case has a distinct, checkable result; fails the
    first call of some cases with a retryable outage."""

    provider, model, endpoint = "fake", "j", "e2e"

    def __init__(self, outage=(), usage=None):
        self.outage, self.usage = set(outage), usage or Usage(200, 20)
        self.ok_calls, self.attempts = Counter(), Counter()
        self.lock = threading.Lock()

    def call(self, req):
        n = int(req.user.split("answer-")[1].split()[0].strip("<>/\n"))
        with self.lock:
            self.attempts[n] += 1
            first = self.attempts[n] == 1
            if n in self.outage and first:
                raise EvalFailure(FailureClass.INFRA, "provider_unavailable", "503")
            self.ok_calls[n] += 1
        return LLMResponse(
            payload=judged(quality=1 + n % 5), stop_reason="tool_use", usage=self.usage
        )


def dataset(kit):
    kit.datasets.import_cases(
        "qa",
        [{"case_key": f"c{i:02}", "prompt": f"q{i}", "output": f"answer-{i}"} for i in range(N)],
    )


def make(kit, judge, **policy):
    return kit.controller.create(
        "qa",
        target=PrecomputedTarget(),
        evaluators=[llm_judge_spec("quality", RUBRIC, judge)],
        policy={**POLICY, **policy},
    )


def scores(kit, run):
    out = {}
    for cr in kit.runs.case_results(run.id):
        (er,) = kit.runs.evaluator_results(cr.id)
        out[cr.case_key] = (er.status, er.score)
    return out


def test_reliability_interrupt_fail_resume_retry_matches_a_clean_run(kit):
    dataset(kit)
    clean_judge = Judge()
    clean = make(kit, clean_judge)
    kit.controller.execute(clean.id, clients=[clean_judge])
    expected = scores(kit, clean)
    assert all(status == "ok" for status, _ in expected.values())

    judge = Judge(outage={2, 5, 11, 17, 23})
    run = make(kit, judge, concurrency=1)
    token = CancelToken()

    def interrupt_after_ten(n):
        if n >= 10:
            token.cancel("test interrupt")

    first = kit.controller.execute(
        run.id, clients=[judge], token=token, on_progress=interrupt_after_ten
    )
    assert first.status == "cancelled" and kit.runs.counts(run.id).pending > 0
    second = kit.controller.execute(run.id, clients=[judge])  # resume: finish what is left
    assert second.status == "succeeded"
    failed = [k for k, (s, _) in scores(kit, run).items() if s == "failed"]
    assert len(failed) >= 1  # the outages are recorded as failures, not scores
    third = kit.controller.execute(run.id, clients=[judge], retry_failed=True)
    assert third.status == "succeeded" and third.retry["reopened_evaluators"] == len(failed)

    assert scores(kit, run) == expected  # identical to a run that never failed
    assert all(n == 1 for n in judge.ok_calls.values()) and len(judge.ok_calls) == N  # no repeat
    assert sum(judge.attempts.values()) == N + len(failed)  # each failure cost one extra call
    assert kit.runs.verify(run.id).ok
    assert len(kit.store.failure_history(run.id)) == len(failed)


def test_production_operations_end_to_end(kit):
    dataset(kit)
    stream = io.StringIO()
    events.configure_logging("INFO", "json", stream=stream)
    try:
        judge = Judge()
        run = make(kit, judge, cache={"mode": "readwrite"})
        kit.controller.execute(run.id, clients=[judge])
        rerun = make(kit, judge, cache={"mode": "readwrite"})
        report = kit.controller.execute(rerun.id, clients=[judge])
    finally:
        events.reset_logging()

    assert sum(judge.attempts.values()) == N  # the rerun made no provider call
    assert (report.spend.calls, report.spend.cache_hits) == (0, N)

    first = run_status(kit, run.id)
    assert first["tokens"]["total"] == N * 220 and first["calls"]["provider_calls"] == N
    expected = N * (200 * 3.0 + 20 * 15.0) / 1e6
    assert abs(first["cost"]["usd_estimate"] - expected) < 1e-9 and first["cost"]["complete"]
    second = run_status(kit, rerun.id)
    assert second["cache"]["hit_rate"] == 1.0 and second["cost"]["usd_estimate"] == 0.0
    assert second["tokens"]["total"] == 0

    recs = [json.loads(x) for x in stream.getvalue().splitlines()]
    finished = [r for r in recs if r["event"] == "run.finished"]
    assert [r["run_id"] for r in finished] == [run.id, rerun.id]
    assert finished[0]["calls"] == N and finished[1]["cache_hits"] == N
    assert "answer-" not in stream.getvalue() and "Good?" not in stream.getvalue()

    page = render_report(kit, rerun.id)
    assert "<h2>Operations</h2>" in page and "e2e-1" in page and "100%" in page
    assert "Cache hits" in page and "Estimated cost" in page


def test_budget_enforcement_end_to_end_stops_calls_and_resumes(kit):
    dataset(kit)
    # each call really spends near its worst case (the judge uses its whole 4096-token output
    # cap), so the budget binds: $0.5 admits a handful of ~$0.06 calls, never more
    judge = Judge(usage=Usage(200, 4096))
    run = make(kit, judge, budget={"max_cost_usd": 0.5}, concurrency=8)
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "budget")
    spent = report.run_spend.cost_usd
    assert spent <= 0.5
    calls = sum(judge.attempts.values())
    assert 0 < calls < N
    again = kit.controller.execute(run.id, clients=[judge])
    assert sum(judge.attempts.values()) == calls and again.stop_reason == "budget"
    status = run_status(kit, run.id)
    assert status["budget"]["state"] == "exhausted" and status["cases"]["pending"] == N - calls
    done = kit.controller.execute(
        run.id, clients=[judge], overrides={"budget": {"max_cost_usd": 100.0}}
    )
    assert done.status == "succeeded" and sum(judge.attempts.values()) == N
    assert done.run_spend.cost_usd > 0.5 and done.run_spend.calls == N
