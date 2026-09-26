"""Run observability: the status dictionary, its text form, and the report's Operations section."""

import re
import threading

from conftest import judged

from evalkit import EvaluatorSpec, Rubric
from evalkit.calls import estimate_call
from evalkit.evaluators import llm_judge_spec
from evalkit.llm import LLMResponse, Usage
from evalkit.operations import format_status, run_status
from evalkit.pricing import NO_PRICING
from evalkit.report import render_report
from evalkit.targets import PrecomputedTarget

RUBRIC = Rubric.from_dict({"quality": "Good?"})
POLICY = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0, "max_attempts": 1}}
PRICING = {
    "version": "ops-1",
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
    provider, model = "fake", "judge-1"

    def __init__(self, endpoint="us-east-1", fail_case=None, usage=None):
        self.endpoint, self.fail_case = endpoint, fail_case
        self.usage = usage or Usage(100, 10)
        self.lock, self.calls = threading.Lock(), 0

    def call(self, req):
        from evalkit import EvalFailure, FailureClass

        with self.lock:
            self.calls += 1
        if self.fail_case and self.fail_case in req.user:
            raise EvalFailure(FailureClass.INFRA, "provider_unavailable", "503")
        return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=self.usage)


def dataset(kit, n=6):
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i}", "prompt": f"q{i}", "output": f"out-{i}"} for i in range(n)]
    )


def run_with(kit, judge, policy=None):
    run = kit.controller.create(
        "qa",
        target=PrecomputedTarget(),
        evaluators=[
            llm_judge_spec("quality", RUBRIC, judge),
            EvaluatorSpec(kind="regex", name="any", params={"pattern": "."}),
        ],
        policy={**POLICY, **(policy or {})},
    )
    kit.controller.execute(run.id, clients=[judge])
    return run


def test_status_reports_every_operational_number(kit):
    dataset(kit)
    judge = Judge(fail_case="out-2")
    run = run_with(kit, judge, {"pricing": PRICING, "cache": {"mode": "readwrite"}})
    warm = run_with(kit, Judge(), {"pricing": PRICING, "cache": {"mode": "readwrite"}})
    s = run_status(kit, run.id)
    assert s["run"]["status"] == "succeeded"
    assert s["cases"] == {
        "total": 6, "completed": 6, "failed": 0, "pending": 0, "missing": 0, "target_failures": {},
    }  # fmt: skip
    judge_ev = next(e for e in s["evaluators"] if e["kind"] == "llm_judge")
    assert (judge_ev["scored"], judge_ev["failed"]) == (5, 1)
    assert judge_ev["coverage"] == 5 / 6
    assert s["calls"]["provider_calls"] == 6 and s["calls"]["failed_calls"] == 1
    assert (
        s["cache"]["hits"] == 0 and s["cache"]["misses"] == 6 and s["cache"]["mode"] == "readwrite"
    )
    assert s["tokens"] == {"input": 500, "output": 50, "total": 550, "unknown_usage_attempts": 0}
    assert s["cost"]["usd_estimate"] > 0 and s["cost"]["complete"] is True
    assert s["cost"]["price_versions"] == ["ops-1"] and s["cost"]["estimate"] is True
    assert s["duration"]["executions"] == 1 and s["duration"]["execution_s"] >= 0
    assert s["budget"]["state"] == "none"
    assert any(f["kind"] == "provider_unavailable" for f in s["failures"])
    w = run_status(kit, warm.id)
    assert w["cache"]["hits"] == 5 and w["calls"]["provider_calls"] == 1  # only the failure repeats


def test_status_of_a_partial_budgeted_run_names_the_limit(kit):
    from evalkit.judge_eval import build_request

    dataset(kit, 20)
    j = Judge()
    probe = kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[llm_judge_spec("quality", RUBRIC, j)]
    )
    spec = probe.config.evaluators[0]
    case = next(iter(kit.datasets.cases("qa")))
    req = build_request(
        case.prompt, case.output, case.reference, case.context, RUBRIC, temperature=0.0,
        timeout_s=1.0, max_tokens=spec.params["max_tokens"],
    )  # fmt: skip
    worst = estimate_call(req, NO_PRICING, "fake", "judge-1").tokens
    spender = Judge(usage=Usage(worst - 4096 + 64, 4096))  # each call spends its worst case
    run = run_with(kit, spender, {"budget": {"max_tokens": worst * 2}})
    s = run_status(kit, run.id)
    assert s["run"]["status"] == "partial" and s["run"]["stop_reason"] == "budget"
    b = s["budget"]
    assert b["state"] == "exhausted" and b["exhausted_by"] == "budget"
    assert b["limits"] == {"max_tokens": worst * 2} and "max_tokens" in b["detail"]
    assert s["cases"]["pending"] > 0
    text = format_status(s)
    assert "budget     exhausted" in text and "pending" in text


def test_status_of_an_unpriced_run_says_cost_is_unknown(kit):
    dataset(kit, 3)
    run = run_with(kit, Judge())
    s = run_status(kit, run.id)
    assert s["cost"]["usd_estimate"] is None and s["cost"]["complete"] is False
    assert "cost       unknown" in format_status(s)


def test_status_of_a_run_that_never_executed(kit):
    dataset(kit, 3)
    run = kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[EvaluatorSpec(kind="regex", name="r",
                                                                    params={"pattern": "."})]
    )  # fmt: skip
    s = run_status(kit, run.id)
    assert s["cases"]["pending"] == 3 and s["duration"] == {
        "execution_s": None, "wall_s": None, "executions": 0,
    }  # fmt: skip
    assert s["calls"]["provider_calls"] == 0 and s["cache"]["hit_rate"] is None


def test_the_report_has_an_operations_section_that_stays_self_contained(kit):
    dataset(kit, 4)
    run = run_with(kit, Judge("<script>alert(1)</script>"), {"pricing": PRICING})
    page = render_report(kit, run.id)
    assert "<h2>Operations</h2>" in page
    for label in ("Provider calls", "Cache hits", "Estimated cost", "Budget", "Executions",
                  "Evaluator coverage", "Reproducibility snapshot", "ops-1"):  # fmt: skip
        assert label in page, label
    assert "<script>alert(1)</script>" not in page  # data is escaped
    assert "&lt;script&gt;" in page
    assert "Content-Security-Policy" in page and "default-src" in page
    assert not re.search(r"<script|<link|<img|src=|@import|url\(", page)
    assert "http://" not in page and "https://" not in page


def test_the_report_for_a_run_that_never_ran_still_renders(kit):
    dataset(kit, 2)
    run = kit.controller.create(
        "qa", target=PrecomputedTarget(),
        evaluators=[EvaluatorSpec(kind="regex", name="r", params={"pattern": "."})],
    )  # fmt: skip
    assert "<h2>Operations</h2>" in render_report(kit, run.id)
