"""Local benchmark of the platform (no network, fake zero-latency target/judge).

    uv run python scripts/benchmark.py [--cases 10000] [--out benchmark.json]

Measures: dataset import, run planning, execution throughput (cases/s and result rows/s) with three
deterministic evaluators plus a scripted judge, aggregation, comparison, report generation, DB size
and peak memory. It records the machine so the numbers are interpretable. It states what it
measured and nothing about scale it did not: one process, one local SQLite file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from evalkit import EvalKit, EvaluatorSpec, Rubric
from evalkit.compare import compare
from evalkit.evaluators import llm_judge_spec
from evalkit.failures import EvalFailure, FailureClass
from evalkit.llm import LLMResponse, Usage
from evalkit.operations import run_status
from evalkit.report import render_report
from evalkit.targets import CallableTarget, PrecomputedTarget


class Judge:
    provider, model = "fake", "bench"

    def call(self, req):
        return LLMResponse(
            payload={"quality": {"reasoning": "fine", "score": 4}},
            stop_reason="tool_use",
            usage=Usage(200, 20),
        )


class FlakyJudge(Judge):
    """Like `Judge`, but every 50th call (2%) fails once with a retryable outage."""

    provider, model = "fake", "bench"

    def __init__(self):
        self.n = 0
        self.failed_units: set[str] = set()

    def call(self, req):
        self.n += 1
        key = req.user
        if self.n % 50 == 0 and key not in self.failed_units:
            self.failed_units.add(key)
            raise EvalFailure(FailureClass.INFRA, "provider_unavailable", "503")
        return super().call(req)


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def timed(fn):
    t = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=10_000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out")
    args = ap.parse_args()
    n = args.cases
    tmp = Path(tempfile.mkdtemp(prefix="evalkit-bench-"))
    db = tmp / "bench.db"
    kit = EvalKit.open(db)
    result: dict = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
        "cases": n,
        "concurrency": args.concurrency,
    }

    cases = (
        {
            "case_key": f"c{i:07}",
            "prompt": f"question number {i}",
            "reference": f"answer {i}",
            "output": f"answer {i}" if i % 10 else "wrong",
            "tags": [f"t{i % 5}"],
        }
        for i in range(n)
    )
    _, dt = timed(lambda: kit.datasets.import_cases("bench", cases))
    result["import"] = {"seconds": round(dt, 3), "cases_per_s": round(n / dt)}

    judge = Judge()
    rubric = Rubric.from_dict({"quality": "Good?"})
    evaluators = [
        EvaluatorSpec(kind="exact_match", name="em"),
        EvaluatorSpec(kind="regex", name="digits", params={"pattern": r"\d"}),
        EvaluatorSpec(kind="regex", name="answer", params={"pattern": "answer"}),
        llm_judge_spec("quality", rubric, judge),
    ]
    policy = {"concurrency": args.concurrency, "batch_size": 200}
    run, dt = timed(
        lambda: kit.controller.create(
            "bench", target=PrecomputedTarget(), evaluators=evaluators, policy=policy
        )
    )
    result["create_and_plan"] = {"seconds": round(dt, 3), "cases_per_s": round(n / dt)}

    # paging the pending units: every page must cost the same (audit P1-14: it used to sort all the
    # remaining pending rows per page, i.e. quadratic over the run)
    kit.runs.start(run.id)
    page_times, after, seen = [], None, 0
    t_all = time.perf_counter()
    while True:
        t = time.perf_counter()
        page = kit.store.next_pending(run.id, after, 256)
        page_times.append(time.perf_counter() - t)
        seen += len(page)
        if len(page) < 256:
            break
        after = (page[-1][0], page[-1][1].result_id)
    thirds = max(1, len(page_times) // 3)
    result["paging_pending_units"] = {
        "units": seen,
        "pages": len(page_times),
        "seconds_total": round(time.perf_counter() - t_all, 3),
        "ms_per_page_first_third": round(1000 * sum(page_times[:thirds]) / thirds, 3),
        "ms_per_page_last_third": round(1000 * sum(page_times[-thirds:]) / thirds, 3),
        "ms_per_page_max": round(1000 * max(page_times), 3),
    }
    report, dt = timed(lambda: kit.controller.execute(run.id, clients=[judge]))
    rows = n + n * len(evaluators)
    metrics = kit.store._conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    result["execute_precomputed"] = {
        "status": report.status,
        "seconds": round(dt, 3),
        "cases_per_s": round(n / dt),
        "result_rows_per_s": round(rows / dt),
        "result_rows": rows,
        "metric_rows": metrics,
    }

    target = CallableTarget(lambda i: "answer " + i.prompt.split()[-1], name="b", fingerprint="1")
    run2 = kit.controller.create("bench", target=target, evaluators=evaluators[:3], policy=policy)
    report2, dt = timed(lambda: kit.controller.execute(run2.id, target=target))
    result["execute_callable_target"] = {
        "status": report2.status,
        "seconds": round(dt, 3),
        "cases_per_s": round(n / dt),
    }

    _, dt = timed(lambda: kit.store.pending_units(run.id, len(evaluators)))
    result["recovery_scan_of_a_finished_run"] = {"seconds": round(dt, 3)}
    summary, dt = timed(lambda: kit.summarize(run.id))
    result["aggregate"] = {"seconds": round(dt, 3), "evaluators": len(summary.evaluators)}
    cmp, dt = timed(lambda: compare(kit, run2.id, run2.id))
    result["compare_self"] = {"seconds": round(dt, 3), "paired": cmp.common_cases}
    cmp2, dt = timed(lambda: compare(kit, run.id, run2.id, allow_confounders=True))
    result["compare_two_runs"] = {"seconds": round(dt, 3), "metrics": len(cmp2.metrics)}
    page, dt = timed(lambda: render_report(kit, run.id, comparison=None, max_cases=200))
    result["report_200_cases"] = {"seconds": round(dt, 3), "bytes": len(page)}
    v, dt = timed(lambda: kit.runs.verify(run.id))
    result["verify_run"] = {"seconds": round(dt, 3), "ok": v.ok}

    # ---- P2.1: cache, cost tracking, budgets, retry, observability ---------------------------
    priced = {
        "version": "bench-1",
        "models": [
            {"provider": "fake", "model": "bench", "input_per_mtok": 3.0, "output_per_mtok": 15.0}
        ],
    }
    judge_only = [llm_judge_spec("quality", rubric, judge)]
    p2 = {**policy, "pricing": priced, "cache": {"mode": "readwrite"}}
    cold = kit.controller.create(
        "bench", target=PrecomputedTarget(), evaluators=judge_only, policy=p2
    )
    rep_cold, dt = timed(lambda: kit.controller.execute(cold.id, clients=[judge]))
    result["p2_cache_cold_priced_judge"] = {
        "status": rep_cold.status,
        "seconds": round(dt, 3),
        "cases_per_s": round(n / dt),
        "provider_calls": rep_cold.spend.calls,
        "cost_usd_estimate": round(rep_cold.spend.cost_usd, 4),
    }
    warm = kit.controller.create(
        "bench", target=PrecomputedTarget(), evaluators=judge_only, policy=p2
    )
    rep_warm, dt = timed(lambda: kit.controller.execute(warm.id, clients=[judge]))
    result["p2_cache_warm_rerun"] = {
        "status": rep_warm.status,
        "seconds": round(dt, 3),
        "cases_per_s": round(n / dt),
        "provider_calls": rep_warm.spend.calls,
        "cache_hits": rep_warm.spend.cache_hits,
    }
    uncached = kit.controller.create(
        "bench",
        target=PrecomputedTarget(),
        evaluators=judge_only,
        policy={**policy, "pricing": priced},
    )
    rep_plain, dt = timed(lambda: kit.controller.execute(uncached.id, clients=[judge]))
    result["p2_priced_judge_no_cache"] = {
        "seconds": round(dt, 3),
        "cases_per_s": round(n / dt),
        "provider_calls": rep_plain.spend.calls,
    }
    flaky = FlakyJudge()
    fl = kit.controller.create(
        "bench", target=PrecomputedTarget(), evaluators=[llm_judge_spec("quality", rubric, flaky)],
        policy={**policy, "retry": {"max_attempts": 1, "base_s": 0.001, "cap_s": 0.002}},
    )  # fmt: skip
    rep_fl, dt = timed(lambda: kit.controller.execute(fl.id, clients=[flaky]))
    failed = sum(v.get("failed", 0) for v in kit.runs.counts(fl.id).evaluator_results.values())
    rep_retry, dt_retry = timed(
        lambda: kit.controller.execute(fl.id, clients=[flaky], retry_failed=True)
    )
    result["p2_retry_failed"] = {
        "first_pass_seconds": round(dt, 3),
        "failed_after_first_pass": failed,
        "retry_seconds": round(dt_retry, 3),
        "retry_reopened_evaluators": rep_retry.retry["reopened_evaluators"],
        "provider_calls_in_retry": rep_retry.spend.calls,
        "failed_after_retry": sum(
            v.get("failed", 0) for v in kit.runs.counts(fl.id).evaluator_results.values()
        ),
        "verify_ok": kit.runs.verify(fl.id).ok,
    }
    status, dt = timed(lambda: run_status(kit, cold.id))
    result["p2_run_status"] = {"seconds": round(dt, 3), "tokens": status["tokens"]["total"]}
    page, dt = timed(lambda: render_report(kit, cold.id, max_cases=200))
    result["p2_report_with_operations"] = {"seconds": round(dt, 3), "bytes": len(page)}

    kit.close()
    size = sum(f.stat().st_size for f in tmp.glob("bench.db*"))
    result["db_megabytes"] = round(size / 1e6, 1)
    result["peak_rss_mb"] = round(rss_mb(), 1)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
