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
from evalkit.llm import LLMResponse, Usage
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
