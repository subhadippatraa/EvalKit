"""Run observability (docs/TARGET-ARCHITECTURE.md §12): what a run has done, what it cost, and what
is left -- in one dictionary the CLI prints and the HTML report embeds.

Everything is derived from stored rows (results, attempts, executions), so it is available for a
run of any age and while a run is still going; nothing here changes anything. Numbers keep their
denominators: cost is an *estimate* and says whether it is complete, and a budget says which limit
was hit and what is left.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalkit.analysis import run_usage

if TYPE_CHECKING:
    from evalkit.kit import EvalKit


def _budget(run: Any, executions: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    """The budget in force (the last execution's, which includes any override; else the frozen
    policy's), what has been spent against it, and its state."""
    last = executions[-1]["budget"] if executions and executions[-1]["budget"] else None
    limits = dict(last["limits"]) if last else dict(run.config.policy.get("budget") or {})
    spent = {
        "tokens": usage["input_tokens"] + usage["output_tokens"],
        "cost_usd": usage["cost_usd_estimate"],
    }
    if not limits:
        return {
            "limits": {},
            "state": "none",
            "spent": spent,
            "remaining": {},
            "exhausted_by": None,
        }
    exhausted = last["exhausted"] if last else None
    remaining: dict[str, float | int | None] = {}
    if "max_tokens" in limits:
        remaining["tokens"] = max(0, limits["max_tokens"] - spent["tokens"])
    if "max_cost_usd" in limits and spent["cost_usd"] is not None:
        remaining["cost_usd"] = max(0.0, limits["max_cost_usd"] - spent["cost_usd"])
    if exhausted is None and (remaining.get("tokens") == 0 or remaining.get("cost_usd") == 0.0):
        exhausted = "budget"
    return {
        "limits": limits,
        "state": "exhausted" if exhausted else "within",
        "exhausted_by": exhausted,
        "detail": last.get("detail") if last else None,
        "spent": spent,
        "remaining": remaining,
    }


def run_status(kit: EvalKit, run_id: str) -> dict[str, Any]:
    run = kit.runs.get(run_id)
    counts = kit.runs.counts(run_id)
    usage = run_usage(kit, run_id)
    executions = kit.store.list_executions(run_id)
    terminal = counts.complete + counts.failed
    by_class: dict[str, int] = {}
    failures = []
    for f in kit.runs.failure_counts(run_id):
        failures.append(
            {"scope": f.scope, "class": f.failure_class, "kind": f.kind, "count": f.count}
        )
        if f.scope == "case":
            by_class[f.failure_class] = by_class.get(f.failure_class, 0) + f.count
    evaluators = []
    for spec in run.config.evaluators:
        states = counts.evaluator_results.get(spec.key, {})
        ok, na = states.get("ok", 0), states.get("not_applicable", 0)
        applicable = counts.total_cases - na
        evaluators.append(
            {
                "key": spec.key,
                "name": spec.name,
                "kind": spec.kind,
                "scored": ok,
                "not_applicable": na,
                "skipped": states.get("skipped", 0),
                "failed": states.get("failed", 0),
                "missing": max(0, terminal - sum(states.values())),
                "coverage": ok / applicable if applicable > 0 else None,
            }
        )
    retry = kit.store.retry_stats(run_id)
    elapsed = [e["elapsed_s"] for e in executions if e["elapsed_s"] is not None]
    wall = None
    if run.finished_at is not None and run.started_at is not None:
        wall = (run.finished_at - run.started_at).total_seconds()
    return {
        "run": {
            "id": run.id,
            "name": run.name,
            "dataset": run.dataset_ref,
            "status": run.status,
            "stop_reason": run.stop_reason,
            "created_at": run.created_at.isoformat(),
            "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
        },
        "cases": {
            "total": counts.total_cases,
            "completed": counts.complete,
            "failed": counts.failed,
            "pending": counts.pending,
            "missing": counts.missing,
            "target_failures": by_class,
        },
        "evaluators": evaluators,
        "failures": failures,
        "calls": {
            "provider_calls": usage["provider_calls"],
            "failed_calls": usage["failed_attempts"],
            "retries": usage["retries"],
        },
        "cache": {**usage["cache"], "mode": _cache_mode(run, executions)},
        "retry_failed": {
            "executions": sum(1 for e in executions if e["retry_failed"]),
            "units_retried": retry["units_retried"],
            "superseded_case_failures": retry["superseded_case_failures"],
            "superseded_evaluator_failures": retry["superseded_evaluator_failures"],
            "retry_attempts": usage["retry_round_attempts"],
        },
        "tokens": {
            "input": usage["input_tokens"],
            "output": usage["output_tokens"],
            "total": usage["input_tokens"] + usage["output_tokens"],
            "unknown_usage_attempts": usage["unknown_usage_attempts"],
        },
        "cost": {
            "usd_estimate": usage["cost_usd_estimate"],
            "complete": usage["cost_complete"],
            "unpriced_attempts": usage["unpriced_attempts"],
            "price_versions": usage["price_versions"],
            "estimate": True,
        },
        "duration": {
            "execution_s": round(sum(elapsed), 3) if elapsed else None,
            "wall_s": wall,
            "executions": len(executions),
        },
        "budget": _budget(run, executions, usage),
    }


def _cache_mode(run: Any, executions: list[dict[str, Any]]) -> str:
    if executions:
        return executions[-1]["environment"].get("cache_mode", "off")
    return (run.config.policy.get("cache") or {}).get("mode", "off")


def format_status(s: dict[str, Any]) -> str:
    """A short, human-readable summary (what `runs status` prints to a terminal)."""
    c, cost, t, b = s["cases"], s["cost"], s["tokens"], s["budget"]
    run = s["run"]
    head = f"run {run['id']}  {run['status']}" + (
        f" ({run['stop_reason']})" if run["stop_reason"] else ""
    )
    usd = "unknown" if cost["usd_estimate"] is None else f"${cost['usd_estimate']:.4f}"
    if cost["usd_estimate"] is not None and not cost["complete"]:
        usd += " (partial: some calls unpriced)"
    cache = s["cache"]
    hit = "n/a" if cache["hit_rate"] is None else f"{cache['hit_rate']:.0%}"
    lines = [
        head,
        f"  cases      {c['completed']} completed, {c['failed']} failed, {c['pending']} pending "
        f"of {c['total']}",
    ]
    for e in s["evaluators"]:
        cov = "n/a" if e["coverage"] is None else f"{e['coverage']:.1%}"
        lines.append(
            f"  evaluator  {e['name']}: coverage {cov} ({e['scored']} scored, "
            f"{e['failed']} failed, {e['skipped']} skipped, {e['not_applicable']} n/a)"
        )
    lines += [
        f"  calls      {s['calls']['provider_calls']} provider calls, "
        f"{s['calls']['retries']} retries, {cache['hits']} cache hits ({hit})",
        f"  tokens     {t['total']} ({t['input']} in / {t['output']} out)"
        + (
            f", {t['unknown_usage_attempts']} calls reported none"
            if t["unknown_usage_attempts"]
            else ""
        ),
        f"  cost       {usd} (estimate)",
    ]
    d = s["duration"]
    if d["execution_s"] is not None:
        lines.append(f"  duration   {d['execution_s']}s over {d['executions']} execution(s)")
    if b["state"] != "none":
        limits = ", ".join(f"{k}={v}" for k, v in b["limits"].items())
        lines.append(f"  budget     {b['state']} ({limits})")
    r = s["retry_failed"]
    if r["units_retried"]:
        lines.append(
            f"  retried    {r['units_retried']} unit(s) over {r['executions']} retry run(s)"
        )
    return "\n".join(lines)
