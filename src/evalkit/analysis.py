"""Run aggregation (docs/TARGET-ARCHITECTURE.md §9.1): every number with its denominator, its
failure attribution and its uncertainty.

For each evaluator the summary states *how many cases each outcome accounts for* (scored,
not applicable, skipped because the target failed, evaluator failed, infrastructure failed, input
excluded, still missing), so nothing disappears into a mean:

    coverage = scored / (cases - not_applicable)

A headline value is withheld (`value = None`) when coverage is below `min_coverage`: an average over
the cases that happened to work says nothing about the ones that did not. The raw `observed_mean`
stays visible and flagged. Failures are never scored: `pass_rate_strict` counts every case that did
not pass (target failures included) in its denominator, while the quality means are over scored
cases only, and the failure rates are reported beside them.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from evalkit import stats
from evalkit.runs import SCORING_VERSION

if TYPE_CHECKING:
    from evalkit.kit import EvalKit

DEFAULT_MIN_COVERAGE = 0.95
MIN_N = 30  # below this many scored cases a metric carries a small-sample warning
MIN_SLICE = 10  # a tag slice needs at least this many scored cases to be shown
LENGTH_BIAS_MIN_N = 10


@dataclass
class MetricSummary:
    name: str
    n: int
    observed_mean: float
    value: float | None  # the headline; None when coverage is insufficient
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float
    ci_low: float
    ci_high: float
    ci_method: str  # wilson | bootstrap | normal | none
    binary: bool
    reliable: bool
    note: str | None = None


@dataclass
class EvaluatorSummary:
    key: str
    kind: str
    name: str
    counts: dict[str, int]
    scored: int
    applicable: int
    coverage: float | None
    sufficient_coverage: bool
    evaluator_failure_rate: float | None  # evaluator-class failures / cases it was attempted on
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    verdicts: dict[str, int] = field(default_factory=dict)
    pass_rate: dict[str, Any] | None = None  # over confident verdicts, with bounds for UNCERTAIN
    pass_rate_strict: float | None = None  # passes / all cases: failures count as non-passes
    latency_ms: dict[str, float] | None = None
    length_bias: dict[str, Any] | None = None
    must_pass_failures: int = 0


@dataclass
class RunSummary:
    run_id: str
    dataset_ref: str
    identity_hash: str
    status: str
    stop_reason: str | None
    cases: int
    case_stage: dict[str, int]
    target_failure_rate: float | None
    case_coverage: float | None  # complete case results / cases
    target_latency_ms: dict[str, float] | None
    evaluators: dict[str, EvaluatorSummary]
    failures: list[dict[str, Any]]
    usage: dict[str, Any]
    slices: dict[str, dict[str, dict[str, float]]]
    min_coverage: float
    seed: int
    scoring_version: int = SCORING_VERSION
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evalkit_version() -> str:
    try:
        return importlib.metadata.version("evalkit")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def metric_seed(seed: int, key: str, name: str) -> int:
    """A per-metric resampling seed: stable, so a report is reproducible."""
    return int.from_bytes(hashlib.sha256(f"{seed}|{key}|{name}".encode()).digest()[:8], "big")


def summarize_metric(
    name: str, values: list[float], *, reliable: bool, seed: int, reason: str | None = None
) -> MetricSummary:
    # storage order is arbitrary (row ids are random); the interval must depend on the values only
    values = sorted(values)
    n = len(values)
    binary = stats.is_binary(values)
    mean = stats.mean(values)
    if binary:
        lo, hi = stats.wilson(int(round(mean * n)), n)
        method = "wilson"
    else:
        lo, hi, method = stats.mean_ci(values, seed=seed)
    note = reason
    if reliable and n < MIN_N:
        note = f"small sample (n={n}): the interval is wide"
    return MetricSummary(
        name=name,
        n=n,
        observed_mean=mean,
        value=mean if reliable else None,
        median=stats.median(values),
        p10=stats.quantile(values, 0.10),
        p90=stats.quantile(values, 0.90),
        minimum=min(values),
        maximum=max(values),
        ci_low=lo,
        ci_high=hi,
        ci_method=method,
        binary=binary,
        reliable=reliable,
        note=note,
    )


def _latency(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": stats.mean(values),
        "p50": stats.quantile(values, 0.50),
        "p95": stats.quantile(values, 0.95),
    }


def _family(model: str | None) -> str | None:
    if not model:
        return None
    parts = re.split(r"[.:/\-]", model.lower())
    return ".".join(parts[:2])


def summarize(
    kit: EvalKit,
    run_id: str,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    seed: int = 0,
    prices: Mapping[str, tuple[float, float]] | None = None,
) -> RunSummary:
    """Aggregate a run from its stored rows. Deterministic for a given (rows, seed).

    `prices` maps a model id to (USD per million input tokens, USD per million output tokens);
    the cost is an ESTIMATE from those user-supplied prices, and attempts of a model without a
    price are counted separately, never guessed.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be within [0, 1]")
    store = kit.store
    run = kit.runs.get(run_id)
    info = store.run_info(run_id)
    total = info["case_count"]

    stage = {"complete": 0, "pending": 0, "failed_target": 0, "failed_input": 0}
    stage["failed_infrastructure"] = 0
    for status, cls, n in store.case_accounting(run_id):
        if status == "failed":
            stage[f"failed_{cls}"] += n
        else:
            stage[status] += n
    terminal = stage["complete"] + sum(v for k, v in stage.items() if k.startswith("failed_"))
    stage["missing"] = total - terminal - stage["pending"]
    target_lat, eval_lat = store.durations(run_id)

    by_eval: dict[str, dict[str, int]] = {}
    for key, status, cls, n in store.evaluator_accounting(run_id):
        c = by_eval.setdefault(key, {})
        label = f"failed_{cls}" if status == "failed" else status
        c[label] = c.get(label, 0) + n

    warnings: list[str] = []
    if run.status != "succeeded":
        warnings.append(
            f"the run is {run.status}" + (f" ({run.stop_reason})" if run.stop_reason else "")
        )
    target = run.config.target
    if target.kind == "callable" and not target.identity.get("fingerprint"):
        warnings.append("unpinned target: the callable declared no fingerprint")
    target_family = _family(target.identity.get("model"))

    metric_names: dict[str, list[str]] = {}
    for key, name in store.metric_names(run_id):
        metric_names.setdefault(key, []).append(name)

    evaluators: dict[str, EvaluatorSummary] = {}
    for spec in run.config.evaluators:
        key = spec.key
        raw = by_eval.get(key, {})
        counts = {
            k: raw.get(k, 0)
            for k in (
                "ok", "not_applicable", "skipped",
                "failed_evaluator", "failed_infrastructure", "failed_input",
            )
        }  # fmt: skip
        counts["missing"] = max(0, terminal - sum(counts.values()))
        scored = counts["ok"]
        applicable = total - counts["not_applicable"]
        coverage = scored / applicable if applicable > 0 else None
        sufficient = coverage is not None and coverage >= min_coverage
        attempted = (
            counts["ok"]
            + counts["not_applicable"]
            + sum(counts[k] for k in counts if k.startswith("failed_"))
        )
        ev = EvaluatorSummary(
            key=key,
            kind=spec.kind,
            name=spec.name,
            counts=counts,
            scored=scored,
            applicable=applicable,
            coverage=coverage,
            sufficient_coverage=sufficient,
            evaluator_failure_rate=counts["failed_evaluator"] / attempted if attempted else None,
            latency_ms=_latency(eval_lat.get(key, [])),
            must_pass_failures=store.must_pass_failures(run_id, key),
        )
        reason = None
        if not sufficient:
            shown = "n/a" if coverage is None else f"{coverage:.1%}"
            reason = f"coverage {shown} is below the required {min_coverage:.0%}; no headline value"
            warnings.append(f"{key}: {reason}")
        for name in sorted(metric_names.get(key, [])):
            values = store.metric_values(run_id, key, name)
            ev.metrics[name] = summarize_metric(
                name, values, reliable=sufficient, seed=metric_seed(seed, key, name), reason=reason
            )
        verdicts = store.verdict_counts(run_id, key)
        ev.verdicts = {
            "PASS": verdicts.get("PASS", 0),
            "FAIL": verdicts.get("FAIL", 0),
            "UNCERTAIN": verdicts.get("UNCERTAIN", 0),
            "none": verdicts.get("", 0),
        }
        passes, fails, unsure = ev.verdicts["PASS"], ev.verdicts["FAIL"], ev.verdicts["UNCERTAIN"]
        if passes + fails + unsure:
            confident = passes + fails
            lo, hi = stats.wilson(passes, confident) if confident else (0.0, 1.0)
            ev.pass_rate = {
                "value": passes / confident if confident and sufficient else None,
                "observed": passes / confident if confident else None,
                "ci_low": lo,
                "ci_high": hi,
                "n": confident,
                # UNCERTAIN as all-FAIL / all-PASS: how much the number leans on doubtful verdicts
                "lower_bound": passes / (confident + unsure),
                "upper_bound": (passes + unsure) / (confident + unsure),
                "uncertain": unsure,
            }
            ev.pass_rate_strict = passes / total if total else None
        if spec.kind == "llm_judge":
            pairs = store.score_vs_length(run_id, key)
            rho = (
                stats.spearman([p[0] for p in pairs], [float(p[1]) for p in pairs])
                if len(pairs) >= LENGTH_BIAS_MIN_N
                else None
            )
            ev.length_bias = {"n": len(pairs), "spearman": rho}
            if rho is not None and abs(rho) >= 0.5:
                warnings.append(
                    f"{key}: judge score correlates with output length (Spearman {rho:+.2f}): "
                    "possible verbosity bias"
                )
            if target_family and _family(spec.params.get("model")) == target_family:
                warnings.append(
                    f"{key}: the judge model looks like the same family as the target "
                    f"({target_family}): self-preference is possible (heuristic)"
                )
        evaluators[key] = ev

    failures = [
        {"scope": f.scope, "class": f.failure_class, "kind": f.kind, "count": f.count}
        for f in kit.runs.failure_counts(run_id)
    ]
    usage = _usage(store.attempt_stats(run_id), prices, total)
    slices = _slices(store.slice_stats(run_id))
    return RunSummary(
        run_id=run_id,
        dataset_ref=run.dataset_ref,
        identity_hash=run.identity_hash,
        status=run.status,
        stop_reason=run.stop_reason,
        cases=total,
        case_stage=stage,
        target_failure_rate=stage["failed_target"] / total if total else None,
        case_coverage=stage["complete"] / total if total else None,
        target_latency_ms=_latency(target_lat),
        evaluators=evaluators,
        failures=failures,
        usage=usage,
        slices=slices,
        min_coverage=min_coverage,
        seed=seed,
        warnings=warnings,
    )


def _usage(
    rows: list[dict[str, Any]], prices: Mapping[str, tuple[float, float]] | None, cases: int
) -> dict[str, Any]:
    attempts = sum(r["attempts"] for r in rows)
    tin, tout = sum(r["in"] for r in rows), sum(r["out"] for r in rows)
    usage: dict[str, Any] = {
        "attempts": attempts,
        "failed_attempts": sum(r["failed"] for r in rows),
        "retries": sum(r["retries"] for r in rows),
        "input_tokens": tin,
        "output_tokens": tout,
        "tokens_per_case": (tin + tout) / cases if cases else None,
        "by_model": [
            {k: r[k] for k in ("scope", "provider", "model", "attempts", "failed", "in", "out")}
            for r in rows
        ],
        "cost_usd_estimate": None,
        "unpriced_attempts": 0,
    }
    if prices is not None:
        cost, unpriced = 0.0, 0
        for r in rows:
            price = prices.get(r["model"]) if r["model"] else None
            if price is None:
                unpriced += r["attempts"]
                continue
            cost += (r["in"] * price[0] + r["out"] * price[1]) / 1_000_000
        if not math.isfinite(cost):
            raise ValueError("prices must be finite")
        usage["cost_usd_estimate"] = cost
        usage["unpriced_attempts"] = unpriced
        usage["cost_per_1k_cases_usd"] = cost / cases * 1000 if cases and not unpriced else None
    return usage


def _slices(rows: list[tuple[str, str, str, int, float]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tag, key, name, n, mean in rows:
        if n >= MIN_SLICE:
            out.setdefault(tag, {})[f"{key}.{name}"] = {"n": n, "mean": mean}
    return out


def take_snapshot(kit: EvalKit, run_id: str) -> str:
    """Compute and store the run's summary (immutable; recomputable from the rows)."""
    summary = summarize(kit, run_id)
    return kit.store.save_summary(run_id, evalkit_version(), summary.to_dict())
