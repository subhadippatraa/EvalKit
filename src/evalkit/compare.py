"""Comparing two runs fairly, and gating on the result (docs/TARGET-ARCHITECTURE.md §9.2-9.4).

Comparing averages of two runs answers nothing on its own: the datasets, the evaluators or the
scoring logic may differ, one run may have crashed on the hard cases, and a small dataset cannot
show a small difference. So:

* **Comparable?** Anything that differs other than what you varied is a *confounder*, named:
  a different dataset content, a different evaluator key (judge model, rubric, prompt
  fingerprint, parameters), a different scoring version, and an environment difference that can
  change what is measured (the endpoint a model was served from, how unusable judge answers are
  retried; see `envsnapshot`). Confounded comparisons are refused unless explicitly allowed, and
  then carry the confounders in the result. Execution-only differences (`exec_hash`, timeouts,
  versions of Python / SDKs / EvalKit, price tables) are informational.
* **Paired by case.** Both runs are scored on the same cases, so between-case difficulty cancels.
  A case that is unscored on either side (any failure class, not applicable, missing) is excluded
  and counted per side and reason; a coverage gap of more than 5 points raises a survivorship
  warning (a candidate that "improves" by crashing on hard cases would otherwise look better).
* **Uncertainty is kept.** Every metric reports n, the paired-difference interval (seeded
  bootstrap, normal approximation for large n), the minimum detectable difference, and
  higher/lower/tied case counts; proportions add an exact McNemar test on the discordant pairs.
* **Decisions** compare the interval with a tolerance delta: REGRESSION if it lies wholly below
  -delta, IMPROVEMENT if wholly above +delta, EQUIVALENT if wholly inside (-delta, +delta), else
  INCONCLUSIVE; fewer than `min_n` pairs is always INCONCLUSIVE. Only *declared* gates can fail a
  build, and a candidate whose coverage is below the minimum can never pass.
"""

from __future__ import annotations

import fnmatch
import json
import math
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from evalkit import stats
from evalkit.analysis import DEFAULT_MIN_COVERAGE, metric_seed, summarize
from evalkit.envsnapshot import environment_differences
from evalkit.errors import RunError

if TYPE_CHECKING:
    from evalkit.analysis import RunSummary
    from evalkit.kit import EvalKit

MIN_PAIRED = 30
COVERAGE_GAP = 0.05
NA_GAP = 0.05  # a difference in not-applicable share worth a warning
TRUNCATION_GAP = 0.05
_TOP = 10
CI_B = stats.BOOTSTRAP_B  # bootstrap resamples for paired intervals (tests lower it for speed)

EXIT_PASS, EXIT_ERROR, EXIT_USAGE, EXIT_REGRESSION, EXIT_INCONCLUSIVE = 0, 1, 2, 3, 4


class ComparisonError(RunError):
    """The two runs cannot be compared (no common cases, or confounded and not allowed)."""


@dataclass
class Confounder:
    kind: str  # dataset | evaluator | scoring_version
    detail: str


@dataclass
class MetricComparison:
    evaluator: str  # "kind:name" (or "run")
    baseline_key: str
    candidate_key: str
    metric: str
    n_paired: int
    mean_baseline: float
    mean_candidate: float
    diff: float  # candidate - baseline
    ci_low: float
    ci_high: float
    ci_method: str
    sd: float
    mde: float | None
    candidate_higher: int
    candidate_lower: int
    tied: int
    binary: bool
    p_value: float | None  # exact McNemar, for proportions
    worst_cases: list[dict[str, Any]] = field(default_factory=list)  # most negative diffs
    best_cases: list[dict[str, Any]] = field(default_factory=list)
    confounded: bool = False


@dataclass
class Comparison:
    baseline_id: str
    candidate_id: str
    baseline_ref: str
    candidate_ref: str
    same_dataset_version: bool
    common_cases: int
    confounders: list[Confounder]
    informational: list[str]
    same_identity: bool  # the same measurement repeated: differences are run-to-run noise
    exclusions: dict[str, dict[str, int]]
    unpaired: dict[str, int]  # input_changed | only_baseline | only_candidate: cases not paired
    coverage: dict[str, dict[str, float | None]]
    pairs: dict[str, str]  # candidate evaluator key -> the baseline key it was compared with
    not_applicable: dict[str, dict[str, float]]  # per side and evaluator: share of common cases
    truncated: dict[str, float]  # per side: share of complete outputs the target cut off
    metrics: list[MetricComparison]
    warnings: list[str]
    noise_floor: dict[str, float] = field(default_factory=dict)
    latency: dict[str, dict[str, float | None]] = field(default_factory=dict)
    seed: int = 0

    @property
    def confounded(self) -> bool:
        return bool(self.confounders)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- comparing -------------------------------------------------------------------------------


def _param_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> str:
    changed = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            if k == "rubric":
                continue  # its content hash carries the change
            changed.append(f"{k}: {_short(a.get(k))} -> {_short(b.get(k))}")
    return "; ".join(changed) or "parameters differ"


def _short(v: Any) -> str:
    text = json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
    return text if len(text) <= 60 else text[:57] + "..."


def _exclusion_reason(entry: dict[str, Any] | None, key: str) -> str:
    """Why a case has no score for evaluator `key` in a run."""
    if entry is None:
        return "no_result"
    status, cls, kind = entry["case"]
    if status == "pending":
        return "pending"
    if status == "failed":
        return f"{cls}_failure"
    e = entry["evaluators"].get(key)
    if e is None:
        return "missing"
    est, ecls, _ = e
    if est == "failed":
        return f"{ecls}_failure"
    return est  # not_applicable | skipped


def compare(
    kit: EvalKit,
    baseline_id: str,
    candidate_id: str,
    *,
    allow_confounders: bool = False,
    seed: int = 0,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    include_run_metrics: bool = True,
) -> Comparison:
    """Pair two runs by case and compare their named metrics. See the module docstring."""
    store = kit.store
    base, cand = kit.runs.get(baseline_id), kit.runs.get(candidate_id)
    binfo, cinfo = store.run_info(baseline_id), store.run_info(candidate_id)
    confounders: list[Confounder] = []
    info: list[str] = []
    warnings: list[str] = []

    same_version = base.dataset_version_id == cand.dataset_version_id
    if not same_version and binfo["content_hash"] != cinfo["content_hash"]:
        info.append(
            f"the runs used different dataset versions ({base.dataset_ref}, {cand.dataset_ref}); "
            "cases are paired by key and by their input side (prompt, context, reference, "
            "relevance labels) -- the outputs may, and usually do, differ"
        )
    if base.config.scoring_version != cand.config.scoring_version:
        confounders.append(
            Confounder(
                "scoring_version",
                f"{base.config.scoring_version} vs {cand.config.scoring_version}: normalization or "
                "verdict logic differs",
            )
        )

    # evaluators: same key = comparable; same kind+name with another key = confounded
    b_specs = {s.key: s for s in base.config.evaluators}
    c_specs = {s.key: s for s in cand.config.evaluators}
    pairs: list[tuple[str, str, bool]] = []  # (baseline key, candidate key, confounded)
    for ck, cs in c_specs.items():
        if ck in b_specs:
            pairs.append((ck, ck, False))
            continue
        match = next(
            (bk for bk, bs in b_specs.items() if (bs.kind, bs.name) == (cs.kind, cs.name)), None
        )
        if match is None:
            info.append(f"evaluator {cs.kind}:{cs.name} exists only in the candidate: not compared")
            continue
        confounders.append(
            Confounder(
                "evaluator",
                f"{cs.kind}:{cs.name} is configured differently "
                f"({_param_diff(b_specs[match].params, cs.params)})",
            )
        )
        pairs.append((match, ck, True))
    for bk, bs in b_specs.items():
        if bk not in c_specs and not any(p[0] == bk for p in pairs):
            info.append(f"evaluator {bs.kind}:{bs.name} exists only in the baseline: not compared")
    env_confounders, env_notes = environment_differences(
        base.environment,
        cand.environment,
        [e["environment"] for e in store.list_executions(baseline_id)],
        [e["environment"] for e in store.list_executions(candidate_id)],
    )
    confounders.extend(Confounder(kind, detail) for kind, detail in env_confounders)
    info.extend(env_notes)
    if confounders and not allow_confounders:
        raise ComparisonError(
            "the runs are confounded, so a difference cannot be attributed to what you varied: "
            + "; ".join(f"[{c.kind}] {c.detail}" for c in confounders)
            + " (allow_confounders=True / --allow-confounders shows the numbers anyway)"
        )

    if base.config.target != cand.config.target:
        info.append(
            f"target varied: {base.config.target.identity} -> {cand.config.target.identity}"
        )
    if base.exec_hash != cand.exec_hash:
        info.append("execution policy differs (speed and cost only; not a confounder)")
    bt, ct = base.environment.get("request_timeout_s"), cand.environment.get("request_timeout_s")
    if bt != ct:
        info.append(
            f"request timeout differs ({bt} s vs {ct} s): a shorter timeout turns slow answers "
            "into failures, which shows up as coverage, not as a different score"
        )
    same_identity = base.identity_hash == cand.identity_hash
    unpinned = [r.id for r in (base, cand) if _unpinned(r.config.target)]
    if unpinned:
        # EvalKit cannot hash code: two runs of an undeclared callable look identical whatever
        # changed in it, so neither "same identity" nor a noise floor may be claimed
        same_identity = False
        info.append(
            "a callable target declares no fingerprint (unpinned): EvalKit cannot tell a rerun "
            "from a code change, so no noise floor is offered"
        )

    b_out, c_out = store.case_outcomes(baseline_id), store.case_outcomes(candidate_id)
    # Pair on (case_key, input_hash): the question side of the case. The system's own output is
    # deliberately NOT part of it -- pairing on the full content would drop exactly the cases whose
    # output changed, i.e. the regressions (audit P0-1).
    both_keys = b_out.keys() & c_out.keys()
    common = {
        k
        for k in both_keys
        if b_out[k]["input_hash"] is not None and b_out[k]["input_hash"] == c_out[k]["input_hash"]
    }
    unpaired = {
        "input_changed": len(both_keys) - len(common),
        "only_baseline": len(b_out.keys() - c_out.keys()),
        "only_candidate": len(c_out.keys() - b_out.keys()),
    }
    if not common:
        raise ComparisonError(
            "the runs have no case in common (same key and identical prompt, context, reference "
            "and relevance)"
        )
    if unpaired["input_changed"]:
        warnings.append(
            f"{unpaired['input_changed']} case(s) share a key but differ in their input (prompt, "
            "context, reference or relevance) between the runs and were NOT paired"
        )
    for side, n in (
        ("baseline", unpaired["only_baseline"]),
        ("candidate", unpaired["only_candidate"]),
    ):
        if n:
            warnings.append(f"{n} case(s) exist only in the {side} run and were not paired")

    exclusions: dict[str, Counter[str]] = {"baseline": Counter(), "candidate": Counter()}
    metrics: list[MetricComparison] = []
    seen_excl: set[tuple[str, str, str]] = set()
    for bk, ck, confounded in pairs:
        names_b = {n for k, n in store.metric_names(baseline_id) if k == bk}
        names_c = {n for k, n in store.metric_names(candidate_id) if k == ck}
        scored_b = {k for k in common if _scored(b_out[k], bk)}
        scored_c = {k for k in common if _scored(c_out[k], ck)}
        for side, out, scored, key in (
            ("baseline", b_out, scored_b, bk),
            ("candidate", c_out, scored_c, ck),
        ):
            for case_key in common - scored:
                tag = (side, key, case_key)
                if tag not in seen_excl:
                    seen_excl.add(tag)
                    exclusions[side][_exclusion_reason(out.get(case_key), key)] += 1
        both = scored_b & scored_c
        spec = c_specs[ck]
        for name in sorted(names_b & names_c):
            bv, cv = (
                store.metric_by_case(baseline_id, bk, name),
                store.metric_by_case(candidate_id, ck, name),
            )
            keys = sorted(both & bv.keys() & cv.keys())
            if keys:
                metrics.append(
                    _paired(
                        f"{spec.kind}:{spec.name}",
                        bk,
                        ck,
                        name,
                        keys,
                        bv,
                        cv,
                        seed,
                        confounded,
                    )  # fmt: skip
                )
    if include_run_metrics:
        metrics += _run_metrics(kit, baseline_id, candidate_id, b_out, c_out, common, seed)

    sb = summarize(kit, baseline_id, min_coverage=min_coverage, seed=seed)
    sc = summarize(kit, candidate_id, min_coverage=min_coverage, seed=seed)
    coverage: dict[str, dict[str, float | None]] = {"baseline": {}, "candidate": {}}
    for bk, ck, _ in pairs:
        coverage["baseline"][bk], coverage["candidate"][ck] = (
            sb.evaluators[bk].coverage,
            sc.evaluators[ck].coverage,
        )
        cb, cc = sb.evaluators[bk].coverage, sc.evaluators[ck].coverage
        if cb is not None and cc is not None and abs(cb - cc) > COVERAGE_GAP:
            warnings.append(
                f"survivorship: coverage differs by {abs(cb - cc):.1%} for {c_specs[ck].kind}:"
                f"{c_specs[ck].name} (baseline {cb:.1%}, candidate {cc:.1%}); an apparent change "
                "may come from which cases were scored"
            )
    na_share: dict[str, dict[str, float]] = {"baseline": {}, "candidate": {}}
    for bk, ck, _ in pairs:
        nb = sum(1 for k in common if b_out[k]["evaluators"].get(bk, ("",))[0] == "not_applicable")
        nc = sum(1 for k in common if c_out[k]["evaluators"].get(ck, ("",))[0] == "not_applicable")
        na_share["baseline"][bk], na_share["candidate"][ck] = nb / len(common), nc / len(common)
        if abs(nb - nc) / len(common) > NA_GAP:
            warnings.append(
                f"not applicable share differs for {c_specs[ck].kind}:{c_specs[ck].name} "
                f"(baseline {nb / len(common):.1%}, candidate {nc / len(common):.1%}): the metric "
                "is computed on different cases, and a system can leave a metric by changing what "
                "it outputs"
            )
    truncated = {
        "baseline": sb.truncated_outputs / sb.cases if sb.cases else 0.0,
        "candidate": sc.truncated_outputs / sc.cases if sc.cases else 0.0,
    }
    if abs(truncated["baseline"] - truncated["candidate"]) > TRUNCATION_GAP:
        warnings.append(
            f"truncated outputs differ (baseline {truncated['baseline']:.1%}, candidate "
            f"{truncated['candidate']:.1%}): cut-off answers were scored as if complete"
        )
    noise = {}
    if same_identity:
        info.append(
            "same identity: both runs measure the same thing, so differences are run-to-run noise"
        )
        noise = {
            f"{m.evaluator}.{m.metric}": max(abs(m.diff), (m.ci_high - m.ci_low) / 2)
            for m in metrics
        }
    if not any(m.evaluator != "run" for m in metrics):
        warnings.append("no evaluator metric could be compared (nothing scored on both sides)")
    return Comparison(
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        baseline_ref=base.dataset_ref,
        candidate_ref=cand.dataset_ref,
        same_dataset_version=same_version,
        common_cases=len(common),
        confounders=confounders,
        informational=info,
        same_identity=same_identity,
        exclusions={k: dict(v) for k, v in exclusions.items()},
        unpaired=unpaired,
        coverage=coverage,
        pairs={ck: bk for bk, ck, _ in pairs},
        not_applicable=na_share,
        truncated=truncated,
        metrics=metrics,
        warnings=warnings,
        noise_floor=noise,
        latency=_latency(b_out, c_out, common),
        seed=seed,
    )


def _unpinned(target: Any) -> bool:
    return target.kind == "callable" and not target.identity.get("fingerprint")


def _scored(entry: dict[str, Any], key: str) -> bool:
    e = entry["evaluators"].get(key)
    return entry["case"][0] == "complete" and e is not None and e[0] == "ok"


def _paired(
    evaluator: str,
    bk: str,
    ck: str,
    name: str,
    keys: Sequence[str],
    bv: Mapping[str, float],
    cv: Mapping[str, float],
    seed: int,
    confounded: bool,
) -> MetricComparison:
    b = [bv[k] for k in keys]
    c = [cv[k] for k in keys]
    diffs = [y - x for x, y in zip(b, c, strict=True)]
    n = len(diffs)
    binary = stats.is_binary(b) and stats.is_binary(c)
    if n >= 2:
        lo, hi, method = stats.mean_ci(diffs, seed=metric_seed(seed, f"{bk}|{ck}", name), b=CI_B)
    else:
        lo = hi = diffs[0]
        method = "none"
    p_value = None
    if binary:
        only_b = sum(1 for x, y in zip(b, c, strict=True) if x == 1 and y == 0)
        only_c = sum(1 for x, y in zip(b, c, strict=True) if x == 0 and y == 1)
        p_value = stats.mcnemar_exact(only_b, only_c)
    ranked = sorted(zip(keys, b, c, diffs, strict=True), key=lambda t: t[3])

    def row(t: tuple[str, float, float, float]) -> dict[str, Any]:
        return {"case_key": t[0], "baseline": t[1], "candidate": t[2], "diff": t[3]}

    return MetricComparison(
        evaluator=evaluator,
        baseline_key=bk,
        candidate_key=ck,
        metric=name,
        n_paired=n,
        mean_baseline=stats.mean(b),
        mean_candidate=stats.mean(c),
        diff=stats.mean(diffs),
        ci_low=lo,
        ci_high=hi,
        ci_method=method,
        sd=stats.stdev(diffs),
        mde=stats.mde(diffs),
        candidate_higher=sum(1 for d in diffs if d > 0),
        candidate_lower=sum(1 for d in diffs if d < 0),
        tied=sum(1 for d in diffs if d == 0),
        binary=binary,
        p_value=p_value,
        worst_cases=[row(t) for t in ranked[:_TOP] if t[3] < 0],
        best_cases=[row(t) for t in reversed(ranked[-_TOP:]) if t[3] > 0],
        confounded=confounded,
    )


def _run_metrics(kit, baseline_id, candidate_id, b_out, c_out, common, seed):
    """Run-level paired series: target failure per case (a proportion)."""
    keys = sorted(
        k for k in common if b_out[k]["case"][0] != "pending" and c_out[k]["case"][0] != "pending"
    )
    if not keys:
        return []

    def failed(entry: dict[str, Any]) -> float:
        status, cls, _ = entry["case"]
        return 1.0 if status == "failed" and cls == "target" else 0.0

    bv = {k: failed(b_out[k]) for k in keys}
    cv = {k: failed(c_out[k]) for k in keys}
    return [_paired("run", "run", "run", "target_failure_rate", keys, bv, cv, seed, False)]


def _latency(b_out, c_out, common) -> dict[str, dict[str, float | None]]:
    def series(out: dict[str, dict[str, Any]]) -> list[float]:
        return [
            out[k]["duration_ms"]
            for k in common
            if out[k]["case"][0] == "complete" and out[k]["duration_ms"] is not None
        ]

    b, c = series(b_out), series(c_out)
    result: dict[str, dict[str, float | None]] = {}
    for label, q in (("p50", 0.5), ("p95", 0.95)):
        if b and c:
            vb, vc = stats.quantile(b, q), stats.quantile(c, q)
            rel = (vc - vb) / vb if vb > 0 else None
            result[f"latency_{label}_ms"] = {"baseline": vb, "candidate": vc, "rel_change": rel}
    return result


# ---- gates -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricGate:
    ref: str  # "<evaluator selector>.<metric>" or "run.<metric>"
    direction: Literal["higher", "lower"]
    delta: float | None = None  # absolute tolerance
    rel_delta: float | None = None  # relative to the baseline mean

    def __post_init__(self) -> None:
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"gate {self.ref!r}: direction must be 'higher' or 'lower'")
        if (self.delta is None) == (self.rel_delta is None):
            raise ValueError(f"gate {self.ref!r}: give exactly one of delta / rel_delta")
        for v in (self.delta, self.rel_delta):
            if v is not None and (isinstance(v, bool) or not 0 <= v < float("inf")):
                raise ValueError(f"gate {self.ref!r}: tolerances must be finite and >= 0")
        if "." not in self.ref:
            raise ValueError(f"gate ref {self.ref!r} must look like 'kind:name.metric' or 'run.x'")


@dataclass(frozen=True)
class Gates:
    metrics: tuple[MetricGate, ...] = ()
    floors: Mapping[str, float] = field(default_factory=dict)  # absolute minimum of the candidate
    must_pass: Mapping[str, float] = field(default_factory=dict)  # selector -> max failure rate
    min_coverage: float = DEFAULT_MIN_COVERAGE
    min_n: int = MIN_PAIRED
    strict: bool = False  # inconclusive is a failure (exit 4)
    # a candidate must not be able to leave a metric to hide a regression: the not-applicable share
    # may differ from the baseline's by at most this (None: not checked), and may not exceed
    # `max_na_share` outright (None: no cap)
    max_na_asymmetry: float | None = NA_GAP
    max_na_share: float | None = None
    max_truncated_share: float | None = None  # cap on the candidate's truncated-output share
    allow_unpinned: bool = False  # accept a callable target that declares no fingerprint

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be within [0, 1]")
        if isinstance(self.min_n, bool) or not isinstance(self.min_n, int) or self.min_n < 2:
            raise ValueError("min_n must be an integer >= 2")
        for ref, v in self.floors.items():
            if "." not in ref or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"floor {ref!r} must map a 'selector.metric' to a finite number")
        for sel, rate in self.must_pass.items():
            if isinstance(rate, bool) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"must_pass {sel!r}: the allowed failure rate is within [0, 1]")
        for name in ("max_na_asymmetry", "max_na_share", "max_truncated_share"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int | float)):
                raise ValueError(f"{name} must be a number or absent")
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not isinstance(self.allow_unpinned, bool):
            raise ValueError("allow_unpinned must be true or false")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Gates:
        data = dict(data)
        raw = dict(data.pop("gates", {}))
        floors = dict(raw.pop("floor", {}))
        known = {
            "min_coverage", "min_n", "strict", "must_pass", "max_na_asymmetry", "max_na_share",
            "max_truncated_share", "allow_unpinned",
        }  # fmt: skip
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown gate setting(s): {sorted(unknown)}")
        must = data.get("must_pass", {})
        if isinstance(must, list):
            must = dict.fromkeys(must, 0.0)
        metrics = []
        for ref, spec in raw.items():
            if not isinstance(spec, Mapping) or set(spec) - {"direction", "delta", "rel_delta"}:
                raise ValueError(f"gate {ref!r} must be a table with direction and delta/rel_delta")
            metrics.append(
                MetricGate(ref, spec.get("direction", ""), spec.get("delta"), spec.get("rel_delta"))
            )
        return cls(
            tuple(metrics),
            floors,
            must,
            data.get("min_coverage", DEFAULT_MIN_COVERAGE),
            data.get("min_n", MIN_PAIRED),
            data.get("strict", False),
            data.get("max_na_asymmetry", NA_GAP),
            data.get("max_na_share"),
            data.get("max_truncated_share"),
            data.get("allow_unpinned", False),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> Gates:
        p = Path(path)
        raw = p.read_bytes()
        try:
            data = json.loads(raw) if p.suffix == ".json" else tomllib.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"cannot read gates file {p}: {e}") from e
        return cls.from_mapping(data)


Decision = Literal["regression", "improvement", "equivalent", "inconclusive"]


@dataclass
class GateDecision:
    ref: str
    evaluator: str
    metric: str
    direction: str
    delta: float
    decision: Decision
    reason: str
    n_paired: int
    diff: float | None
    ci_low: float | None
    ci_high: float | None
    mde: float | None
    p_value: float | None = None


@dataclass
class GateResult:
    status: Literal["pass", "fail", "inconclusive"]
    exit_code: int
    decisions: list[GateDecision]
    failures: list[dict[str, str]]  # {code, message}: why the gate failed
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _undecided(gate: MetricGate, selector: str, metric: str, reason: str) -> GateDecision:
    return GateDecision(
        ref=gate.ref, evaluator=selector, metric=metric, direction=gate.direction,
        delta=gate.delta or 0.0, decision="inconclusive", reason=reason, n_paired=0,
        diff=None, ci_low=None, ci_high=None, mde=None,
    )  # fmt: skip


def _selector_matches(selector: str, key: str) -> bool:
    return fnmatch.fnmatchcase(key, selector) or fnmatch.fnmatchcase(key, selector + ":*")


def _split_ref(ref: str) -> tuple[str, str]:
    selector, _, metric = ref.partition(".")
    return selector, metric


def decide(mc: MetricComparison, gate: MetricGate, min_n: int) -> GateDecision:
    """The decision rule (design 9.2/9.4) for one paired metric against one gate."""
    sign = 1.0 if gate.direction == "higher" else -1.0
    delta = (
        gate.delta if gate.delta is not None else (gate.rel_delta or 0.0) * abs(mc.mean_baseline)
    )
    lo, hi = sorted((sign * mc.ci_low, sign * mc.ci_high))
    common = dict(
        ref=gate.ref, evaluator=mc.evaluator, metric=mc.metric, direction=gate.direction,
        delta=delta, n_paired=mc.n_paired, diff=mc.diff, ci_low=mc.ci_low, ci_high=mc.ci_high,
        mde=mc.mde, p_value=mc.p_value,
    )  # fmt: skip
    if mc.n_paired < min_n:
        return GateDecision(
            decision="inconclusive",
            reason=f"insufficient_data: {mc.n_paired} paired cases, at least {min_n} needed",
            **common,
        )
    band = (
        f"tolerance {delta:g}, difference {mc.diff:+.4g}, "
        f"95% CI [{mc.ci_low:+.4g}, {mc.ci_high:+.4g}]"
    )
    if hi < -delta:
        return GateDecision(
            decision="regression",
            reason=f"the whole interval is worse than -{delta:g} ({band})",
            **common,
        )
    if lo > delta:
        return GateDecision(
            decision="improvement",
            reason=f"the whole interval is better than +{delta:g} ({band})",
            **common,
        )
    if delta > 0 and lo > -delta and hi < delta:
        return GateDecision(
            decision="equivalent",
            reason=f"the whole interval is within +/-{delta:g} ({band})",
            **common,
        )
    mde = f"; this dataset cannot detect a change smaller than {mc.mde:.3g}" if mc.mde else ""
    return GateDecision(
        decision="inconclusive",
        reason=f"the interval overlaps the tolerance band ({band}){mde}",
        **common,
    )


def _point(name: str, baseline: float, candidate: float, n: int) -> MetricComparison:
    """A run-level quantity with no per-case pairing (latency, tokens, cost): only its relative
    change against a tolerance can be judged, not a confidence interval."""
    d = candidate - baseline
    return MetricComparison(
        evaluator="run", baseline_key="run", candidate_key="run", metric=name, n_paired=n,
        mean_baseline=baseline, mean_candidate=candidate, diff=d, ci_low=d, ci_high=d,
        ci_method="point", sd=0.0, mde=None, candidate_higher=0, candidate_lower=0, tied=0,
        binary=False, p_value=None,
    )  # fmt: skip


def _run_metric_comparison(
    cmp: Comparison, name: str, sb: RunSummary, sc: RunSummary
) -> MetricComparison | None:
    if name == "target_failure_rate":
        return next((m for m in cmp.metrics if m.evaluator == "run" and m.metric == name), None)
    if name in ("latency_p50_ms", "latency_p95_ms"):
        lat = cmp.latency.get(name)
        if lat is None or lat["rel_change"] is None:
            return None
        return _point(name, lat["baseline"], lat["candidate"], cmp.common_cases)
    if name == "tokens_per_case":
        b, c = sb.usage["tokens_per_case"], sc.usage["tokens_per_case"]
    elif name == "cost_per_1k_usd":
        b, c = sb.usage.get("cost_per_1k_cases_usd"), sc.usage.get("cost_per_1k_cases_usd")
    else:
        return None
    return None if b is None or c is None else _point(name, b, c, cmp.common_cases)


def evaluate_gates(
    kit: EvalKit,
    candidate_id: str,
    gates: Gates,
    comparison: Comparison | None = None,
    *,
    seed: int = 0,
    prices: Mapping[str, tuple[float, float]] | None = None,
) -> GateResult:
    """Apply declared gates to a candidate (and its comparison with a baseline, if any).

    The gate FAILS on: a regression beyond tolerance, a candidate value under an absolute floor, a
    must-pass criterion failing more often than allowed, or coverage below `min_coverage` for any
    evaluator a gate refers to -- missing data is not evidence of health. It is INCONCLUSIVE when
    nothing failed but a declared metric could not be decided. Everything is explained from the
    stored numbers in each decision.
    """
    sc = summarize(kit, candidate_id, min_coverage=gates.min_coverage, seed=seed, prices=prices)
    sb = None
    if comparison is not None:
        sb = summarize(
            kit, comparison.baseline_id, min_coverage=gates.min_coverage, seed=seed, prices=prices
        )
    decisions: list[GateDecision] = []
    failures: list[dict[str, str]] = []
    notes: list[str] = []
    referenced: set[str] = set()

    def evaluators_for(selector: str) -> list[str]:
        return [k for k in sc.evaluators if _selector_matches(selector, k)]

    for gate in gates.metrics:
        selector, metric = _split_ref(gate.ref)
        if comparison is None or sb is None:
            reason = "no baseline: a regression gate needs a comparison"
            decisions.append(_undecided(gate, selector, metric, reason))
            continue
        if selector == "run":
            mc = _run_metric_comparison(comparison, metric, sb, sc)
            matches = [mc] if mc is not None else []
        else:
            matches = [
                m
                for m in comparison.metrics
                if m.metric == metric
                and m.evaluator != "run"
                and _selector_matches(selector, m.candidate_key)
            ]
            referenced.update(m.candidate_key for m in matches)
            referenced.update(evaluators_for(selector))
        if not matches:
            reason = "the metric was not compared (absent, or unscored on both sides)"
            decisions.append(_undecided(gate, selector, metric, reason))
            continue
        for mc in matches:
            if mc.ci_method == "point":  # latency / tokens / cost: relative change vs tolerance
                decisions.append(_point_decision(mc, gate))
            else:
                decisions.append(decide(mc, gate, gates.min_n))
    for d in decisions:
        if d.decision == "regression":
            failures.append({"code": "regression", "message": f"{d.ref}: {d.reason}"})

    for ref, floor in gates.floors.items():
        selector, metric = _split_ref(ref)
        keys = evaluators_for(selector)
        referenced.update(keys)
        if not keys:
            failures.append(
                {"code": "floor", "message": f"{ref}: no evaluator matches {selector!r}"}
            )
        for key in keys:
            m = sc.evaluators[key].metrics.get(metric)
            if m is None or m.value is None:
                why = "no value (insufficient coverage or not scored)"
                failures.append({"code": "floor", "message": f"{ref} on {key}: {why}"})
            elif m.value < floor:
                failures.append(
                    {
                        "code": "floor",
                        "message": f"{ref} on {key}: {m.value:.4g} is below the floor {floor:g}",
                    }
                )

    for selector, max_rate in gates.must_pass.items():
        keys = evaluators_for(selector)
        referenced.update(keys)
        if not keys:
            failures.append({"code": "must_pass", "message": f"{selector}: no evaluator matches"})
        for key in keys:
            ev = sc.evaluators[key]
            rate = ev.must_pass_failures / ev.scored if ev.scored else 0.0
            if rate > max_rate:
                failures.append(
                    {
                        "code": "must_pass",
                        "message": f"{key}: {ev.must_pass_failures} of {ev.scored} scored cases "
                        f"failed a must-pass criterion ({rate:.1%} > allowed {max_rate:.1%})",
                    }
                )

    for key in sorted(referenced):
        ev = sc.evaluators[key]
        if not ev.sufficient_coverage:
            shown = "undefined" if ev.coverage is None else f"{ev.coverage:.1%}"
            failures.append(
                {
                    "code": "coverage",
                    "message": f"{key}: coverage {shown} is below the required "
                    f"{gates.min_coverage:.0%}; a gate cannot pass on missing data",
                }
            )
    declared = bool(gates.metrics or gates.floors or gates.must_pass)
    if declared and not gates.allow_unpinned:
        runs = [candidate_id] + ([comparison.baseline_id] if comparison is not None else [])
        for rid in runs:
            if _unpinned(kit.runs.get(rid).config.target):
                failures.append(
                    {
                        "code": "unpinned_target",
                        "message": f"run {rid}: the callable target declares no fingerprint, so a "
                        "gate cannot tell what code it measured (pass --fingerprint, or set "
                        "allow_unpinned)",
                    }
                )
    for key in sorted(referenced):
        share = sc.evaluators[key].not_applicable_share
        if gates.max_na_share is not None and share > gates.max_na_share:
            failures.append(
                {
                    "code": "not_applicable_share",
                    "message": f"{key}: {share:.1%} of the candidate's cases are not applicable "
                    f"(allowed {gates.max_na_share:.1%}): the metric covers too little",
                }
            )
        bk = None if comparison is None else comparison.pairs.get(key)
        if gates.max_na_asymmetry is not None and comparison is not None and bk is not None:
            gap = comparison.not_applicable["candidate"].get(key, 0.0) - (
                comparison.not_applicable["baseline"].get(bk, 0.0)
            )
            if abs(gap) > gates.max_na_asymmetry:
                failures.append(
                    {
                        "code": "not_applicable_asymmetry",
                        "message": f"{key}: the not-applicable share moved by {gap:+.1%} against "
                        f"the baseline (allowed {gates.max_na_asymmetry:.1%}): the metric is "
                        "computed on different cases",
                    }
                )
    if gates.max_truncated_share is not None:
        share = sc.truncated_outputs / sc.cases if sc.cases else 0.0
        if share > gates.max_truncated_share:
            failures.append(
                {
                    "code": "truncated_share",
                    "message": f"{share:.1%} of the candidate's outputs were truncated by the "
                    f"target (allowed {gates.max_truncated_share:.1%})",
                }
            )
    inconclusive = [d for d in decisions if d.decision == "inconclusive"]
    if failures:
        status, code = "fail", EXIT_REGRESSION
    elif inconclusive:
        status, code = "inconclusive", (EXIT_INCONCLUSIVE if gates.strict else EXIT_PASS)
    else:
        status, code = "pass", EXIT_PASS
    if comparison is not None and comparison.confounded:
        notes.append(
            "the comparison is confounded (see its confounders); decisions are not attributable"
        )
    if not gates.metrics and not gates.floors and not gates.must_pass:
        notes.append("no gates are declared: nothing can fail")
    return GateResult(status, code, decisions, failures, notes)


def _point_decision(mc: MetricComparison, gate: MetricGate) -> GateDecision:
    """Latency, tokens and cost: relative change against the tolerance (no significance test)."""
    base = abs(mc.mean_baseline)
    change = (mc.mean_candidate - mc.mean_baseline) / base if base > 0 else 0.0
    sign = 1.0 if gate.direction == "higher" else -1.0
    tol = (
        gate.rel_delta
        if gate.rel_delta is not None
        else (gate.delta or 0.0) / base
        if base > 0
        else 0.0
    )
    eff = sign * change
    if eff < -tol:
        decision: Decision = "regression"
    elif eff > tol:
        decision = "improvement"
    else:
        decision = "equivalent"
    reason = (
        f"{mc.metric}: {mc.mean_baseline:.4g} -> {mc.mean_candidate:.4g} ({change:+.1%}) "
        f"against a tolerance of {tol:.1%}"
    )
    return GateDecision(
        gate.ref, "run", mc.metric, gate.direction, tol, decision, reason, mc.n_paired,
        mc.diff, None, None, None,
    )  # fmt: skip


def resolve_baseline(kit: EvalKit, spec: str, candidate_id: str) -> str:
    """A baseline run id, or `tag:NAME` = the latest succeeded run with that tag on the same
    dataset lineage as the candidate (no baselines table: tags do the job)."""
    if not spec.startswith("tag:"):
        return kit.runs.get(spec).id
    tag = spec[4:]
    info = kit.store.run_info(candidate_id)
    found = kit.store.latest_run_with_tag(tag, info["dataset_version_id"], exclude=candidate_id)
    if found is None:
        raise ComparisonError(
            f"no succeeded run with tag {tag!r} on this dataset to use as baseline"
        )
    return found
