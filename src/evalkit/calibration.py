"""Human review, calibration and disagreement (docs/TARGET-ARCHITECTURE.md §9.5-9.6).

A judge is never ground truth. Reviews attach to a case result and name the `evaluator_key` whose
verdict they grade. From them:

* **calibration** per evaluator key: judge verdicts vs the human consensus (majority; ties are
  excluded) as accuracy, FAIL-precision, FAIL-recall, Cohen's kappa and a confusion matrix, from
  *random-sample* reviews only. Below `n_min` pairs the evaluator is labelled UNCALIBRATED.
* **reviewer agreement**: humans disagree too; pairwise agreement and pooled kappa are reported.
* **two queues for two purposes**: `random` / `stratified` give an unbiased sample for calibration;
  `uncertain` / `disagreement` surface where the platform is least sure, for triage. Triage
  reviews are stored (`sample='targeted'`) but never enter calibration statistics: mixing the two
  would bias the agreement estimate.
* **disagreement between evaluators** (judge vs judge, judge vs deterministic) per case.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evalkit import stats
from evalkit.errors import RunError
from evalkit.limits import MAX_REVIEW_COMMENT_CHARS, MAX_REVIEWER_CHARS

if TYPE_CHECKING:
    from evalkit.kit import EvalKit

N_MIN = 30


class CaseReview(BaseModel):
    """A human verdict on one case result, optionally grading one evaluator's verdict."""

    model_config = ConfigDict(extra="forbid")

    case_key: str = Field(min_length=1)
    evaluator_key: str | None = None
    reviewer: str = Field(min_length=1, max_length=MAX_REVIEWER_CHARS)
    verdict: Literal["PASS", "FAIL"]
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    comment: str | None = Field(default=None, max_length=MAX_REVIEW_COMMENT_CHARS)
    sample: Literal["random", "targeted"] = "targeted"


@dataclass
class Calibration:
    run_id: str
    evaluator_key: str
    n_paired: int  # random-sample reviewed cases with a confident evaluator verdict and a consensus
    n_min: int
    uncalibrated: bool
    accuracy: float | None
    accuracy_ci: tuple[float, float] | None
    fail_precision: float | None
    fail_recall: float | None
    kappa: float | None
    confusion: dict[str, int]  # FAIL is the positive class
    excluded: dict[str, int]  # uncertain / no_verdict / tie / targeted_only
    reviewer_agreement: dict[str, Any]
    disagreements: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _consensus(verdicts: list[str]) -> str | None:
    counts = Counter(verdicts)
    (top, n), *rest = counts.most_common()
    return None if rest and rest[0][1] == n else top


class ReviewService:
    """Add and read human reviews of a run's case results. Obtain via `EvalKit.reviews`."""

    def __init__(self, kit: EvalKit):
        self.kit = kit

    def add(
        self,
        run_id: str,
        case_key: str,
        *,
        reviewer: str,
        verdict: str,
        evaluator_key: str | None = None,
        score: float | None = None,
        comment: str | None = None,
        sample: str = "targeted",
    ) -> str:
        """Record a review (append-only). `sample="random"` marks it as drawn from the unbiased
        queue: only those enter calibration statistics."""
        try:
            review = CaseReview.model_validate(
                dict(
                    case_key=case_key, evaluator_key=evaluator_key, reviewer=reviewer,
                    verdict=verdict, score=score, comment=comment, sample=sample,
                )
            )  # fmt: skip
        except Exception as e:  # pydantic ValidationError: report all problems in one message
            raise RunError(f"invalid review: {e}") from e
        return self.kit.store.add_case_review(
            run_id,
            review.case_key,
            evaluator_key=review.evaluator_key,
            reviewer=review.reviewer,
            verdict=review.verdict,
            score=review.score,
            comment=review.comment,
            sample=review.sample,
        )

    def list(self, run_id: str, evaluator_key: str | None = None) -> list[dict[str, Any]]:
        return self.kit.store.case_reviews(run_id, evaluator_key)

    def calibrate(self, run_id: str, evaluator_key: str, *, n_min: int = N_MIN) -> Calibration:
        return calibrate(self.kit, run_id, evaluator_key, n_min=n_min)

    def queue(
        self,
        run_id: str,
        evaluator_key: str,
        *,
        strategy: str = "random",
        n: int = 30,
        seed: int = 0,
    ) -> list[dict[str, str]]:
        return review_queue(self.kit, run_id, evaluator_key, strategy=strategy, n=n, seed=seed)

    def disagreements(self, run_id: str, *, tau: float = 0.25) -> dict[str, Any]:
        return disagreements(self.kit, run_id, tau=tau)


def calibrate(kit: EvalKit, run_id: str, evaluator_key: str, *, n_min: int = N_MIN) -> Calibration:
    """Judge-vs-human agreement for one evaluator key of a run. See the module docstring."""
    outcomes = kit.store.case_outcomes(run_id)
    reviews = kit.store.case_reviews(run_id, evaluator_key)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in reviews:
        by_case[r["case_key"]].append(r)
    excluded = Counter()
    pairs: list[tuple[str, str]] = []  # (evaluator verdict, human consensus)
    confusion = Counter()
    disagreements_: list[dict[str, str]] = []
    for case_key in sorted(by_case):
        random_reviews = [r for r in by_case[case_key] if r["sample"] == "random"]
        if not random_reviews:
            excluded["targeted_only"] += 1
            continue
        consensus = _consensus([r["verdict"] for r in random_reviews])
        if consensus is None:
            excluded["tie"] += 1
            continue
        entry = outcomes.get(case_key)
        ev = None if entry is None else entry["evaluators"].get(evaluator_key)
        if ev is None or ev[0] != "ok" or ev[2] is None:
            excluded["no_verdict"] += 1
            continue
        verdict = ev[2]
        if verdict == "UNCERTAIN":
            excluded["uncertain"] += 1
            continue
        pairs.append((verdict, consensus))
        cell = ("t" if verdict == consensus else "f") + ("p" if verdict == "FAIL" else "n")
        confusion[cell] += 1  # tp: judge FAIL & human FAIL, tn: PASS/PASS, fp/fn: mismatches
        if verdict != consensus:
            disagreements_.append({"case_key": case_key, "evaluator": verdict, "human": consensus})
    tp, tn, fp, fn = confusion["tp"], confusion["tn"], confusion["fp"], confusion["fn"]
    n = len(pairs)
    return Calibration(
        run_id=run_id,
        evaluator_key=evaluator_key,
        n_paired=n,
        n_min=n_min,
        uncalibrated=n < n_min,
        accuracy=(tp + tn) / n if n else None,
        accuracy_ci=stats.wilson(tp + tn, n) if n else None,
        fail_precision=tp / (tp + fp) if tp + fp else None,
        fail_recall=tp / (tp + fn) if tp + fn else None,
        kappa=stats.cohen_kappa(pairs),
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        excluded=dict(excluded),
        reviewer_agreement=_reviewer_agreement(reviews),
        disagreements=disagreements_,
    )


def _reviewer_agreement(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for r in reviews:
        by_item[r["case_key"]][r["reviewer"]] = r["verdict"]  # the reviewer's latest verdict
    pairs: list[tuple[str, str]] = []
    for item in by_item.values():
        names = sorted(item)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs.append((item[a], item[b]))
    if not pairs:
        return {"pairs": 0, "percent": None, "kappa": None}
    return {
        "pairs": len(pairs),
        "percent": sum(1 for a, b in pairs if a == b) / len(pairs),
        "kappa": stats.cohen_kappa(pairs),
    }


def review_queue(
    kit: EvalKit,
    run_id: str,
    evaluator_key: str,
    *,
    strategy: str = "random",
    n: int = 30,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Cases to review next for one evaluator (never already reviewed for it), each with why.

    random / stratified: an unbiased sample for calibration (stratified draws equally from the
    evaluator's PASS and FAIL verdicts). uncertain / disagreement: triage (UNCERTAIN verdicts;
    cases where evaluators disagree). Only the first two should be recorded as `sample="random"`.
    """
    if strategy not in ("random", "stratified", "uncertain", "disagreement"):
        raise RunError(f"unknown strategy {strategy!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise RunError("n must be a positive integer")
    outcomes = kit.store.case_outcomes(run_id)
    done = {r["case_key"] for r in kit.store.case_reviews(run_id, evaluator_key)}
    scored = sorted(
        k
        for k, e in outcomes.items()
        if k not in done and e["evaluators"].get(evaluator_key, ("",))[0] == "ok"
    )
    rng = random.Random(seed)

    def verdict(k: str) -> str | None:
        return outcomes[k]["evaluators"][evaluator_key][2]

    if strategy == "random":
        chosen = rng.sample(scored, min(n, len(scored)))
        return [{"case_key": k, "reason": "random sample"} for k in chosen]
    if strategy == "stratified":
        groups = {v: [k for k in scored if verdict(k) == v] for v in ("PASS", "FAIL")}
        per = max(1, n // 2)
        chosen = []
        for v in ("FAIL", "PASS"):
            chosen += rng.sample(groups[v], min(per, len(groups[v])))
        return [
            {"case_key": k, "reason": f"stratified: judge said {verdict(k)}"} for k in chosen[:n]
        ]
    if strategy == "uncertain":
        keys = [k for k in scored if verdict(k) == "UNCERTAIN"]
        return [{"case_key": k, "reason": "judge verdict is UNCERTAIN"} for k in keys[:n]]
    flagged = {c["case_key"]: c["detail"] for c in disagreements(kit, run_id)["cases"]}
    keys = [k for k in scored if k in flagged]
    return [{"case_key": k, "reason": flagged[k]} for k in keys[:n]]


def disagreements(kit: EvalKit, run_id: str, *, tau: float = 0.25) -> dict[str, Any]:
    """Per-case disagreement between the run's evaluators: opposite verdicts (a judge saying PASS
    where a deterministic check says FAIL is the most telling), and two judges' scores that differ
    by at least `tau`."""
    if not 0 < tau <= 1:
        raise RunError("tau must be within (0, 1]")
    run = kit.runs.get(run_id)
    kinds = {s.key: s.kind for s in run.config.evaluators}
    judges = [k for k, kind in kinds.items() if kind == "llm_judge"]
    outcomes = kit.store.case_outcomes(run_id)
    scores = {k: kit.store.metric_by_case(run_id, k, "score") for k in judges}
    cases: list[dict[str, str]] = []
    compared = 0
    for case_key in sorted(outcomes):
        verdicts = {
            k: v[2]
            for k, v in outcomes[case_key]["evaluators"].items()
            if v[0] == "ok" and v[2] in ("PASS", "FAIL")
        }
        if len(verdicts) >= 2:
            compared += 1
        keys = sorted(verdicts)
        found = False
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if verdicts[a] != verdicts[b]:
                    a_is_judge, b_is_judge = kinds[a] == "llm_judge", kinds[b] == "llm_judge"
                    kind = (
                        "judge_vs_deterministic"
                        if a_is_judge != b_is_judge
                        else "verdict_disagreement"
                    )
                    cases.append(
                        {"case_key": case_key, "kind": kind,
                         "detail": f"{a} says {verdicts[a]}, {b} says {verdicts[b]}"}
                    )  # fmt: skip
                    found = True
                    break
            if found:
                break
        if not found:
            for i, a in enumerate(judges):
                for b in judges[i + 1 :]:
                    sa, sb = scores[a].get(case_key), scores[b].get(case_key)
                    if sa is not None and sb is not None and abs(sa - sb) >= tau:
                        cases.append(
                            {"case_key": case_key, "kind": "score_disagreement",
                             "detail": f"{a} scored {sa:.2f}, {b} scored {sb:.2f} (>= {tau})"}
                        )  # fmt: skip
    rate = len(cases) / compared if compared else None
    assert rate is None or math.isfinite(rate)
    return {
        "compared": compared,
        "disagreeing": len(cases),
        "rate": rate,
        "tau": tau,
        "cases": cases,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
