"""Aggregation: denominators, coverage, failure attribution, uncertainty, honesty."""

import json
import math
from datetime import UTC, datetime

import pytest
from conftest import case
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import (
    CaseOutcome,
    EvalKit,
    EvaluatorOutcome,
    EvaluatorSpec,
    Failure,
    RunAttempt,
    RunConfig,
)
from evalkit.analysis import MIN_N, summarize, take_snapshot
from evalkit.evaluators import resolve

EM = resolve(EvaluatorSpec(kind="exact_match", name="em"))
JD = EvaluatorSpec(kind="llm_judge", name="quality", params={"provider": "p", "model": "m"})


def make_run(kit, n, specs=(EM,), tags=None, target=None, outputs=True):
    keys = [f"c{i:03}" for i in range(n)]
    kit.datasets.import_cases(
        "qa",
        [
            case(
                k,
                reference="r",
                **({"output": "o"} if outputs else {}),
                **({"tags": tags(i)} if tags else {}),
            )
            for i, k in enumerate(keys)
        ],
    )
    config = RunConfig(evaluators=list(specs), **({"target": target} if target else {}))
    run = kit.runs.create("qa", config)
    kit.runs.start(run.id)
    return run, keys


def target_fail(kit, run, key, cls="target", kind="exception", **kw):
    kit.runs.record_case_result(
        run.id, key, CaseOutcome.fail(Failure(failure_class=cls, kind=kind, message="m"), **kw)
    )


def complete(kit, run, key, output="o", **kw):
    return kit.runs.record_case_result(run.id, key, CaseOutcome.complete(output, **kw))


def ok(kit, cr, spec, value, verdict="PASS", name="match", **detail):
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=spec.key,
            status="ok",
            metrics={name: value},
            verdict=verdict,
            detail=detail,
        ),
    )


def efail(kit, cr, spec, cls, kind):
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=spec.key,
            status="failed",
            failure=Failure(failure_class=cls, kind=kind, message="m"),
        ),
    )


def skip(kit, cr, spec):
    kit.runs.record_evaluator_result(
        cr.id, EvaluatorOutcome(evaluator_key=spec.key, status="skipped", detail={"reason": "t"})
    )


def na(kit, cr, spec):
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(evaluator_key=spec.key, status="not_applicable", detail={"reason": "r"}),
    )


# --- accounting -------------------------------------------------------------------------------


def test_every_case_is_accounted_for_by_exactly_one_outcome(kit):
    run, keys = make_run(kit, 12)
    for k in keys[:5]:  # 5 scored, 3 pass 2 fail
        ok(
            kit,
            complete(kit, run, k),
            EM,
            1.0 if k < "c003" else 0.0,
            "PASS" if k < "c003" else "FAIL",
        )
    na(kit, complete(kit, run, keys[5]), EM)
    efail(kit, complete(kit, run, keys[6]), EM, "evaluator", "refused")
    efail(kit, complete(kit, run, keys[7]), EM, "infrastructure", "rate_limited")
    efail(kit, complete(kit, run, keys[8]), EM, "input", "oversize")
    target_fail(kit, run, keys[9])
    skip(kit, kit.runs.case_result(run.id, keys[9]), EM)
    # keys[10] planned-but-pending, keys[11] has no row at all
    kit.runs.plan(run.id)
    s = summarize(kit, run.id)
    ev = s.evaluators[EM.key]
    assert ev.counts == {
        "ok": 5, "not_applicable": 1, "skipped": 1, "failed_evaluator": 1,
        "failed_infrastructure": 1, "failed_input": 1, "missing": 0,
    }  # fmt: skip
    assert s.case_stage["complete"] == 9 and s.case_stage["failed_target"] == 1
    assert s.case_stage["pending"] == 2 and s.case_stage["missing"] == 0
    assert s.cases == 12 and ev.scored == 5
    # coverage = scored / (cases - not_applicable)
    assert ev.applicable == 11 and ev.coverage == pytest.approx(5 / 11)
    assert s.target_failure_rate == pytest.approx(1 / 12)
    assert s.case_coverage == pytest.approx(9 / 12)


def test_a_case_with_no_evaluator_result_yet_is_counted_as_missing_not_hidden(kit):
    run, keys = make_run(kit, 4)
    ok(kit, complete(kit, run, keys[0]), EM, 1.0)
    complete(kit, run, keys[1])  # target done, evaluator result not written (crash)
    ev = summarize(kit, run.id).evaluators[EM.key]
    assert ev.counts["missing"] == 1 and ev.scored == 1


def test_a_target_failure_lowers_coverage_and_is_never_a_score(kit):
    run, keys = make_run(kit, 10)
    for k in keys[:8]:
        ok(kit, complete(kit, run, k), EM, 1.0)
    for k in keys[8:]:
        target_fail(kit, run, k, kind="timeout")
        skip(kit, kit.runs.case_result(run.id, k), EM)
    s = summarize(kit, run.id, min_coverage=0.9)
    ev = s.evaluators[EM.key]
    assert ev.coverage == 0.8 and not ev.sufficient_coverage
    m = ev.metrics["match"]
    assert m.observed_mean == 1.0  # over the scored cases only, no invented zeros
    assert m.value is None and not m.reliable and "below the required 90%" in m.note
    assert s.target_failure_rate == 0.2
    assert ev.pass_rate_strict == 0.8  # strict: failed cases count as non-passes
    assert ev.pass_rate["observed"] == 1.0 and ev.pass_rate["value"] is None
    assert any("coverage 80.0% is below" in w for w in s.warnings)


def test_a_headline_is_given_when_coverage_is_sufficient(kit):
    run, keys = make_run(kit, 40)
    for k in keys:
        ok(kit, complete(kit, run, k), EM, 1.0)
    m = summarize(kit, run.id).evaluators[EM.key].metrics["match"]
    assert m.value == 1.0 and m.reliable and m.note is None


def test_min_coverage_is_validated_and_zero_always_gives_a_headline(kit):
    run, keys = make_run(kit, 4)
    ok(kit, complete(kit, run, keys[0]), EM, 0.0, "FAIL")
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            summarize(kit, run.id, min_coverage=bad)
    assert summarize(kit, run.id, min_coverage=0.0).evaluators[EM.key].metrics["match"].value == 0.0


def test_no_results_yet_is_summarized_without_inventing_numbers(kit):
    run, _ = make_run(kit, 3)
    ev = summarize(kit, run.id).evaluators[EM.key]
    assert ev.metrics == {} and ev.coverage == 0.0 and ev.pass_rate is None


def test_all_cases_not_applicable_has_undefined_coverage_not_a_division_error(kit):
    run, keys = make_run(kit, 3)
    for k in keys:
        na(kit, complete(kit, run, k), EM)
    ev = summarize(kit, run.id).evaluators[EM.key]
    assert ev.applicable == 0 and ev.coverage is None and not ev.sufficient_coverage


# --- metric statistics ------------------------------------------------------------------------


def test_binary_metrics_get_a_wilson_interval_and_continuous_ones_a_seeded_bootstrap(kit):
    spec = resolve(EvaluatorSpec(kind="retrieval", name="r", params={"k": [1]}))
    run, keys = make_run(kit, 40, specs=(EM, spec))
    for i, k in enumerate(keys):
        cr = complete(kit, run, k)
        ok(kit, cr, EM, 1.0 if i < 30 else 0.0, "PASS" if i < 30 else "FAIL")
        ok(kit, cr, spec, (i % 10) / 10, None, name="mrr")
    s = summarize(kit, run.id, seed=3)
    m = s.evaluators[EM.key].metrics["match"]
    assert (m.binary, m.ci_method, m.observed_mean) == (True, "wilson", 0.75)
    assert m.ci_low < 0.75 < m.ci_high and 0.58 < m.ci_low and m.ci_high < 0.88
    mrr = s.evaluators[spec.key].metrics["mrr"]
    assert (mrr.binary, mrr.ci_method) == (False, "bootstrap")
    assert mrr.minimum == 0.0 and mrr.maximum == 0.9 and mrr.observed_mean == pytest.approx(0.45)
    assert mrr.p10 <= mrr.median <= mrr.p90
    assert summarize(kit, run.id, seed=3).evaluators[spec.key].metrics["mrr"] == mrr  # reproducible
    other = summarize(kit, run.id, seed=4).evaluators[spec.key].metrics["mrr"]
    assert (other.ci_low, other.ci_high) != (mrr.ci_low, mrr.ci_high)


def test_a_small_sample_is_flagged(kit):
    run, keys = make_run(kit, MIN_N - 1)
    for k in keys:
        ok(kit, complete(kit, run, k), EM, 1.0)
    m = summarize(kit, run.id).evaluators[EM.key].metrics["match"]
    assert "small sample" in m.note and m.value == 1.0


def test_pass_rate_is_over_confident_verdicts_with_bounds_for_uncertain(kit):
    run, keys = make_run(kit, 10)
    verdicts = ["PASS"] * 5 + ["FAIL"] * 3 + ["UNCERTAIN"] * 2
    for k, v in zip(keys, verdicts, strict=True):
        ok(kit, complete(kit, run, k), EM, 1.0, v)
    pr = summarize(kit, run.id, min_coverage=0.5).evaluators[EM.key].pass_rate
    assert pr["n"] == 8 and pr["value"] == 5 / 8 and pr["uncertain"] == 2
    assert pr["lower_bound"] == 0.5 and pr["upper_bound"] == 0.7  # uncertain as all-FAIL / all-PASS
    assert pr["ci_low"] < 5 / 8 < pr["ci_high"]


def test_latency_tokens_retries_and_cost_estimates(kit):
    run, keys = make_run(kit, 4)
    now = datetime.now(UTC)
    for i, k in enumerate(keys):
        cr = complete(kit, run, k, duration_ms=100 * (i + 1))
        attempts = [
            RunAttempt.succeeded(
                1, now, 5, model="m1", provider="p", input_tokens=1000, output_tokens=500
            )
        ]
        if i == 0:
            attempts = [
                RunAttempt.failed(
                    1,
                    now,
                    5,
                    Failure(failure_class="infrastructure", kind="timeout", message="t"),
                    model="m1",
                    provider="p",
                ),
                RunAttempt.succeeded(
                    2, now, 5, model="m1", provider="p", input_tokens=1000, output_tokens=500
                ),
            ]
        kit.runs.record_evaluator_result(
            cr.id,
            EvaluatorOutcome(
                evaluator_key=EM.key, status="ok", metrics={"match": 1.0}, duration_ms=10 * (i + 1)
            ),
            attempts=attempts,
        )
    s = summarize(kit, run.id, prices={"m1": (3.0, 15.0)})
    assert s.target_latency_ms == {"n": 4, "mean": 250.0, "p50": 250.0, "p95": pytest.approx(385.0)}
    assert s.evaluators[EM.key].latency_ms["p50"] == 25.0
    u = s.usage
    assert (u["attempts"], u["failed_attempts"], u["retries"]) == (5, 1, 1)
    assert (u["input_tokens"], u["output_tokens"]) == (4000, 2000) and u["tokens_per_case"] == 1500
    assert u["cost_usd_estimate"] == pytest.approx((4000 * 3 + 2000 * 15) / 1e6)
    assert u["cost_per_1k_cases_usd"] == pytest.approx(u["cost_usd_estimate"] / 4 * 1000)
    assert u["unpriced_attempts"] == 0
    assert summarize(kit, run.id).usage["cost_usd_estimate"] is None  # no prices: no guess
    unknown = summarize(kit, run.id, prices={"other": (1.0, 1.0)}).usage
    assert unknown["unpriced_attempts"] == 5 and unknown["cost_per_1k_cases_usd"] is None


def test_tag_slices_need_a_minimum_size(kit):
    run, keys = make_run(kit, 30, tags=lambda i: ["big"] if i < 20 else ["tiny"] if i < 25 else [])
    for i, k in enumerate(keys):
        ok(kit, complete(kit, run, k), EM, 1.0 if i % 2 else 0.0)
    slices = summarize(kit, run.id, min_coverage=0.5).slices
    assert set(slices) == {"big"} and slices["big"][f"{EM.key}.match"] == {"n": 20, "mean": 0.5}


def test_a_judge_score_that_tracks_output_length_is_flagged(kit):
    judge = EvaluatorSpec(kind="llm_judge", name="q", params={"provider": "p", "model": "m"})
    run, keys = make_run(kit, 20, specs=(judge,))
    for i, k in enumerate(keys):
        cr = complete(kit, run, k, output="x" * (10 + i * 5))
        ok(kit, cr, judge, i / 20, "PASS", name="score")
    s = summarize(kit, run.id, min_coverage=0.5)
    lb = s.evaluators[judge.key].length_bias
    assert lb["n"] == 20 and lb["spearman"] == pytest.approx(1.0)
    assert any("verbosity bias" in w for w in s.warnings)


def test_self_preference_and_unpinned_target_warnings(kit):
    from evalkit import TargetSpec

    judge = EvaluatorSpec(
        kind="llm_judge", name="q", params={"provider": "bedrock", "model": "anthropic.claude-opus"}
    )
    target = TargetSpec(kind="model", identity={"model": "anthropic.claude-haiku"})
    run, _ = make_run(kit, 3, specs=(judge,), target=target)
    warnings = summarize(kit, run.id).warnings
    assert any("same family as the target" in w for w in warnings)
    kit.datasets.import_cases("other", [case("a")])
    unpinned = kit.runs.create(
        "other",
        RunConfig(target=TargetSpec(kind="callable", identity={"name": "f", "fingerprint": None})),
    )
    assert any("unpinned target" in w for w in summarize(kit, unpinned.id).warnings)


def test_must_pass_failures_are_counted(kit):
    run, keys = make_run(kit, 5)
    for i, k in enumerate(keys):
        ok(kit, complete(kit, run, k), EM, 1.0, failed_gates=["safety"] if i < 2 else [])
    assert summarize(kit, run.id).evaluators[EM.key].must_pass_failures == 2


# --- snapshots --------------------------------------------------------------------------------


def test_snapshots_are_stored_immutable_and_recomputable_from_the_rows(kit, store):
    run, keys = make_run(kit, 6)
    for k in keys:
        ok(kit, complete(kit, run, k), EM, 1.0)
    kit.runs.transition(run.id, "succeeded")
    first = take_snapshot(kit, run.id)
    stored = store.latest_summary(run.id)
    assert stored["computed_at"] == first and stored["run_id"] == run.id
    recomputed = summarize(kit, run.id).to_dict()
    assert {
        k: v for k, v in stored.items() if k not in ("computed_at", "evalkit_version")
    } == recomputed
    json.dumps(stored, allow_nan=False)  # every number finite
    with pytest.raises(Exception, match="snapshots"):
        store._conn.execute("UPDATE run_summaries SET summary_json = '{}'")
    with pytest.raises(Exception, match="snapshots"):
        store._conn.execute("DELETE FROM run_summaries")


def test_unknown_runs_raise(kit):
    from evalkit import RunError

    with pytest.raises(RunError):
        summarize(kit, "nope")


# --- properties -------------------------------------------------------------------------------

outcome_kinds = st.sampled_from(
    ["ok_pass", "ok_fail", "na", "e_eval", "e_infra", "e_input", "target", "pending", "crash"]
)


@settings(max_examples=60, deadline=None)
@given(kinds=st.lists(outcome_kinds, min_size=1, max_size=25), min_cov=st.floats(0, 1))
def test_property_the_accounting_identity_holds_and_numbers_stay_in_range(kinds, min_cov):
    kit = EvalKit.open(":memory:")
    run, keys = make_run(kit, len(kinds))
    for k, kind in zip(keys, kinds, strict=True):
        if kind == "pending":
            kit.runs.plan(run.id)
        elif kind == "target":
            target_fail(kit, run, k)
            skip(kit, kit.runs.case_result(run.id, k), EM)
        else:
            cr = complete(kit, run, k)
            if kind == "ok_pass":
                ok(kit, cr, EM, 1.0, "PASS")
            elif kind == "ok_fail":
                ok(kit, cr, EM, 0.0, "FAIL")
            elif kind == "na":
                na(kit, cr, EM)
            elif kind == "e_eval":
                efail(kit, cr, EM, "evaluator", "refused")
            elif kind == "e_infra":
                efail(kit, cr, EM, "infrastructure", "timeout")
            elif kind == "e_input":
                efail(kit, cr, EM, "input", "oversize")
    s = summarize(kit, run.id, min_coverage=min_cov)
    ev = s.evaluators[EM.key]
    terminal = s.case_stage["complete"] + s.case_stage["failed_target"]
    assert sum(ev.counts.values()) == terminal  # each finished case has exactly one outcome here
    assert s.case_stage["pending"] + terminal + s.case_stage["missing"] == s.cases == len(kinds)
    if ev.coverage is not None:
        assert 0.0 <= ev.coverage <= 1.0
        assert ev.sufficient_coverage == (ev.coverage >= min_cov)
    for m in ev.metrics.values():
        assert 0.0 <= m.minimum <= m.observed_mean <= m.maximum <= 1.0
        assert 0.0 <= m.ci_low <= m.ci_high <= 1.0
        assert (m.value is None) == (not m.reliable)
    if ev.pass_rate_strict is not None:
        assert (
            ev.pass_rate_strict <= (ev.pass_rate["observed"] or 0) + 1e-12
            or ev.pass_rate["observed"] is None
        )
    json.dumps(s.to_dict(), allow_nan=False)
    assert all(math.isfinite(x) for x in [s.target_failure_rate or 0.0, s.case_coverage or 0.0])
    kit.close()


def test_an_interval_depends_on_the_values_not_on_the_order_they_were_stored():
    """Regression: rows come back in arbitrary (random-id) order; a seeded bootstrap over an
    unsorted list changed with it, so 'reproducible' reports were not."""
    import random

    from evalkit.analysis import summarize_metric

    rng = random.Random(9)
    values = [rng.random() for _ in range(40)]
    assert len(set(values)) == 40
    reference = summarize_metric("m", values, reliable=True, seed=3)
    for i in range(20):
        shuffled = values[:]
        random.Random(i).shuffle(shuffled)
        assert summarize_metric("m", shuffled, reliable=True, seed=3) == reference
