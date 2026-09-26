"""Paired comparison, decision rules, gates, and the comparator's own statistical self-test."""

import random

import pytest

from evalkit import (
    CaseOutcome,
    EvaluatorOutcome,
    EvaluatorSpec,
    Failure,
    RunConfig,
    TargetSpec,
)
from evalkit import compare as C
from evalkit.compare import (
    ComparisonError,
    Confounder,
    Gates,
    MetricGate,
    compare,
    decide,
    evaluate_gates,
    resolve_baseline,
)
from evalkit.evaluators import resolve

EM = resolve(EvaluatorSpec(kind="exact_match", name="em"))


def dataset(kit, n, name="qa", **extra):
    kit.datasets.import_cases(
        name,
        [
            {"case_key": f"c{i:03}", "prompt": "p", "reference": "r", "output": "o", **extra}
            for i in range(n)
        ],
    )


def build_run(
    kit, scores, *, spec=EM, name="match", ref="qa", config=None, finish=True, target_fail=()
):
    """A run whose case i scored scores[i] (None = the target failed on it)."""
    run = kit.runs.create(ref, config or RunConfig(evaluators=[spec]))
    kit.runs.start(run.id)
    for i, v in enumerate(scores):
        key = f"c{i:03}"
        if v is None:
            kit.runs.record_case_result(
                run.id,
                key,
                CaseOutcome.fail(Failure(failure_class="target", kind="exception", message="m")),
            )
            kit.runs.record_evaluator_result(
                kit.runs.case_result(run.id, key).id,
                EvaluatorOutcome(evaluator_key=spec.key, status="skipped", detail={"reason": "t"}),
            )
            continue
        cr = kit.runs.record_case_result(run.id, key, CaseOutcome.complete("o", duration_ms=100))
        kit.runs.record_evaluator_result(
            cr.id,
            EvaluatorOutcome(
                evaluator_key=spec.key, status="ok", metrics={name: v},
                verdict="PASS" if v >= 0.5 else "FAIL",
            ),
        )  # fmt: skip
    if finish:
        for cr in kit.runs.case_results(run.id):  # evaluators the spec list adds: nothing to score
            have = {e.evaluator_key for e in kit.runs.evaluator_results(cr.id)}
            for key in kit.runs.get(run.id).config.evaluator_keys:
                if key not in have:
                    kit.runs.record_evaluator_result(
                        cr.id,
                        EvaluatorOutcome(
                            evaluator_key=key,
                            status="not_applicable" if cr.status == "complete" else "skipped",
                            detail={"reason": "not scored here"},
                        ),
                    )
        kit.runs.transition(run.id, "succeeded")
    return run


def gate(**kw):
    return Gates.from_mapping({"gates": {"exact_match:*.match": {"direction": "higher", **kw}}})


def one(cmp):
    (m,) = [m for m in cmp.metrics if m.evaluator != "run"]
    return m


@pytest.fixture
def k300(kit):
    dataset(kit, 300)
    return kit


# --- pairing and statistics -------------------------------------------------------------------


def test_paired_counts_differences_and_worst_cases_are_exact(kit):
    dataset(kit, 40)
    base = [1.0] * 40
    cand = [1.0] * 30 + [0.0] * 6 + [1.0] * 4  # candidate lower on 6 cases
    cmp = compare(kit, build_run(kit, base).id, build_run(kit, cand).id)
    m = one(cmp)
    assert (m.n_paired, m.candidate_lower, m.candidate_higher, m.tied) == (40, 6, 0, 34)
    assert m.diff == pytest.approx(-6 / 40) and m.mean_baseline == 1.0 and m.mean_candidate == 0.85
    assert m.ci_low <= m.diff <= m.ci_high and m.binary and m.ci_method == "bootstrap"
    assert [w["case_key"] for w in m.worst_cases] == [f"c{i:03}" for i in range(30, 36)]
    assert m.best_cases == [] and m.p_value == pytest.approx(2 * (1 / 2**6))  # McNemar 0 vs 6
    assert m.mde is not None and m.sd > 0


def test_a_wins_losses_ties_mix(kit):
    dataset(kit, 40)
    base = [0.0] * 10 + [1.0] * 10 + [1.0] * 20
    cand = [1.0] * 10 + [0.0] * 10 + [1.0] * 20
    m = one(compare(kit, build_run(kit, base).id, build_run(kit, cand).id))
    assert (m.candidate_higher, m.candidate_lower, m.tied) == (10, 10, 20)
    assert m.diff == 0 and m.p_value == 1.0


def test_continuous_metrics_have_no_mcnemar_and_a_paired_interval(kit):
    dataset(kit, 40)
    rng = random.Random(1)
    base = [rng.random() for _ in range(40)]
    cand = [min(1.0, b + 0.1) for b in base]
    m = one(compare(kit, build_run(kit, base).id, build_run(kit, cand).id))
    assert not m.binary and m.p_value is None and m.diff > 0.05 and m.ci_low > 0


def test_interval_is_seeded_so_a_comparison_is_reproducible(kit):
    dataset(kit, 50)
    rng = random.Random(2)
    a = build_run(kit, [rng.random() for _ in range(50)])
    b = build_run(kit, [rng.random() for _ in range(50)])
    one_, two = compare(kit, a.id, b.id, seed=5), compare(kit, a.id, b.id, seed=5)
    assert (one(one_).ci_low, one(one_).ci_high) == (one(two).ci_low, one(two).ci_high)
    other = compare(kit, a.id, b.id, seed=6)
    assert (one(other).ci_low, one(other).ci_high) != (one(one_).ci_low, one(one_).ci_high)


def test_only_cases_scored_on_both_sides_are_paired_and_the_rest_are_excluded_by_reason(kit):
    dataset(kit, 40)
    base = [1.0] * 40
    cand = [1.0] * 30 + [None] * 6 + [0.0] * 4
    cmp = compare(kit, build_run(kit, base).id, build_run(kit, cand).id)
    m = one(cmp)
    assert m.n_paired == 34 and cmp.common_cases == 40
    assert cmp.exclusions == {"baseline": {}, "candidate": {"target_failure": 6}}
    assert m.diff == pytest.approx(-4 / 34)  # the six crashed cases are not silently zeros or ones


def test_a_candidate_that_crashes_on_hard_cases_raises_a_survivorship_warning(kit):
    dataset(kit, 50)
    base = [1.0] * 25 + [0.0] * 25  # the second half is hard
    cand = [1.0] * 25 + [None] * 25  # ...and the candidate simply fails on all of it
    cmp = compare(kit, build_run(kit, base).id, build_run(kit, cand).id)
    assert any("survivorship" in w for w in cmp.warnings)
    assert cmp.coverage["candidate"][EM.key] == 0.5 and cmp.coverage["baseline"][EM.key] == 1.0
    rate = next(m for m in cmp.metrics if m.evaluator == "run")
    assert (rate.metric, rate.mean_baseline, rate.mean_candidate) == (
        "target_failure_rate",
        0.0,
        0.5,
    )
    assert rate.p_value == pytest.approx(2 / 2**25)  # 25 discordant pairs, all one way


def test_no_survivorship_warning_when_coverage_matches(kit):
    dataset(kit, 40)
    cmp = compare(kit, build_run(kit, [1.0] * 40).id, build_run(kit, [0.0] * 40).id)
    assert not any("survivorship" in w for w in cmp.warnings)


def test_different_dataset_versions_pair_by_key_and_input_and_report_what_was_not_paired(kit):
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40)
    kit.datasets.import_cases(
        "qa",
        [
            {
                "case_key": f"c{i:03}",
                "prompt": "p",
                "reference": "r" if i < 30 else "a different reference",  # the question changed
                "output": "another output" if i % 2 else "o",  # outputs may differ freely
            }
            for i in range(40)
        ],
    )
    b = build_run(kit, [1.0] * 40, ref="qa@2")
    cmp = compare(kit, a.id, b.id)
    assert not cmp.same_dataset_version and cmp.common_cases == 30 and one(cmp).n_paired == 30
    assert cmp.unpaired == {"input_changed": 10, "only_baseline": 0, "only_candidate": 0}
    assert any("10 case(s)" in w and "NOT paired" in w for w in cmp.warnings)
    assert any("different dataset versions" in i for i in cmp.informational)


def test_cases_present_in_only_one_run_are_counted_and_warned_about(kit):
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40)
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i:03}", "prompt": "p", "reference": "r"} for i in range(35)]
    )
    b = build_run(kit, [1.0] * 35, ref="qa@2")
    cmp = compare(kit, a.id, b.id)
    assert cmp.unpaired == {"input_changed": 0, "only_baseline": 5, "only_candidate": 0}
    assert any("only in the baseline" in w for w in cmp.warnings)


def test_runs_with_nothing_in_common_are_refused(kit):
    dataset(kit, 5, name="one")
    dataset(kit, 5, name="two", reference="a different reference")
    a = build_run(kit, [1.0] * 5, ref="one")
    b = build_run(kit, [1.0] * 5, ref="two")
    with pytest.raises(ComparisonError, match="no case in common"):
        compare(kit, a.id, b.id)


# --- confounders ------------------------------------------------------------------------------


def judge(model="m1", rubric="h1", prompt="p1", temperature=0.0):
    return EvaluatorSpec(
        kind="llm_judge", name="quality",
        params={"provider": "x", "model": model, "rubric_content_hash": rubric,
                "prompt_version": prompt, "temperature": temperature},
    )  # fmt: skip


@pytest.mark.parametrize(
    "changed,expected",
    [
        ({"model": "m2"}, "model: m1 -> m2"),
        ({"rubric": "h2"}, "rubric_content_hash: h1 -> h2"),
        ({"prompt": "p2"}, "prompt_version: p1 -> p2"),
        ({"temperature": 0.7}, "temperature: 0.0 -> 0.7"),
    ],
)
def test_a_different_judge_is_a_named_confounder_and_refused(kit, changed, expected):
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40, spec=judge())
    b = build_run(kit, [1.0] * 40, spec=judge(**changed))
    with pytest.raises(ComparisonError, match="confounded") as info:
        compare(kit, a.id, b.id)
    assert expected in str(info.value) and "llm_judge:quality" in str(info.value)
    cmp = compare(kit, a.id, b.id, allow_confounders=True)  # shown, but flagged
    assert (
        cmp.confounded
        and cmp.confounders[0].kind == "evaluator"
        and expected in cmp.confounders[0].detail
    )
    assert one(cmp).confounded


def test_a_different_scoring_version_is_a_confounder(kit):
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40)
    b = build_run(kit, [1.0] * 40, config=RunConfig(evaluators=[EM], scoring_version=1))
    kit.store._conn.execute("PRAGMA foreign_keys=ON")
    conn = kit.store._conn
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute(
        "UPDATE runs SET config_json = replace(config_json, '\"scoring_version\":1', "
        "'\"scoring_version\":2') WHERE id = ?",
        (b.id,),
    )
    conn.commit()
    with pytest.raises(ComparisonError, match="scoring_version"):
        compare(kit, a.id, b.id)
    assert any(
        c.kind == "scoring_version"
        for c in compare(kit, a.id, b.id, allow_confounders=True).confounders
    )


def test_varying_the_target_is_the_point_not_a_confounder(kit):
    dataset(kit, 40)
    t1 = TargetSpec(kind="model", identity={"model": "m1"})
    t2 = TargetSpec(kind="model", identity={"model": "m2"})
    a = build_run(kit, [1.0] * 40, config=RunConfig(target=t1, evaluators=[EM]))
    b = build_run(kit, [0.0] * 40, config=RunConfig(target=t2, evaluators=[EM]))
    cmp = compare(kit, a.id, b.id)
    assert not cmp.confounded and not cmp.same_identity
    assert any("target varied" in i for i in cmp.informational)


def test_evaluators_present_on_one_side_only_are_reported_not_compared(kit):
    dataset(kit, 40)
    other = resolve(EvaluatorSpec(kind="regex", name="r", params={"pattern": "o"}))
    a = build_run(kit, [1.0] * 40)
    b = build_run(kit, [1.0] * 40, config=RunConfig(evaluators=[EM, other]))
    cmp = compare(kit, a.id, b.id)
    assert any("exists only in the candidate" in i for i in cmp.informational)
    assert {m.evaluator for m in cmp.metrics if m.evaluator != "run"} == {"exact_match:em"}


def test_the_confounder_dataclass_and_execution_differences(kit):
    assert Confounder("evaluator", "x").kind == "evaluator"
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40, config=RunConfig(evaluators=[EM], policy={"concurrency": 1}))
    b = build_run(kit, [1.0] * 40, config=RunConfig(evaluators=[EM], policy={"concurrency": 9}))
    cmp = compare(kit, a.id, b.id)
    assert cmp.same_identity and not cmp.confounded
    assert any("execution policy differs" in i for i in cmp.informational)


# --- reruns and noise floor -------------------------------------------------------------------


def test_an_identical_rerun_is_equivalent_and_reports_a_noise_floor(kit):
    dataset(kit, 100)
    rng = random.Random(4)
    scores = [float(rng.random() < 0.8) for _ in range(100)]
    a, b = build_run(kit, scores), build_run(kit, scores)
    cmp = compare(kit, a.id, b.id)
    assert cmp.same_identity and f"{'exact_match:em'}.match" in cmp.noise_floor
    (d,) = evaluate_gates(kit, b.id, gate(delta=0.02), cmp).decisions
    assert d.decision == "equivalent" and d.n_paired == 100 and d.diff == 0


def test_a_rerun_with_a_little_noise_reports_a_defensible_delta(kit):
    dataset(kit, 100)
    rng = random.Random(5)
    base = [float(rng.random() < 0.8) for _ in range(100)]
    noisy = [b if rng.random() > 0.05 else 1 - b for b in base]
    cmp = compare(kit, build_run(kit, base).id, build_run(kit, noisy).id)
    floor = cmp.noise_floor["exact_match:em.match"]
    assert floor > 0 and floor == pytest.approx(
        max(abs(one(cmp).diff), (one(cmp).ci_high - one(cmp).ci_low) / 2)
    )


# --- decisions --------------------------------------------------------------------------------


def synthetic(diff, sd, n, mean_b=0.8):
    import evalkit.compare as c

    b = [mean_b] * n
    half = n // 2
    cvals = [mean_b + diff + sd * (1 if i < half else -1) for i in range(n)]
    keys = [f"k{i}" for i in range(n)]
    return c._paired(
        "e:x",
        "k",
        "k",
        "m",
        keys,
        dict(zip(keys, b, strict=True)),
        dict(zip(keys, cvals, strict=True)),
        0,
        False,
    )


@pytest.mark.parametrize(
    "diff,delta,decision",
    [
        (-0.10, 0.02, "regression"),
        (+0.10, 0.02, "improvement"),
        (0.0, 0.05, "equivalent"),
        (-0.045, 0.05, "inconclusive"),  # the interval straddles the band's edge
        (-0.003, 0.0, "inconclusive"),  # the interval straddles zero
        (-0.01, 0.0, "regression"),
    ],
)
def test_the_decision_rule_compares_the_interval_with_the_tolerance(diff, delta, decision):
    mc = synthetic(diff, 0.03, 60)
    assert decide(mc, MetricGate("e:x.m", "higher", delta=delta), 30).decision == decision


def test_delta_zero_can_regress_or_improve_but_never_be_equivalent():
    assert (
        decide(synthetic(-0.2, 0.01, 60), MetricGate("a.m", "higher", delta=0.0), 30).decision
        == "regression"
    )
    assert (
        decide(synthetic(0.2, 0.01, 60), MetricGate("a.m", "higher", delta=0.0), 30).decision
        == "improvement"
    )
    assert (
        decide(synthetic(0.0, 0.0, 60), MetricGate("a.m", "higher", delta=0.0), 30).decision
        == "inconclusive"
    )


def test_direction_lower_flips_the_meaning_of_a_difference():
    up = synthetic(+0.2, 0.01, 60)
    assert decide(up, MetricGate("a.m", "higher", delta=0.02), 30).decision == "improvement"
    assert decide(up, MetricGate("a.m", "lower", delta=0.02), 30).decision == "regression"


def test_relative_delta_scales_with_the_baseline_mean():
    mc = synthetic(-0.05, 0.005, 60, mean_b=0.5)  # -10 % of the baseline
    assert decide(mc, MetricGate("a.m", "higher", rel_delta=0.05), 30).decision == "regression"
    assert decide(mc, MetricGate("a.m", "higher", rel_delta=0.5), 30).decision != "regression"


@pytest.mark.parametrize("n", [1, 2, 10, 20, 29])
def test_fewer_than_min_n_pairs_is_always_inconclusive_however_large_the_effect(n):
    mc = synthetic(-0.5, 0.0, n)
    d = decide(mc, MetricGate("a.m", "higher", delta=0.01), 30)
    assert d.decision == "inconclusive" and "insufficient_data" in d.reason


def test_an_inconclusive_decision_states_the_minimum_detectable_difference():
    d = decide(synthetic(-0.01, 0.2, 40), MetricGate("a.m", "higher", delta=0.02), 30)
    assert d.decision == "inconclusive" and "cannot detect a change smaller than" in d.reason


def test_decisions_are_explainable_from_their_own_numbers():
    d = decide(synthetic(-0.1, 0.03, 60), MetricGate("e:x.m", "higher", delta=0.02), 30)
    assert (d.n_paired, d.delta) == (60, 0.02) and d.ci_high < -0.02 and "95% CI" in d.reason
    assert d.diff == pytest.approx(-0.1) and d.mde is not None


# --- the comparator evaluated on synthetic data with known effects (design 9.3) ---------------


def simulate(effect_in_mde, seed, sims, n=60, sd=0.15, ci_b=100):
    """Fraction of simulations the comparator calls REGRESSION for a true effect of
    `effect_in_mde` x MDE (0 = no effect)."""
    mde = 2.8 * sd / n**0.5
    rng = random.Random(seed)
    old, C.CI_B = C.CI_B, ci_b
    try:
        hits = 0
        keys = [f"k{i}" for i in range(n)]
        for s in range(sims):
            b = [rng.gauss(0.7, 0.1) for _ in range(n)]
            c = [x - effect_in_mde * mde + rng.gauss(0, sd) for x in b]
            mc = C._paired(
                "e:x",
                "k",
                "k",
                "m",
                keys,
                dict(zip(keys, b, strict=True)),
                dict(zip(keys, c, strict=True)),
                s,
                False,
            )
            hits += (
                decide(mc, MetricGate("e:x.m", "higher", delta=0.0), 30).decision == "regression"
            )
        return hits / sims
    finally:
        C.CI_B = old


def test_false_regression_rate_under_the_null_is_at_most_eight_percent():
    assert simulate(0.0, seed=1, sims=1000) <= 0.08


def test_an_effect_of_one_and_a_half_mde_is_detected_at_least_eighty_percent_of_the_time():
    assert simulate(1.5, seed=2, sims=300) >= 0.80


# --- gates ------------------------------------------------------------------------------------


def test_a_five_point_degradation_over_300_cases_is_a_regression_that_fails_the_gate(k300):
    base = [1.0] * 270 + [0.0] * 30
    cand = base[:]
    for i in range(15):
        cand[i] = 0.0
    a, b = build_run(k300, base), build_run(k300, cand)
    cmp = compare(k300, a.id, b.id)
    r = evaluate_gates(k300, b.id, gate(delta=0.02), cmp)
    assert (r.status, r.exit_code) == ("fail", 3)
    (d,) = r.decisions
    assert d.decision == "regression" and d.diff == pytest.approx(-0.05) and d.n_paired == 300
    assert r.failures[0]["code"] == "regression" and "worse than -0.02" in r.failures[0]["message"]


def test_twenty_paired_cases_are_always_inconclusive(kit):
    dataset(kit, 20)
    a, b = build_run(kit, [1.0] * 20), build_run(kit, [0.0] * 20)
    r = evaluate_gates(kit, b.id, gate(delta=0.02), compare(kit, a.id, b.id))
    assert (r.status, r.exit_code) == ("inconclusive", 0)
    assert r.decisions[0].decision == "inconclusive"
    strict = Gates.from_mapping(
        {"strict": True, "gates": {"exact_match:*.match": {"direction": "higher", "delta": 0.02}}}
    )
    assert evaluate_gates(kit, b.id, strict, compare(kit, a.id, b.id)).exit_code == 4


def test_improvement_and_equivalence_pass_the_gate(k300):
    a = build_run(k300, [1.0] * 200 + [0.0] * 100)
    better = build_run(k300, [1.0] * 260 + [0.0] * 40)
    r = evaluate_gates(k300, better.id, gate(delta=0.02), compare(k300, a.id, better.id))
    assert (r.status, r.decisions[0].decision) == ("pass", "improvement")


def test_a_candidate_below_minimum_coverage_cannot_pass_even_with_perfect_scores(k300):
    a = build_run(k300, [1.0] * 300)
    b = build_run(k300, [1.0] * 200 + [None] * 100)  # every scored case is perfect, a third crashed
    cmp = compare(k300, a.id, b.id)
    r = evaluate_gates(k300, b.id, gate(delta=0.02), cmp)
    assert r.status == "fail" and r.exit_code == 3
    coverage = [f for f in r.failures if f["code"] == "coverage"]
    assert (
        coverage
        and "66.7%" in coverage[0]["message"]
        and "cannot pass on missing data" in coverage[0]["message"]
    )
    assert any("survivorship" in w for w in cmp.warnings)
    lenient = Gates.from_mapping(
        {
            "min_coverage": 0.5,
            "gates": {"exact_match:*.match": {"direction": "higher", "delta": 0.02}},
        }
    )
    assert not [
        f for f in evaluate_gates(k300, b.id, lenient, cmp).failures if f["code"] == "coverage"
    ]


def test_floors_are_absolute_and_need_a_value(k300):
    good, bad = build_run(k300, [1.0] * 280 + [0.0] * 20), build_run(k300, [1.0] * 240 + [0.0] * 60)
    floors = Gates.from_mapping({"gates": {"floor": {"exact_match:*.match": 0.90}}})
    assert evaluate_gates(k300, good.id, floors).status == "pass"
    r = evaluate_gates(k300, bad.id, floors)
    assert r.status == "fail" and "below the floor 0.9" in r.failures[0]["message"]
    missing = Gates.from_mapping({"gates": {"floor": {"retrieval:*.mrr": 0.5}}})
    assert "no evaluator matches" in evaluate_gates(k300, good.id, missing).failures[0]["message"]
    nometric = Gates.from_mapping({"gates": {"floor": {"exact_match:*.nope": 0.5}}})
    assert "no value" in evaluate_gates(k300, good.id, nometric).failures[0]["message"]


def test_a_must_pass_criterion_failure_fails_the_gate(kit):
    dataset(kit, 40)
    run = kit.runs.create("qa", RunConfig(evaluators=[EM]))
    kit.runs.start(run.id)
    for i in range(40):
        cr = kit.runs.record_case_result(run.id, f"c{i:03}", CaseOutcome.complete("o"))
        kit.runs.record_evaluator_result(
            cr.id,
            EvaluatorOutcome(
                evaluator_key=EM.key, status="ok", metrics={"match": 1.0}, verdict="PASS",
                detail={"failed_gates": ["safety"] if i < 3 else []},
            ),
        )  # fmt: skip
    kit.runs.transition(run.id, "succeeded")
    strict = Gates.from_mapping({"must_pass": ["exact_match:*"]})
    r = evaluate_gates(kit, run.id, strict)
    assert (
        r.status == "fail"
        and "3 of 40 scored cases failed a must-pass criterion" in r.failures[0]["message"]
    )
    tolerant = Gates.from_mapping({"must_pass": {"exact_match:*": 0.10}})
    assert evaluate_gates(kit, run.id, tolerant).status == "pass"


def test_a_metric_gate_without_a_baseline_or_without_the_metric_is_inconclusive_never_a_pass(kit):
    dataset(kit, 40)
    a, b = build_run(kit, [1.0] * 40), build_run(kit, [1.0] * 40)
    no_baseline = evaluate_gates(kit, b.id, gate(delta=0.02))
    assert no_baseline.status == "inconclusive" and "no baseline" in no_baseline.decisions[0].reason
    absent = Gates.from_mapping({"gates": {"regex:*.match": {"direction": "higher", "delta": 0.0}}})
    r = evaluate_gates(kit, b.id, absent, compare(kit, a.id, b.id))
    assert r.decisions[0].decision == "inconclusive" and "not compared" in r.decisions[0].reason


def test_target_failure_rate_is_a_lower_is_better_gate(k300):
    a = build_run(k300, [1.0] * 300)
    b = build_run(k300, [1.0] * 270 + [None] * 30)
    cmp = compare(k300, a.id, b.id)
    g = Gates.from_mapping(
        {
            "min_coverage": 0.5,
            "gates": {"run.target_failure_rate": {"direction": "lower", "delta": 0.005}},
        }
    )
    r = evaluate_gates(k300, b.id, g, cmp)
    assert r.status == "fail" and r.decisions[0].decision == "regression"
    assert r.decisions[0].p_value is not None and r.decisions[0].p_value < 0.001


def test_latency_is_compared_by_relative_change_only(kit):
    dataset(kit, 40)
    a = build_run(kit, [1.0] * 40)
    b = build_run(kit, [1.0] * 40)
    kit.store._conn.execute("PRAGMA foreign_keys=ON")
    conn = kit.store._conn
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("UPDATE case_results SET duration_ms = 300 WHERE run_id = ?", (b.id,))
    conn.commit()
    g = Gates.from_mapping(
        {"gates": {"run.latency_p95_ms": {"direction": "lower", "rel_delta": 0.25}}}
    )
    r = evaluate_gates(kit, b.id, g, compare(kit, a.id, b.id))
    assert r.decisions[0].decision == "regression" and "100.0 -> 300" in r.decisions[
        0
    ].reason.replace("100 ", "100.0 ")


def test_no_declared_gates_means_nothing_can_fail(k300):
    a, b = build_run(k300, [1.0] * 300), build_run(k300, [0.0] * 300)
    r = evaluate_gates(k300, b.id, Gates(), compare(k300, a.id, b.id))
    assert r.status == "pass" and any("nothing can fail" in n for n in r.notes)


def test_a_confounded_comparison_carries_a_note_into_the_gate(kit):
    dataset(kit, 40)
    a, b = build_run(kit, [1.0] * 40, spec=judge()), build_run(kit, [1.0] * 40, spec=judge("m2"))
    cmp = compare(kit, a.id, b.id, allow_confounders=True)
    g = Gates.from_mapping({"gates": {"llm_judge:*.match": {"direction": "higher", "delta": 0.0}}})
    assert any("confounded" in n for n in evaluate_gates(kit, b.id, g, cmp).notes)


# --- gate configuration -----------------------------------------------------------------------


def test_gates_parse_from_toml_and_json_files(tmp_path):
    toml = tmp_path / "gates.toml"
    toml.write_text(
        "min_coverage = 0.9\n[gates]\n"
        '"llm_judge:quality.score" = { direction = "higher", delta = 0.02 }\n'
        '"run.latency_p95_ms" = { direction = "lower", rel_delta = 0.25 }\n'
        '[gates.floor]\n"exact_match:*.match" = 0.90\n'
    )
    g = Gates.from_file(toml)
    assert g.min_coverage == 0.9 and dict(g.floors) == {"exact_match:*.match": 0.9}
    assert [(m.ref, m.direction, m.delta, m.rel_delta) for m in g.metrics] == [
        ("llm_judge:quality.score", "higher", 0.02, None),
        ("run.latency_p95_ms", "lower", None, 0.25),
    ]
    js = tmp_path / "gates.json"
    js.write_text(
        '{"gates": {"exact_match:*.match": {"direction": "higher", "delta": 0}}, "strict": true}'
    )
    assert Gates.from_file(js).strict and Gates.from_file(js).metrics[0].delta == 0


@pytest.mark.parametrize(
    "data",
    [
        {"gates": {"a.m": {"direction": "up", "delta": 0.1}}},
        {"gates": {"a.m": {"direction": "higher"}}},
        {"gates": {"a.m": {"direction": "higher", "delta": 0.1, "rel_delta": 0.1}}},
        {"gates": {"a.m": {"direction": "higher", "delta": -1}}},
        {"gates": {"nodot": {"direction": "higher", "delta": 0}}},
        {"gates": {"a.m": {"direction": "higher", "delta": 0, "extra": 1}}},
        {"gates": {"a.m": 0.5}},
        {"min_coverage": 2},
        {"min_n": 1},
        {"unknown": 1},
        {"gates": {"floor": {"nodot": 0.5}}},
        {"gates": {"floor": {"a.m": float("nan")}}},
        {"must_pass": {"sel": 2}},
    ],
)
def test_invalid_gate_configuration_is_refused(data):
    with pytest.raises(ValueError):
        Gates.from_mapping(data)


def test_a_bad_gates_file_is_a_clear_error(tmp_path):
    bad = tmp_path / "g.toml"
    bad.write_text("not = [valid")
    with pytest.raises(ValueError, match="cannot read gates file"):
        Gates.from_file(bad)
    with pytest.raises(OSError):
        Gates.from_file(tmp_path / "missing.toml")


# --- baseline resolution ----------------------------------------------------------------------


def test_tag_baselines_resolve_to_the_latest_succeeded_tagged_run_of_the_dataset(kit, store):
    dataset(kit, 5)
    old = build_run(kit, [1.0] * 5)
    new = build_run(kit, [1.0] * 5)
    unfinished = build_run(kit, [1.0] * 5, finish=False)
    other = build_run(kit, [1.0] * 5)
    for r in (old, new, unfinished):
        store.add_run_tag(r.id, "main")
    candidate = build_run(kit, [1.0] * 5)
    assert resolve_baseline(kit, "tag:main", candidate.id) == new.id
    assert resolve_baseline(kit, other.id, candidate.id) == other.id
    with pytest.raises(ComparisonError, match="no succeeded run with tag 'nightly'"):
        resolve_baseline(kit, "tag:nightly", candidate.id)
    store.add_run_tag(candidate.id, "main")
    assert (
        resolve_baseline(kit, "tag:main", candidate.id) == new.id
    )  # a candidate is never its own baseline
