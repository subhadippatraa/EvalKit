"""P1.1 evaluation validity (audit P1-8, P1-10, P1-11, P1-12), through the real executor."""

import pytest
from conftest import judged

from evalkit import EvalKit, EvaluatorSpec, Rubric
from evalkit.compare import Gates, MetricGate, compare, evaluate_gates
from evalkit.evaluators import EVALUATOR_KINDS, EvaluatorKind, llm_judge_spec
from evalkit.llm import STOP_END, STOP_MAX_TOKENS, LLMResponse, Usage
from evalkit.targets import CallableTarget, ModelTarget, PrecomputedTarget, TargetOutput

N = 60
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}
CITE = EvaluatorSpec(kind="citation_check", name="cite")
CITE_GATE = MetricGate("citation_check:cite.citation_validity", "higher", delta=0.0)


@pytest.fixture
def kit(tmp_path):
    k = EvalKit.open(tmp_path / "v.db")
    yield k
    k.close()


def rag_dataset(kit, n=N):
    kit.datasets.import_cases(
        "rag",
        [
            {
                "case_key": f"k{i:03}",
                "prompt": f"q{i}",
                "reference": "r",
                "retrieved": ["d1", "d2"],
                "relevance": {"d1": 1},
            }
            for i in range(n)
        ],
    )


def run_callable(kit, ref, fn, fingerprint, evaluators, name="rag", policy=FAST):
    target = CallableTarget(fn, name=name, fingerprint=fingerprint)
    run = kit.controller.create(ref, target=target, evaluators=evaluators, policy=policy)
    report = kit.controller.execute(run.id, target=target)
    assert report.status == "succeeded", (report.status, report.stop_reason)
    return run


def cites(_):
    return TargetOutput("the answer [1]", retrieved=["d1", "d2"])


def cites_sometimes(share_every):
    def fn(inp):
        n = int(inp.case_key[1:])
        text = "the answer [9]" if n % share_every == 0 else "the answer"
        return TargetOutput(text, retrieved=["d1", "d2"])

    return fn


# --- not_applicable: a candidate must not hide a regression by leaving the metric ---------


def test_the_summary_states_the_not_applicable_share_and_warns_when_it_dominates(kit):
    rag_dataset(kit)
    run = run_callable(kit, "rag", cites_sometimes(10), "B", [CITE])
    summary = kit.summarize(run.id)
    (ev,) = summary.evaluators.values()
    assert ev.counts["not_applicable"] == 54 and ev.not_applicable_share == pytest.approx(0.9)
    assert any("not applicable" in w and "90" in w for w in summary.warnings)


def test_a_small_not_applicable_share_is_not_a_warning(kit):
    rag_dataset(kit)
    run = run_callable(kit, "rag", cites, "A", [CITE])
    (ev,) = kit.summarize(run.id).evaluators.values()
    assert ev.not_applicable_share == 0.0
    assert not any("not applicable" in w for w in kit.summarize(run.id).warnings)


def test_a_candidate_that_stops_citing_is_flagged_and_fails_the_declared_gate(kit):
    rag_dataset(kit)
    base = run_callable(kit, "rag", cites, "A", [CITE])
    cand = run_callable(kit, "rag", cites_sometimes(10), "B", [CITE])
    cmp = compare(kit, base.id, cand.id)
    (key,) = cmp.not_applicable["candidate"]
    assert cmp.not_applicable["baseline"][key] == 0.0
    assert cmp.not_applicable["candidate"][key] == pytest.approx(0.9)
    assert any("not applicable" in w and "differs" in w for w in cmp.warnings)
    gate = evaluate_gates(kit, cand.id, Gates(metrics=(CITE_GATE,)), cmp)
    assert gate.status == "fail" and gate.exit_code == 3
    assert any(f["code"] == "not_applicable_asymmetry" for f in gate.failures)


def test_the_same_exclusions_on_both_sides_are_not_a_finding(kit):
    rag_dataset(kit)
    base = run_callable(kit, "rag", cites_sometimes(10), "A", [CITE])
    cand = run_callable(kit, "rag", cites_sometimes(10), "B", [CITE])
    cmp = compare(kit, base.id, cand.id)
    assert not any("differs" in w and "not applicable" in w for w in cmp.warnings)
    gate = evaluate_gates(kit, cand.id, Gates(metrics=(CITE_GATE,)), cmp)
    assert not any(f["code"].startswith("not_applicable") for f in gate.failures)


def test_gates_can_cap_the_not_applicable_share_outright_and_tune_the_asymmetry(kit):
    rag_dataset(kit)
    base = run_callable(kit, "rag", cites_sometimes(10), "A", [CITE])
    cand = run_callable(kit, "rag", cites_sometimes(10), "B", [CITE])
    cmp = compare(kit, base.id, cand.id)
    capped = Gates(metrics=(CITE_GATE,), max_na_share=0.5)
    codes = {f["code"] for f in evaluate_gates(kit, cand.id, capped, cmp).failures}
    assert "not_applicable_share" in codes
    off = Gates(metrics=(CITE_GATE,), max_na_asymmetry=None)  # explicitly not checked
    other = run_callable(kit, "rag", cites, "C", [CITE])
    cmp2 = compare(kit, other.id, cand.id)
    assert not any(
        f["code"] == "not_applicable_asymmetry"
        for f in evaluate_gates(kit, cand.id, off, cmp2).failures
    )


def test_the_new_gate_settings_parse_from_a_file():
    g = Gates.from_mapping(
        {"max_na_share": 0.3, "max_na_asymmetry": 0.1, "max_truncated_share": 0.05}
    )
    assert (g.max_na_share, g.max_na_asymmetry, g.max_truncated_share) == (0.3, 0.1, 0.05)
    assert Gates().max_na_asymmetry == 0.05  # on by default
    for bad in ({"max_na_share": 1.5}, {"max_na_asymmetry": -0.1}, {"max_truncated_share": "x"}):
        with pytest.raises(ValueError):
            Gates.from_mapping(bad)


# --- truncation: a cut-off answer is not silently a complete one ---------


class Model:
    provider, model = "fake", "m1"

    def __init__(self, truncate=lambda n: False):
        self.truncate, self.n = truncate, 0

    def call(self, req):
        self.n += 1
        n = int(req.user[1:])
        stop = STOP_MAX_TOKENS if self.truncate(n) else STOP_END
        return LLMResponse(
            text=f"answer {n}", stop_reason=stop, usage=Usage(3, 2), request_id=f"req-{n}",
            provider_stop=f"stopReason={stop}",
        )  # fmt: skip


def model_run(kit, truncate, name="qa"):
    kit.datasets.import_cases(
        name, [{"case_key": f"k{i:03}", "prompt": f"q{i}", "reference": "x"} for i in range(N)]
    )
    target = ModelTarget(Model(truncate), "{prompt}")
    em = EvaluatorSpec(kind="exact_match", name="em")
    run = kit.controller.create(name, target=target, evaluators=[em], policy=FAST)
    assert kit.controller.execute(run.id, target=target).status == "succeeded"
    return run


def test_the_targets_stop_reason_and_truncation_are_stored_with_the_output(kit):
    run = model_run(kit, lambda n: n % 4 == 0)
    cr = kit.runs.case_result(run.id, "k004")
    assert cr.meta["truncated"] is True and cr.meta["stop_reason"] == "max_tokens"
    ok = kit.runs.case_result(run.id, "k005")
    assert ok.meta["truncated"] is False and ok.meta["stop_reason"] == "end_turn"
    assert kit.runs.verify(run.id).ok


def test_truncated_answers_are_counted_warned_about_and_reported(kit):
    run = model_run(kit, lambda n: n % 4 == 0)
    summary = kit.summarize(run.id)
    assert summary.truncated_outputs == 15
    assert any("truncated" in w and "15" in w for w in summary.warnings)
    from evalkit.report import render_report

    html = render_report(kit, run.id)
    assert "Truncated outputs" in html and "truncated" in html


def test_evaluators_can_see_whether_the_output_was_truncated(kit, monkeypatch):
    seen = []

    class Spy:
        kind, requires = "spy", frozenset()

        def __init__(self, spec):
            self.spec, self.key = spec, spec.key

        def evaluate(self, inp, call):
            from evalkit.evaluators import EvalOutcome

            seen.append((inp.case_key, inp.target_meta))
            return EvalOutcome({"seen": 1.0}, "PASS", {})

    monkeypatch.setitem(
        EVALUATOR_KINDS,
        "spy",
        EvaluatorKind("spy", 1, frozenset(), lambda p: p, lambda s, c: Spy(s)),
    )
    kit.datasets.import_cases("qa", [{"case_key": f"k{i:03}", "prompt": f"q{i}"} for i in range(4)])
    target = ModelTarget(Model(lambda n: n == 2), "{prompt}")
    spec = EvaluatorSpec(kind="spy", name="s")
    run = kit.controller.create("qa", target=target, evaluators=[spec], policy=FAST)
    kit.controller.execute(run.id, target=target)
    by_key = dict(seen)
    assert by_key["k002"]["truncated"] is True and by_key["k001"]["truncated"] is False


def test_the_judge_is_never_told_about_truncation(kit):
    """Provenance must not reach the judge's prompt (leakage): the flag is data, not prompt."""
    prompts = []

    class Judge:
        provider, model = "fake", "judge-1"

        def call(self, req):
            prompts.append(req.user)
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=Usage(1, 1))

    kit.datasets.import_cases("qa", [{"case_key": f"k{i:03}", "prompt": f"q{i}"} for i in range(3)])
    target = ModelTarget(Model(lambda n: True), "{prompt}")
    spec = llm_judge_spec("quality", Rubric.from_dict({"quality": "Good?"}), Judge())
    run = kit.controller.create("qa", target=target, evaluators=[spec], policy=FAST)
    kit.controller.execute(run.id, target=target, clients=[Judge()])
    assert prompts and not any("truncat" in p.lower() or "max_tokens" in p for p in prompts)


def test_a_candidate_that_truncates_more_is_warned_about_and_can_be_gated(kit):
    base = model_run(kit, lambda n: False, name="a")
    cand = model_run(kit, lambda n: n % 2 == 0, name="b")
    cmp = compare(kit, base.id, cand.id)
    assert any("truncated" in w for w in cmp.warnings)
    gate = evaluate_gates(kit, cand.id, Gates(max_truncated_share=0.1), cmp)
    assert any(f["code"] == "truncated_share" for f in gate.failures)
    assert gate.status == "fail"


# --- unpinned callable targets: gates refuse them ---------


def two_qa_runs(kit, fp_a, fp_b):
    kit.datasets.import_cases(
        "qa", [{"case_key": f"k{i:03}", "prompt": f"q{i}", "reference": "ok"} for i in range(N)]
    )
    em = [EvaluatorSpec(kind="exact_match", name="em")]
    good = run_callable(kit, "qa", lambda i: "ok", fp_a, em, name="app")
    bad = run_callable(
        kit, "qa", lambda i: "ok" if int(i.case_key[1:]) % 2 else "no", fp_b, em, name="app"
    )
    return good, bad


EM_GATE = Gates(metrics=(MetricGate("exact_match:em.match", "higher", delta=0.02),))


def test_a_gate_refuses_an_unpinned_callable_even_when_it_would_pass(kit):
    good, bad = two_qa_runs(kit, None, None)
    cmp = compare(kit, good.id, good.id)  # a run against itself: nothing regressed
    result = evaluate_gates(kit, good.id, EM_GATE, cmp)
    assert result.status == "fail" and any(f["code"] == "unpinned_target" for f in result.failures)


def test_a_pinned_callable_passes_the_pinning_rule(kit):
    good, bad = two_qa_runs(kit, "v1", "v2")
    result = evaluate_gates(kit, bad.id, EM_GATE, compare(kit, good.id, bad.id))
    assert not any(f["code"] == "unpinned_target" for f in result.failures)
    assert any(f["code"] == "regression" for f in result.failures)


def test_unpinning_can_be_allowed_explicitly(kit):
    good, bad = two_qa_runs(kit, None, None)
    gates = Gates(metrics=EM_GATE.metrics, allow_unpinned=True)
    result = evaluate_gates(kit, bad.id, gates, compare(kit, good.id, bad.id))
    assert not any(f["code"] == "unpinned_target" for f in result.failures)


def test_two_unpinned_runs_are_never_called_run_to_run_noise(kit):
    good, bad = two_qa_runs(kit, None, None)
    cmp = compare(kit, good.id, bad.id)
    assert not cmp.same_identity
    assert not any("run-to-run noise" in i for i in cmp.informational)
    assert any("unpinned" in i for i in cmp.informational)
    assert cmp.noise_floor == {}


# --- judge-check identity: it checks the evaluator a run actually uses ---------


class GoodJudge:
    provider, model = "fake", "judge-1"

    def call(self, req):
        from evalkit.judgecheck import builtin_cases

        for c in builtin_cases():
            if f"\n{c.output}\n" in req.user and c.prompt in req.user:
                return LLMResponse(
                    payload=judged(correctness=5 if c.expected == "PASS" else 1),
                    stop_reason="tool_use",
                    usage=Usage(1, 1),
                )
        raise AssertionError("unknown fixture case")


def test_a_judge_check_of_a_runs_evaluator_is_stored_under_that_evaluators_key_and_shown(kit):
    from evalkit.judgecheck import BUILTIN_RUBRIC, judge_check

    judge = GoodJudge()
    spec = llm_judge_spec("quality", BUILTIN_RUBRIC, judge)  # what a run would freeze
    kit.datasets.import_cases(
        "qa", [{"case_key": f"k{i}", "prompt": "What is 2+2?", "output": "4"} for i in range(3)]
    )
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[spec], policy=FAST)
    kit.controller.execute(run.id, clients=[judge])
    run_spec = run.config.evaluators[0]
    result = judge_check(kit, run_spec, judge)  # the run's own frozen spec, not a fresh one
    assert result.evaluator_key == run_spec.key
    from evalkit.report import render_report

    html = render_report(kit, run.id)
    assert "17/17 correct" in html and "not checked" not in html
