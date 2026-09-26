"""P1.1 / audit P0-1: a regression must be *detected*, end to end, through the real path:
dataset -> run -> target -> evaluators (the executor, not hand-built rows) -> comparison -> gate.

The original bug: cases were paired on their full content hash, which includes the system's own
`output`, so every case whose output changed dropped out of the comparison and a real regression
came back as "equivalent"."""

import pytest

from evalkit import EvalKit, EvaluatorSpec
from evalkit.compare import Gates, MetricGate, compare, evaluate_gates
from evalkit.targets import CallableTarget, PrecomputedTarget

N = 60
EM = EvaluatorSpec(kind="exact_match", name="em")
GATE = Gates(metrics=(MetricGate("exact_match:em.match", "higher", delta=0.02),))


def good(i: int) -> dict:
    return {"case_key": f"k{i:03}", "prompt": f"q{i}", "reference": f"a{i}", "output": f"a{i}"}


def worse(i: int) -> dict:
    """The candidate's output: wrong for every third case."""
    return good(i) | {"output": "wrong" if i % 3 == 0 else f"a{i}"}


def run_precomputed(kit, ref):
    run = kit.controller.create(ref, target=PrecomputedTarget(), evaluators=[EM])
    assert kit.controller.execute(run.id).status == "succeeded"
    return run


@pytest.fixture
def kit(tmp_path):
    k = EvalKit.open(tmp_path / "e2e.db")
    yield k
    k.close()


def test_a_worse_candidate_is_paired_detected_and_fails_the_gate_precomputed_outputs(kit):
    kit.datasets.import_cases("qa", [good(i) for i in range(N)])  # qa@1: the baseline's outputs
    kit.datasets.import_cases("qa", [worse(i) for i in range(N)])  # qa@2: the candidate's outputs
    baseline, candidate = run_precomputed(kit, "qa@1"), run_precomputed(kit, "qa@2")

    cmp = compare(kit, baseline.id, candidate.id)

    # every case is paired, including the 20 whose output changed
    assert cmp.common_cases == N
    match = next(m for m in cmp.metrics if m.metric == "match" and m.evaluator != "run")
    assert match.n_paired == N
    assert match.mean_baseline == 1.0 and match.mean_candidate == pytest.approx(40 / 60)
    assert match.candidate_lower == 20 and match.candidate_higher == 0
    assert match.ci_high < 0  # the whole interval is below zero: a regression
    # the gate fails, with the regression as the stated reason
    gate = evaluate_gates(kit, candidate.id, GATE, cmp)
    assert gate.status == "fail" and gate.exit_code == 3
    assert [d.decision for d in gate.decisions] == ["regression"]
    assert any(f["code"] == "regression" for f in gate.failures)


def test_a_worse_candidate_is_detected_through_a_dataset_of_another_name(kit):
    kit.datasets.import_cases("system-a", [good(i) for i in range(N)])
    kit.datasets.import_cases("system-b", [worse(i) for i in range(N)])
    baseline, candidate = run_precomputed(kit, "system-a"), run_precomputed(kit, "system-b")
    cmp = compare(kit, baseline.id, candidate.id)
    assert cmp.common_cases == N
    assert evaluate_gates(kit, candidate.id, GATE, cmp).exit_code == 3


def test_a_worse_candidate_is_detected_with_callable_targets(kit):
    """Same guarantee when EvalKit calls the systems itself (the dataset carries no output)."""
    kit.datasets.import_cases(
        "qa", [{"case_key": f"k{i:03}", "prompt": str(i), "reference": f"a{i}"} for i in range(N)]
    )
    old = CallableTarget(lambda inp: f"a{inp.prompt}", name="app", fingerprint="v1")
    new = CallableTarget(
        lambda inp: "wrong" if int(inp.prompt) % 3 == 0 else f"a{inp.prompt}",
        name="app",
        fingerprint="v2",
    )
    runs = []
    for target in (old, new):
        run = kit.controller.create("qa", target=target, evaluators=[EM])
        assert kit.controller.execute(run.id, target=target).status == "succeeded"
        runs.append(run)
    cmp = compare(kit, runs[0].id, runs[1].id)
    assert cmp.common_cases == N
    assert evaluate_gates(kit, runs[1].id, GATE, cmp).status == "fail"


def test_pairing_ignores_output_dependent_fields_but_not_the_input_side(kit):
    """Output, metadata, tags and retrieved documents may differ between systems; the prompt, the
    context and the reference (what the case *asks* and what counts as right) may not."""
    base = [good(i) | {"metadata": {"run": "a"}, "tags": ["x"]} for i in range(N)]
    cand = [worse(i) | {"metadata": {"run": "b"}, "tags": ["y"]} for i in range(N)]
    cand[0] = cand[0] | {"reference": "a different question's answer"}  # input side changed
    cand[1] = cand[1] | {"prompt": "a different question"}
    kit.datasets.import_cases("qa", base)
    kit.datasets.import_cases("qa", cand)
    baseline, candidate = run_precomputed(kit, "qa@1"), run_precomputed(kit, "qa@2")

    cmp = compare(kit, baseline.id, candidate.id)

    assert cmp.common_cases == N - 2  # only the two cases whose *inputs* changed are unpaired
    assert cmp.unpaired["input_changed"] == 2
    assert any("input" in w and "2" in w for w in cmp.warnings)
    assert not cmp.same_dataset_version
