"""Property tests: random sequences of operations against an independent model of the rules.

The model below is a second, deliberately naive statement of the run domain's invariants (design
sections 3, 4.2, 7.3). After every operation the real system must (a) accept or refuse exactly what
the model says, (b) hold the same state, and (c) pass `verify`.
"""

import math

from conftest import case
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from evalkit import (
    CaseOutcome,
    DuplicateResultError,
    EvalKit,
    EvaluatorOutcome,
    EvaluatorSpec,
    Failure,
    RunConfig,
    RunError,
)
from evalkit.failures import KINDS, FailureClass
from evalkit.runs import TRANSITIONS

KEYS = [f"k{i}" for i in range(4)]
SPECS = [EvaluatorSpec(kind="exact_match", name=f"e{i}") for i in range(2)]
EV_KEYS = [s.key for s in SPECS]
GHOST = "exact_match:ghost:000000000000"

CASE_CLASSES = [FailureClass.INPUT, FailureClass.TARGET, FailureClass.INFRA]
EVAL_CLASSES = [FailureClass.INPUT, FailureClass.EVALUATOR, FailureClass.INFRA]


def failure(classes):
    def build(cls, data):
        kinds = sorted(KINDS[cls])
        return Failure(failure_class=cls, kind=kinds[data % len(kinds)], message="m")

    return st.builds(build, st.sampled_from(classes), st.integers(0, 50))


case_outcomes = st.one_of(
    st.text(max_size=20).map(lambda t: CaseOutcome.complete(t.replace("\ud800", ""))),
    failure(CASE_CLASSES).map(CaseOutcome.fail),
)
finite = st.floats(allow_nan=False, allow_infinity=False, width=32)
eval_outcomes = st.one_of(
    st.tuples(st.sampled_from(EV_KEYS * 3 + [GHOST]), finite).map(
        lambda t: EvaluatorOutcome(evaluator_key=t[0], status="ok", metrics={"score": t[1]})
    ),
    st.sampled_from(EV_KEYS * 3 + [GHOST]).map(
        lambda k: EvaluatorOutcome(
            evaluator_key=k, status="not_applicable", detail={"reason": "none relevant"}
        )
    ),
    st.tuples(st.sampled_from(EV_KEYS * 3 + [GHOST]), failure(EVAL_CLASSES)).map(
        lambda t: EvaluatorOutcome(evaluator_key=t[0], status="failed", failure=t[1])
    ),
    st.sampled_from(EV_KEYS * 3 + [GHOST]).map(
        lambda k: EvaluatorOutcome(
            evaluator_key=k, status="skipped", detail={"reason": "target failed"}
        )
    ),
)


class RunMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.kit = EvalKit.open(":memory:")
        self.kit.datasets.import_cases("qa", [case(k, output="o") for k in KEYS])
        self.run = self.kit.runs.create("qa", RunConfig(evaluators=SPECS))
        self.status = "created"
        self.cases: dict[str, tuple[str, str]] = {}  # case_key -> (status, id)
        self.evals: dict[tuple[str, str], str] = {}  # (case_key, evaluator_key) -> status
        self.planned = False

    def teardown(self):
        self.kit.close()

    @initialize()
    def start(self):
        """Begin in `running`, where the interesting operations are; the machine may leave it."""
        self.kit.runs.start(self.run.id)
        self.status = "running"

    # -- operations, each compared with the model ----------------------------------------------

    @rule(
        target_status=st.sampled_from(
            ["running"] * 3 + ["succeeded"] * 2 + ["partial", "failed", "cancelled", "created"]
        )
    )
    def transition(self, target_status):
        stop = "why" if target_status in ("partial", "failed", "cancelled") else None
        done = sum(1 for s, _ in self.cases.values() if s != "pending")
        # success needs every case terminal AND every terminal case to have all the evaluators'
        # results (the P1.1 completeness rule, restated here independently of the implementation)
        complete_evals = len(self.evals) == done * len(EV_KEYS)
        legal = target_status in TRANSITIONS[self.status] and (
            target_status != "succeeded" or (done == len(KEYS) and complete_evals)
        )
        try:
            got = self.kit.runs.transition(self.run.id, target_status, stop_reason=stop)
        except RunError:
            assert not legal
        else:
            assert legal and got.status == target_status
            self.status = target_status

    @rule()
    def plan(self):
        expected = 0 if self.status not in ("created", "running") else len(KEYS) - len(self.cases)
        try:
            n = self.kit.runs.plan(self.run.id)
        except RunError:
            assert self.status not in ("created", "running")
            return
        assert n == expected
        rows = {r.case_key: r for r in self.kit.runs.case_results(self.run.id)}
        for key, r in rows.items():
            self.cases.setdefault(key, ("pending", r.id))

    @rule(key=st.sampled_from(KEYS), outcome=case_outcomes)
    def record_case(self, key, outcome):
        terminal = self.cases.get(key, ("none", ""))[0] in ("complete", "failed")
        try:
            got = self.kit.runs.record_case_result(self.run.id, key, outcome)
        except DuplicateResultError:
            assert terminal
        except RunError:
            assert self.status != "running" and not terminal
        else:
            assert self.status == "running" and not terminal
            self.cases[key] = (got.status, got.id)
            assert got.status == ("failed" if outcome.failure else "complete")

    @rule(data=st.data(), outcome=eval_outcomes)
    def record_evaluator(self, data, outcome):
        recorded = sorted(self.cases)
        key = data.draw(st.sampled_from(recorded or KEYS) if recorded else st.sampled_from(KEYS))
        status, cr_id = self.cases.get(key, ("none", None))
        fits = (status == "complete" and outcome.status != "skipped") or (
            status == "failed" and outcome.status == "skipped"
        )
        known = outcome.evaluator_key in EV_KEYS
        duplicate = (key, outcome.evaluator_key) in self.evals
        if cr_id is None:  # nothing to attach to
            return
        try:
            self.kit.runs.record_evaluator_result(cr_id, outcome)
        except DuplicateResultError:
            assert duplicate and self.status == "running" and known and fits
        except RunError:
            assert not (self.status == "running" and known and fits and not duplicate)
        else:
            assert self.status == "running" and known and fits and not duplicate
            self.evals[(key, outcome.evaluator_key)] = outcome.status

    # -- invariants ----------------------------------------------------------------------------

    @invariant()
    def the_system_agrees_with_the_model(self):
        rs = self.kit.runs
        assert rs.get(self.run.id).status == self.status
        counts = rs.counts(self.run.id)
        by = {
            s: sum(1 for x, _ in self.cases.values() if x == s)
            for s in ("pending", "complete", "failed")
        }
        assert (counts.pending, counts.complete, counts.failed) == (
            by["pending"], by["complete"], by["failed"],
        )  # fmt: skip
        assert counts.missing == len(KEYS) - len(self.cases)
        assert counts.pending + counts.complete + counts.failed + counts.missing == len(KEYS)

    @invariant()
    def target_failures_are_never_scored(self):
        for (key, _), status in self.evals.items():
            case_status = self.cases[key][0]
            assert (case_status == "failed") == (status == "skipped")

    @invariant()
    def stored_state_is_verifiably_coherent(self):
        report = self.kit.runs.verify(self.run.id)
        assert report.ok, report.problems
        for status, cr_id in self.cases.values():
            if status == "pending":
                assert self.kit.runs.evaluator_results(cr_id) == []
            for er in self.kit.runs.evaluator_results(cr_id):
                assert all(math.isfinite(v) for v in er.metrics.values())
                assert (er.status == "ok") == bool(er.metrics)
                assert (er.status == "failed") == (er.failure is not None)
                if er.failure:
                    assert er.failure.failure_class is not FailureClass.TARGET


RunMachine.TestCase.settings = settings(
    max_examples=120, stateful_step_count=40, deadline=None, print_blob=True
)
TestRunOperations = RunMachine.TestCase
