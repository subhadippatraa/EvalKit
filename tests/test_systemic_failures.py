"""P1.1 (audit P1-5): a systemic-looking provider error must not end a run on its first sighting.

The design (8.4) stops a run when the *same* systemic failure repeats with no success in between;
one rejected request among thousands is that case's problem, not the run's. And the cases that
were in flight when a run is stopped for a systemic cause are left pending (not recorded as
permanent failures), so resuming after the fix does exactly what is left."""

import threading
from datetime import UTC, datetime

import pytest
from conftest import judged

from evalkit import EvalFailure, EvalKit, Rubric, RunAttempt
from evalkit.calls import RunGuard
from evalkit.evaluators import llm_judge_spec
from evalkit.llm import LLMResponse, Usage
from evalkit.targets import PrecomputedTarget

N = 40
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}


def failed_attempt(cls, kind, provider="p", model="m"):
    f = EvalFailure(cls, kind, "boom")
    return RunAttempt.failed(1, datetime.now(UTC), 1, f, provider=provider, model=model)


def ok_attempt(provider="p", model="m"):
    return RunAttempt.succeeded(1, datetime.now(UTC), 1, provider=provider, model=model)


# --- the guard's rule ------------------------------------------------------------------------


def test_one_systemic_failure_does_not_stop_the_run():
    g = RunGuard(systemic_threshold=3)
    g.record(failed_attempt("evaluator", "bad_request"))
    assert g.abort is None


def test_repeated_systemic_failures_stop_the_run_as_failed():
    g = RunGuard(systemic_threshold=3)
    for _ in range(2):
        g.record(failed_attempt("infrastructure", "auth"))
    assert g.abort is None
    g.record(failed_attempt("infrastructure", "auth"))
    assert g.abort == ("failed", "infrastructure.auth")


def test_a_success_from_the_same_provider_and_model_resets_the_count():
    g = RunGuard(systemic_threshold=3)
    g.record(failed_attempt("infrastructure", "auth"))
    g.record(failed_attempt("infrastructure", "auth"))
    g.record(ok_attempt())
    g.record(failed_attempt("infrastructure", "auth"))
    g.record(failed_attempt("infrastructure", "auth"))
    assert g.abort is None


def test_another_providers_success_does_not_hide_a_broken_one():
    """A callable target that keeps working must not mask a judge model that rejects everything."""
    g = RunGuard(systemic_threshold=3)
    for _ in range(3):
        g.record(failed_attempt("evaluator", "bad_request", provider="bedrock", model="typo"))
        g.record(ok_attempt(provider="callable", model="app"))
    assert g.abort == ("failed", "evaluator.bad_request")


def test_different_systemic_kinds_do_not_add_up():
    g = RunGuard(systemic_threshold=3)
    g.record(failed_attempt("infrastructure", "auth"))
    g.record(failed_attempt("evaluator", "bad_request"))
    g.record(failed_attempt("infrastructure", "quota_exhausted"))
    assert g.abort is None


def test_non_systemic_failures_never_count():
    g = RunGuard(systemic_threshold=1)
    for _ in range(5):
        g.record(failed_attempt("evaluator", "invalid_output"))
        g.record(failed_attempt("target", "exception"))
    assert g.abort is None


# --- through the real executor --------------------------------------------------------------


class Judge:
    provider, model = "fake", "judge-1"

    def __init__(self, reject=lambda text: False, error=None):
        self.reject, self.error = reject, error
        self.calls = 0
        self.lock = threading.Lock()

    def call(self, req):
        with self.lock:
            self.calls += 1
        if self.reject(req.user):
            raise self.error or EvalFailure(
                "evaluator", "bad_request", "400 rejected", provider="fake"
            )
        return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=Usage(1, 1))


def setup(tmp_path, n=N, bad=()):
    kit = EvalKit.open(tmp_path / "s.db")
    kit.datasets.import_cases(
        "qa",
        [
            {
                "case_key": f"c{i:03}",
                "prompt": f"q{i}",
                "output": "BADINPUT" if i in bad else "fine",
            }
            for i in range(n)
        ],
    )
    return kit


def judge_run(kit, judge, policy=None):
    spec = llm_judge_spec("quality", Rubric.from_dict({"quality": "Good?"}), judge)
    return kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[spec], policy={**FAST, **(policy or {})}
    )


def test_one_rejected_case_among_many_is_that_cases_failure_not_the_runs(tmp_path):
    kit = setup(tmp_path, bad={17})
    judge = Judge(reject=lambda text: "BADINPUT" in text)
    run = judge_run(kit, judge, {"concurrency": 1})
    report = kit.controller.execute(run.id, clients=[judge])
    assert report.status == "succeeded"
    counts = kit.runs.counts(run.id)
    (states,) = counts.evaluator_results.values()
    assert states == {"ok": N - 1, "failed": 1}
    (failure,) = kit.runs.failures(run.id)
    assert failure.case_key == "c017" and failure.failure.kind == "bad_request"
    kit.close()


def test_a_judge_that_rejects_everything_stops_the_run_after_a_few_calls(tmp_path):
    kit = setup(tmp_path)
    judge = Judge(reject=lambda text: True)
    run = judge_run(kit, judge, {"concurrency": 2})
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("failed", "evaluator.bad_request")
    assert judge.calls <= 3 + 4  # the threshold plus what was in flight: nowhere near N
    kit.close()


def test_cases_caught_in_a_systemic_stop_stay_pending_and_a_resume_completes_them(tmp_path):
    """The failure that stopped the run is the environment's, not the cases': recording it as a
    permanent (write-once) failure would make the fix useless."""
    kit = setup(tmp_path)
    broken = Judge(reject=lambda text: True)
    run = judge_run(kit, broken, {"concurrency": 2})
    kit.controller.execute(run.id, clients=[broken])
    assert kit.runs.failures(run.id) == []  # nothing systemic was recorded as a result
    fixed = Judge()
    report = kit.controller.execute(run.id, clients=[fixed])
    assert report.status == "succeeded"
    (states,) = kit.runs.counts(run.id).evaluator_results.values()
    assert states == {"ok": N}  # every case scored once the judge worked
    assert kit.runs.verify(run.id).ok
    kit.close()


def test_credentials_that_expire_mid_run_stop_it_but_keep_what_was_scored(tmp_path):
    kit = setup(tmp_path)
    state = {"n": 0}
    lock = threading.Lock()

    def expired(text):
        with lock:
            state["n"] += 1
            return state["n"] > 10

    judge = Judge(reject=expired, error=EvalFailure("infrastructure", "auth", "ExpiredToken"))
    run = judge_run(kit, judge, {"concurrency": 1})
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("failed", "infrastructure.auth")
    counts = kit.runs.counts(run.id)
    assert counts.complete == 10 + 0 or counts.complete >= 10  # the scored ones are kept
    (states,) = counts.evaluator_results.values()
    assert states.get("ok", 0) == 10 and "failed" not in states
    kit.close()


def test_a_systemic_failure_that_ends_with_the_dataset_is_still_recorded(tmp_path):
    """Two rejected cases and nothing after them: the run ends normally, so they are what they look
    like -- those cases' failures -- and are recorded (never silently dropped)."""
    kit = setup(tmp_path, n=6, bad={4, 5})
    judge = Judge(reject=lambda text: "BADINPUT" in text)
    run = judge_run(kit, judge, {"concurrency": 1})
    report = kit.controller.execute(run.id, clients=[judge])
    assert report.status == "succeeded"
    assert sorted(f.case_key for f in kit.runs.failures(run.id)) == ["c004", "c005"]
    kit.close()


def test_the_threshold_is_an_execution_policy_and_is_validated():
    from evalkit.engine import ExecPolicy

    assert ExecPolicy().systemic_threshold == 3
    assert ExecPolicy.from_mapping({"systemic_threshold": 1}).systemic_threshold == 1
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            ExecPolicy.from_mapping({"systemic_threshold": bad})
