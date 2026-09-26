"""`--retry-failed`: eligible failures get another try, nothing that succeeded is repeated, the
history is kept, and it survives crashes and resumes."""

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from collections import Counter

import pytest
from conftest import judged

from evalkit import EvalFailure, EvaluatorSpec, FailureClass, Rubric, RunError
from evalkit.calls import CancelToken
from evalkit.evaluators import llm_judge_spec
from evalkit.failures import retry_eligible
from evalkit.llm import LLMResponse, Usage
from evalkit.targets import CallableTarget, ModelTarget, PrecomputedTarget

N = 12
RX = EvaluatorSpec(kind="regex", name="any", params={"pattern": "."})
POLICY = {
    "concurrency": 4,
    "retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0, "max_attempts": 1},
}
RUBRIC = Rubric.from_dict({"quality": "Good?"})
UNAVAILABLE = EvalFailure(FailureClass.INFRA, "provider_unavailable", "503")


def dataset(kit, n=N):
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i:02}", "prompt": f"q{i}", "output": f"o{i}"} for i in range(n)]
    )


class Flaky:
    """A fake model for the *target* stage that fails the first `fail_times[key]` calls for a case
    (keyed by its case key), then answers `out-<key>`."""

    provider, model = "fake", "gen-1"

    def __init__(self, fail_times=None, error=UNAVAILABLE):
        self.fail_times, self.error = fail_times or {}, error
        self.calls = Counter()
        self.lock = threading.Lock()

    def call(self, req):
        key = f"c{int(req.user[1:]):02}"
        with self.lock:
            self.calls[key] += 1
            n = self.calls[key]
        if n <= self.fail_times.get(key, 0):
            raise self.error
        return LLMResponse(text="out-" + key, stop_reason="end_turn", usage=Usage(3, 1))

    def target(self):
        return ModelTarget(self, "{prompt}")


class Judge:
    provider, model = "fake", "judge-1"

    def __init__(self, fail=None, error=UNAVAILABLE, payload=None):
        self.fail, self.error, self.payload = fail or {}, error, payload
        self.calls = Counter()
        self.lock = threading.Lock()

    def call(self, req):
        key = req.user.split("out-")[-1].split("\n")[0].strip() if "out-" in req.user else "?"
        with self.lock:
            self.calls[key] += 1
            n = self.calls[key]
        if n <= self.fail.get(key, 0):
            raise self.error
        return LLMResponse(
            payload=self.payload or judged(quality=5), stop_reason="tool_use", usage=Usage(10, 2)
        )


def create(kit, target=None, evaluators=(RX,), policy=None):
    return kit.controller.create(
        "qa",
        target=target or PrecomputedTarget(),
        evaluators=list(evaluators),
        policy={**POLICY, **(policy or {})},
    )


# ---- eligibility --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kind,retryable,expected",
    [
        (FailureClass.INFRA, "rate_limited", True, True),
        (FailureClass.INFRA, "provider_unavailable", True, True),
        (FailureClass.INFRA, "timeout", True, True),
        (FailureClass.INFRA, "connection", True, True),
        (FailureClass.TARGET, "timeout", True, True),
        (FailureClass.INFRA, "deadline_exceeded", False, True),  # cut short by the clock
        (FailureClass.INFRA, "budget_exceeded", False, True),  # cut short by the budget
        (FailureClass.EVALUATOR, "invalid_output", True, False),  # a repeat reproduces it
        (FailureClass.EVALUATOR, "truncated", False, False),
        (FailureClass.EVALUATOR, "refused", False, False),
        (FailureClass.EVALUATOR, "internal_error", False, False),
        (FailureClass.TARGET, "exception", False, False),
        (FailureClass.TARGET, "contract_violation", False, False),
        (FailureClass.INPUT, "oversize", False, False),
        (FailureClass.INPUT, "bad_config", True, False),  # systemic: fixed, not retried
        (FailureClass.INFRA, "auth", True, False),
        (FailureClass.INFRA, "quota_exhausted", True, False),
        (FailureClass.INFRA, "internal_error", False, False),
    ],
)
def test_which_failures_are_retried(cls, kind, retryable, expected):
    assert retry_eligible(cls, kind, retryable) is expected


def test_the_sql_predicate_agrees_with_the_python_rule(kit):
    """`--retry-failed` selects rows in SQL; that must be exactly `retry_eligible`."""
    from evalkit.failures import KINDS
    from evalkit.run_store import _retry_predicate

    sql, params = _retry_predicate("t", 3)
    conn = kit.store._conn
    for cls, kinds in KINDS.items():
        for kind in kinds:
            for retryable in (0, 1):
                got = conn.execute(
                    f"SELECT {sql} FROM (SELECT '{cls.value}' AS failure_class, "
                    f"'{kind}' AS failure_kind, {retryable} AS retryable, 0 AS retry_round) t",
                    params,
                ).fetchone()[0]
                assert bool(got) is retry_eligible(cls, kind, bool(retryable)), (cls, kind)
    exhausted = conn.execute(
        f"SELECT {sql} FROM (SELECT 'infrastructure' AS failure_class, "
        "'rate_limited' AS failure_kind, 1 AS retryable, 3 AS retry_round) t",
        params,
    ).fetchone()[0]
    assert not exhausted  # the round cap applies


# ---- target failures ----------------------------------------------------------------------------


def test_failed_targets_are_retried_and_only_they_are_called_again(kit):
    dataset(kit)
    flaky = Flaky({"c03": 1, "c07": 1, "c09": 1})
    run = create(kit, flaky.target())
    first = kit.controller.execute(run.id, target=flaky.target())
    assert first.status == "succeeded" and kit.runs.counts(run.id).failed == 3
    assert sum(flaky.calls.values()) == N

    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert report.status == "succeeded" and report.retry["reopened_cases"] == 3
    assert kit.runs.counts(run.id).failed == 0 and kit.runs.counts(run.id).complete == N
    for key, n in flaky.calls.items():  # nothing that worked was called again
        assert n == (2 if key in ("c03", "c07", "c09") else 1), key
    assert kit.runs.verify(run.id).ok


def test_a_retried_case_keeps_every_attempt_and_the_failure_history(kit):
    dataset(kit, 4)
    flaky = Flaky({"c01": 1})
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    before = kit.runs.case_result(run.id, "c01")
    assert before.status == "failed" and [a.outcome for a in before.attempts] == ["failed"]
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    after = kit.runs.case_result(run.id, "c01")
    assert after.status == "complete" and after.id == before.id
    assert [(a.n, a.outcome, a.retry_round) for a in after.attempts] == [
        (1, "failed", 0),
        (2, "ok", 1),
    ]
    assert after.attempts[0].error_kind == "provider_unavailable"
    (h,) = kit.store.failure_history(run.id)
    assert (h["scope"], h["case_key"], h["retry_round"]) == ("case", "c01", 0)
    assert (h["failure_class"], h["failure_kind"]) == ("infrastructure", "provider_unavailable")
    assert kit.store.retry_stats(run.id)["units_retried"] == 1


def test_a_target_failure_that_recurs_stays_failed_with_both_failures_kept(kit):
    dataset(kit, 3)
    flaky = Flaky({"c00": 2})
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)  # fails again
    cr = kit.runs.case_result(run.id, "c00")
    assert cr.status == "failed" and [a.n for a in cr.attempts] == [1, 2]
    assert len(kit.store.failure_history(run.id)) == 1
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)  # now it works
    assert kit.runs.case_result(run.id, "c00").status == "complete"
    assert len(kit.store.failure_history(run.id)) == 2 and kit.runs.verify(run.id).ok


def test_non_retryable_target_failures_are_left_alone(kit):
    dataset(kit, 4)
    bad = EvalFailure(FailureClass.TARGET, "contract_violation", "bad shape")
    flaky = Flaky({"c02": 99}, error=bad)
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    calls = dict(flaky.calls)
    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert report.retry["eligible_cases"] == report.retry["reopened_cases"] == 0
    assert dict(flaky.calls) == calls and kit.runs.get(run.id).status == "succeeded"
    assert kit.runs.counts(run.id).failed == 1


# ---- evaluator failures -------------------------------------------------------------------------


def judged_run(kit, judge, target=None, **kw):
    spec = llm_judge_spec("quality", RUBRIC, judge)
    return create(kit, target, evaluators=[spec], **kw)


def test_only_failed_judge_calls_are_repeated_and_successes_never_are(kit):
    dataset(kit)
    flaky = Flaky()
    judge = Judge({"c02": 1, "c05": 1})
    run = judged_run(kit, judge, flaky.target())
    first = kit.controller.execute(run.id, target=flaky.target(), clients=[judge])
    assert first.status == "succeeded"
    counts = kit.runs.counts(run.id).evaluator_results
    (states,) = counts.values()
    assert states == {"ok": N - 2, "failed": 2}
    target_calls = sum(flaky.calls.values())

    report = kit.controller.execute(
        run.id, target=flaky.target(), clients=[judge], retry_failed=True
    )
    assert report.status == "succeeded" and report.retry["reopened_evaluators"] == 2
    (states,) = kit.runs.counts(run.id).evaluator_results.values()
    assert states == {"ok": N}
    assert sum(flaky.calls.values()) == target_calls  # the target was not called again
    for key, n in judge.calls.items():
        assert n == (2 if key in ("c02", "c05") else 1), key  # one call per success, always
    assert kit.runs.verify(run.id).ok


def test_a_retried_evaluator_keeps_its_row_attempts_and_history(kit):
    dataset(kit, 3)
    judge = Judge({"c01": 1})
    run = judged_run(kit, judge, Flaky().target())
    kit.controller.execute(run.id, target=Flaky().target(), clients=[judge])
    cr = kit.runs.case_result(run.id, "c01")
    (before,) = kit.runs.evaluator_results(cr.id)
    assert before.status == "failed" and before.retry_round == 0
    kit.controller.execute(run.id, target=Flaky().target(), clients=[judge], retry_failed=True)
    (after,) = kit.runs.evaluator_results(cr.id)
    assert after.id == before.id and after.status == "ok" and after.score == 1.0
    assert [(a.n, a.outcome, a.retry_round) for a in after.attempts] == [
        (1, "failed", 0),
        (2, "ok", 1),
    ]
    (h,) = kit.store.failure_history(run.id)
    assert (h["scope"], h["evaluator_key"]) == ("evaluator", after.evaluator_key)
    assert kit.summarize(run.id).evaluators[after.evaluator_key].scored == 3


def test_an_invalid_judge_output_is_never_retried(kit):
    dataset(kit, 4)
    judge = Judge(payload={"bogus": 1})  # never valid: evaluator.invalid_output after its retry
    run = judged_run(kit, judge, Flaky().target())
    kit.controller.execute(run.id, target=Flaky().target(), clients=[judge])
    calls = sum(judge.calls.values())
    report = kit.controller.execute(
        run.id, target=Flaky().target(), clients=[judge], retry_failed=True
    )
    assert report.retry["eligible_evaluators"] == 0 and sum(judge.calls.values()) == calls
    assert kit.store.failure_history(run.id) == []


def test_a_systemic_failure_is_not_retried(kit):
    dataset(kit, 4)
    auth = EvalFailure(FailureClass.INFRA, "auth", "denied")
    judge = Judge({f"c{i:02}": 99 for i in range(4)}, error=auth)
    run = judged_run(kit, judge, Flaky().target(), policy={"systemic_threshold": 99})
    kit.controller.execute(run.id, target=Flaky().target(), clients=[judge])
    calls = sum(judge.calls.values())
    report = kit.controller.execute(
        run.id, target=Flaky().target(), clients=[judge], retry_failed=True
    )
    assert report.retry["eligible_evaluators"] == 0 and sum(judge.calls.values()) == calls


def test_a_unit_is_retried_at_most_retry_failed_rounds_times(kit):
    dataset(kit, 2)
    judge = Judge({"c00": 99})
    run = judged_run(kit, judge, Flaky().target(), policy={"retry_failed_rounds": 2})
    kit.controller.execute(run.id, target=Flaky().target(), clients=[judge])
    for expected in (1, 1, 0):
        report = kit.controller.execute(
            run.id, target=Flaky().target(), clients=[judge], retry_failed=True
        )
        assert report.retry["reopened_evaluators"] == expected
    assert judge.calls["c00"] == 3  # the first try and two retries, then it is left alone
    assert len(kit.store.failure_history(run.id)) == 2


# ---- lifecycle ----------------------------------------------------------------------------------


def test_a_succeeded_run_is_reopened_only_when_something_is_retryable(kit):
    dataset(kit, 3)
    flaky = Flaky()
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    with pytest.raises(RunError, match="already succeeded"):
        kit.controller.execute(run.id, target=flaky.target())  # unchanged P1 behaviour
    before = kit.runs.get(run.id)
    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert report.units == 0 and kit.runs.get(run.id) == before
    assert kit.store.list_executions(run.id)[-1]["retry_failed"] == 0  # no execution recorded


def test_a_reopened_run_finishes_succeeded_and_records_the_execution(kit):
    dataset(kit, 4)
    flaky = Flaky({"c01": 1})
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    assert kit.runs.get(run.id).status == "succeeded"
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    final = kit.runs.get(run.id)
    assert final.status == "succeeded" and final.finished_at is not None
    first, second = kit.store.list_executions(run.id)
    assert (first["retry_failed"], second["retry_failed"]) == (0, 1)
    assert second["status_before"] == "succeeded" and second["reopened_cases"] == 1
    assert second["outcome_status"] == "succeeded"


def test_retry_on_a_partial_run_finishes_it_and_retries_in_one_execute(kit):
    dataset(kit, 10)
    flaky = Flaky({"c00": 1, "c01": 1, "c02": 1})
    token = CancelToken()
    run = create(kit, flaky.target(), policy={"concurrency": 1})
    seen = []

    def stop_after(n):
        seen.append(n)
        if n >= 4:
            token.cancel("test")

    first = kit.controller.execute(
        run.id, target=flaky.target(), token=token, on_progress=stop_after
    )
    assert first.status == "cancelled" and kit.runs.counts(run.id).pending > 0
    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    # the cases the first execution never reached fail for the first time *here*; a unit is never
    # tried twice in one execution, so those wait for the next retry
    assert report.status == "succeeded" and kit.runs.counts(run.id).pending == 0
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    assert kit.runs.counts(run.id).failed == 0
    assert all(n <= 2 for n in flaky.calls.values()) and flaky.calls["c00"] == 2


def test_a_reopened_run_that_is_interrupted_can_be_resumed_without_repeating_work(kit):
    dataset(kit, 8)
    flaky = Flaky({f"c{i:02}": 1 for i in range(8)})  # every case fails once
    run = create(kit, flaky.target(), policy={"concurrency": 1})
    kit.controller.execute(run.id, target=flaky.target())
    assert kit.runs.counts(run.id).failed == 8
    token = CancelToken()
    token.cancel("stop before any unit")  # the crash: reopened, nothing executed
    report = kit.controller.execute(run.id, target=flaky.target(), retry_failed=True, token=token)
    assert report.status == "cancelled" and report.retry["reopened_cases"] == 8
    assert kit.runs.counts(run.id).pending == 8 and sum(flaky.calls.values()) == 8
    # a plain resume (no flag) finishes the reopened cases: each is called exactly once more
    done = kit.controller.execute(run.id, target=flaky.target())
    assert done.status == "succeeded" and set(flaky.calls.values()) == {2}
    assert kit.runs.verify(run.id).ok


def test_a_crash_mid_retry_never_repeats_a_success(kit):
    dataset(kit, 10)
    judge = Judge({f"c{i:02}": 1 for i in range(10)})
    flaky = Flaky()
    run = judged_run(kit, judge, flaky.target(), policy={"concurrency": 1})
    kit.controller.execute(run.id, target=flaky.target(), clients=[judge])
    assert next(iter(kit.runs.counts(run.id).evaluator_results.values())) == {"failed": 10}

    class Interrupt(Judge):
        def call(self, req):
            with self.lock:
                if sum(self.calls.values()) >= 14:  # 10 first tries + 4 successful retries
                    raise KeyboardInterrupt
            return super().call(req)

    crashing = Interrupt({f"c{i:02}": 1 for i in range(10)})
    crashing.calls = judge.calls
    crashed = kit.controller.execute(
        run.id, target=flaky.target(), clients=[crashing], retry_failed=True
    )
    assert crashed.worker_errors > 0 and crashed.status == "partial"
    (states,) = kit.runs.counts(run.id).evaluator_results.values()
    assert states["ok"] >= 1 and states["ok"] + states.get("failed", 0) == 10
    ok_before = states["ok"]
    calls_before = sum(judge.calls.values())
    kit.controller.execute(run.id, target=flaky.target(), clients=[judge], retry_failed=True)
    (states,) = kit.runs.counts(run.id).evaluator_results.values()
    assert states == {"ok": 10} and kit.runs.verify(run.id).ok
    # exactly the unfinished ones were called again: no success was repeated
    assert sum(judge.calls.values()) - calls_before <= 10 - ok_before + 1


@pytest.mark.skipif(os.name != "posix", reason="needs SIGKILL")
def test_a_killed_retry_resumes_exactly_once(tmp_path):
    db = tmp_path / "retry.db"
    from evalkit import EvalKit

    kit = EvalKit.open(db)
    n = 120
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i:03}", "prompt": f"q{i}", "output": "o"} for i in range(n)]
    )
    state = tmp_path / "healthy"

    def flaky_then_ok(inp):
        if not state.exists():
            time.sleep(0.5)  # longer than the request timeout: target.timeout, which is retryable
        return "a"

    target = CallableTarget(flaky_then_ok, name="flaky", fingerprint="1")
    policy = {**POLICY, "concurrency": 8, "retry": {**POLICY["retry"], "timeout_s": 0.3}}
    run = kit.controller.create("qa", target=target, evaluators=[RX], policy=policy)
    kit.controller.execute(run.id, target=target)
    assert kit.runs.counts(run.id).failed == n
    kit.close()
    state.write_text("up")

    script = tmp_path / "worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, time
            from evalkit import EvalKit
            from evalkit.targets import CallableTarget

            def slow(inp):
                time.sleep(0.02)
                return "a"

            kit = EvalKit.open(sys.argv[1])
            target = CallableTarget(slow, name="flaky", fingerprint="1")
            kit.controller.execute(sys.argv[2], target=target, retry_failed=True)
            """
        )
    )
    proc = subprocess.Popen([sys.executable, str(script), str(db), run.id])
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(0.05)
            try:
                conn = sqlite3.connect(db, timeout=5)
                (done,) = conn.execute(
                    "SELECT COUNT(*) FROM case_results WHERE run_id = ? AND status = 'complete'",
                    (run.id,),
                ).fetchone()
                conn.close()
            except sqlite3.OperationalError:
                continue
            if done >= 30:
                break
        else:
            pytest.fail("the retry never made progress")
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    kit = EvalKit.open(db)
    assert kit.runs.get(run.id).status == "running"  # not falsely finished
    target = CallableTarget(lambda i: "a", name="flaky", fingerprint="1")
    report = kit.controller.execute(run.id, target=target, retry_failed=True)
    assert report.status == "succeeded"
    conn = kit.store._conn
    assert (
        conn.execute("SELECT COUNT(*) FROM case_results WHERE status = 'complete'").fetchone()[0]
        == n
    )
    # every case has exactly one failed attempt (round 0) and exactly one successful one
    rows = conn.execute(
        "SELECT case_result_id, SUM(outcome = 'ok'), SUM(outcome = 'failed'), MAX(n) "
        "FROM attempts WHERE case_result_id IS NOT NULL GROUP BY 1"
    ).fetchall()
    assert len(rows) == n and all(tuple(r[1:]) == (1, 1, 2) for r in rows)
    assert conn.execute("SELECT COUNT(*) FROM result_history").fetchone()[0] == n
    assert kit.runs.verify(run.id).ok
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_reuse_targets_do_not_retry_inherited_failures(kit):
    dataset(kit, 4)
    flaky = Flaky({"c01": 1})
    source = create(kit, flaky.target())
    kit.controller.execute(source.id, target=flaky.target())
    from evalkit.targets import ReuseTarget

    reuse = create(kit, ReuseTarget(source.id))
    kit.controller.execute(reuse.id)
    assert kit.runs.counts(reuse.id).failed == 1
    before = kit.runs.get(reuse.id)
    executions = len(kit.store.list_executions(reuse.id))
    report = kit.controller.execute(reuse.id, retry_failed=True)
    assert report.units == 0 and kit.runs.counts(reuse.id).failed == 1  # retry at the source
    assert report.retry["eligible_cases"] == 0 and report.retry["reopened_cases"] == 0
    assert kit.runs.get(reuse.id) == before  # not even reopened
    assert len(kit.store.list_executions(reuse.id)) == executions
    assert kit.store.failure_history(reuse.id) == []


def test_retrying_does_not_change_the_runs_identity_or_frozen_config(kit):
    dataset(kit, 3)
    flaky = Flaky({"c00": 1})
    run = create(kit, flaky.target())
    kit.controller.execute(run.id, target=flaky.target())
    before = kit.runs.get(run.id)
    kit.controller.execute(run.id, target=flaky.target(), retry_failed=True)
    after = kit.runs.get(run.id)
    assert (after.identity_hash, after.exec_hash, after.config, after.environment) == (
        before.identity_hash, before.exec_hash, before.config, before.environment,
    )  # fmt: skip


def test_a_partial_reuse_run_does_not_reopen_the_failures_it_inherited(kit):
    from evalkit.targets import ReuseTarget

    dataset(kit, 6)
    flaky = Flaky({f"c{i:02}": 1 for i in range(6)})  # every case fails in the source run
    source = create(kit, flaky.target())
    kit.controller.execute(source.id, target=flaky.target())
    reuse = create(kit, ReuseTarget(source.id), policy={"concurrency": 1})
    token = CancelToken()
    first = kit.controller.execute(
        reuse.id, token=token, on_progress=lambda n: token.cancel("stop") if n >= 2 else None
    )
    assert first.status == "cancelled" and kit.runs.counts(reuse.id).failed >= 1
    failed = kit.runs.counts(reuse.id).failed
    done = kit.controller.execute(reuse.id, retry_failed=True)  # finishes the rest; retries nothing
    assert done.retry["reopened_cases"] == 0
    assert kit.store.failure_history(reuse.id) == []
    assert kit.runs.counts(reuse.id).failed == 6 and failed <= 6
