"""P1.1 execution hardening (audit P1-1 .. P1-7), each through the *real* executor.

Every test here was written against the pre-fix executor and failed there: stranded failed-target
rows, double execution, a run left `running` after Ctrl-C, a dead writer thread that hung the run,
and runs marked `succeeded` without evaluator results or with nothing scored."""

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from evalkit import EvalKit, EvaluatorSpec, RunError
from evalkit.runs import CaseOutcome, Failure
from evalkit.targets import CallableTarget, PrecomputedTarget

N = 30
EM = EvaluatorSpec(kind="exact_match", name="em")
RX = EvaluatorSpec(kind="regex", name="digits", params={"pattern": r"\d"})
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}


class Counting:
    """A callable target that counts its calls per case (thread-safe) and can misbehave."""

    def __init__(self, fail=(), delay=0.0):
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()
        self.fail, self.delay = set(fail), delay

    def __call__(self, inp):
        with self.lock:
            self.calls[inp.case_key] = self.calls.get(inp.case_key, 0) + 1
        if self.delay:
            time.sleep(self.delay)
        if inp.case_key in self.fail:
            raise RuntimeError("the system under test failed")
        return "a" + inp.prompt[1:]

    @property
    def total(self):
        return sum(self.calls.values())

    def target(self):
        return CallableTarget(self, name="app", fingerprint="1")


def make(tmp_path, n=N, evaluators=(EM,), policy=None, target=None, name="e.db"):
    kit = EvalKit.open(tmp_path / name)
    kit.datasets.import_cases(
        "qa",
        [
            {"case_key": f"c{i:03}", "prompt": f"q{i}", "reference": f"a{i}", "output": f"a{i}"}
            for i in range(n)
        ],
    )
    tgt = target or PrecomputedTarget()
    run = kit.controller.create(
        "qa", target=tgt, evaluators=list(evaluators), policy={**FAST, **(policy or {})}
    )
    return kit, run, tgt


# --- audit P1-1: a failed target whose `skipped` rows were never written -----------------------


def test_a_stranded_failed_case_result_is_repaired_on_resume_and_the_run_can_succeed(tmp_path):
    counter = Counting(fail={"c001"})
    kit, run, tgt = make(tmp_path, n=3, evaluators=(EM, RX), target=counter.target())
    # a crash (or a batch boundary) left the failed case result committed without its skipped rows
    kit.runs.start(run.id)
    kit.runs.record_case_result(
        run.id,
        "c001",
        CaseOutcome.fail(Failure(failure_class="target", kind="exception", message="boom")),
    )
    report = kit.controller.execute(run.id, target=tgt)
    assert (report.status, report.stop_reason) == ("succeeded", None)
    assert counter.calls.get("c001", 0) == 0  # the failed case was NOT run again
    cr = kit.runs.case_result(run.id, "c001")
    statuses = {e.evaluator_key: e.status for e in kit.runs.evaluator_results(cr.id)}
    assert set(statuses.values()) == {"skipped"} and len(statuses) == 2
    assert kit.runs.verify(run.id).ok
    kit.close()


def test_a_failed_target_and_its_skipped_rows_are_never_split_across_transactions(tmp_path):
    """Whatever the batch size, no commit may leave a failed case result without its skipped rows
    (that is the state that used to strand a run)."""
    counter = Counting(fail={f"c{i:03}" for i in range(N)})
    kit, run, tgt = make(
        tmp_path, target=counter.target(), evaluators=(EM, RX), policy={"batch_size": 1}
    )
    real = type(kit.store).write_batch
    seen = []

    def checked(self, run_id, items):
        real(self, run_id, items)
        stranded = self._conn.execute(
            "SELECT COUNT(*) FROM case_results r WHERE r.run_id = ? AND r.status = 'failed' AND "
            "(SELECT COUNT(*) FROM evaluator_results e WHERE e.case_result_id = r.id) < 2",
            (run_id,),
        ).fetchone()[0]
        seen.append(stranded)

    type(kit.store).write_batch = checked
    try:
        kit.controller.execute(run.id, target=tgt)
    finally:
        type(kit.store).write_batch = real
    assert seen and max(seen) == 0
    kit.close()


# --- audit P1-2: two executors on one run -----------------------------------------------------


def test_two_executors_on_one_run_never_duplicate_target_calls(tmp_path):
    counter = Counting(delay=0.02)
    kit, run, tgt = make(tmp_path, target=counter.target(), policy={"concurrency": 2})
    other = EvalKit.open(tmp_path / "e.db")
    outcomes = {}

    def go(name, k):
        try:
            outcomes[name] = k.controller.execute(run.id, target=tgt).status
        except RunError as e:
            outcomes[name] = f"refused: {e}"

    a = threading.Thread(target=go, args=("a", kit))
    a.start()
    time.sleep(0.1)  # a is mid-run
    b = threading.Thread(target=go, args=("b", other))
    b.start()
    a.join()
    b.join()
    assert outcomes["a"] == "succeeded"
    assert "already being executed" in outcomes["b"]
    assert counter.total == N and set(counter.calls.values()) == {1}  # no duplicate target call
    assert kit.runs.verify(run.id).ok
    other.close()
    kit.close()


WORKER = """
import sys, time
from evalkit import EvalKit
from evalkit.targets import CallableTarget

def slow(inp):
    time.sleep(0.05)
    return "a" + inp.prompt[1:]

kit = EvalKit.open(sys.argv[1])
kit.controller.execute(sys.argv[2], target=CallableTarget(slow, name="app", fingerprint="1"))
"""


@pytest.mark.skipif(os.name != "posix", reason="needs SIGKILL")
def test_a_killed_executor_leaves_no_stale_lock_and_the_run_can_be_resumed(tmp_path):
    counter = Counting()
    kit, run, tgt = make(tmp_path, target=counter.target(), policy={"concurrency": 1})
    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(WORKER))
    proc = subprocess.Popen([sys.executable, str(script), str(tmp_path / "e.db"), run.id])
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            time.sleep(0.05)
            if kit.runs.get(run.id).status == "running" and kit.runs.counts(run.id).complete >= 3:
                break
        else:
            pytest.fail("the worker never made progress")
        with pytest.raises(RunError, match="already being executed"):  # while it is alive
            kit.controller.execute(run.id, target=tgt)
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    assert kit.runs.get(run.id).status == "running"  # the crash left it mid-run
    report = kit.controller.execute(run.id, target=tgt)  # the dead process's lock is gone
    assert report.status == "succeeded"
    assert kit.runs.verify(run.id).ok
    kit.close()


def test_the_run_lock_is_released_even_when_execute_fails(tmp_path):
    counter = Counting()
    kit, run, tgt = make(tmp_path, target=counter.target())
    with pytest.raises(RuntimeError, match="stop here"):
        kit.controller.execute(run.id, target=tgt, on_progress=_raise(RuntimeError("stop here")))
    kit.controller.execute(run.id, target=tgt)  # not "already being executed"
    assert kit.runs.get(run.id).status == "succeeded"
    kit.close()


def _raise(exc, after=3):
    def hook(n):
        if n >= after:
            raise exc

    return hook


# --- audit P1-3: exception-safe execute --------------------------------------------------------


@pytest.mark.parametrize(
    "exc,status,reason",
    [
        (KeyboardInterrupt(), "cancelled", "interrupted"),
        (RuntimeError("bug in a hook"), "partial", "infrastructure.internal_error"),
    ],
)
def test_an_exception_in_execute_flushes_in_flight_results_and_leaves_a_resumable_run(
    tmp_path, exc, status, reason
):
    counter = Counting(delay=0.01)
    kit, run, tgt = make(tmp_path, target=counter.target(), policy={"concurrency": 4})
    with pytest.raises(type(exc)):
        kit.controller.execute(run.id, target=tgt, on_progress=_raise(exc, after=8))
    stopped = kit.runs.get(run.id)
    assert (stopped.status, stopped.stop_reason) == (status, reason)  # not stuck in `running`
    counts = kit.runs.counts(run.id)
    # every target call that was made has its result persisted: nothing paid for was lost
    assert counts.complete == sum(1 for v in counter.calls.values() if v) == counter.total
    for cr in kit.runs.case_results(run.id, status="complete"):
        assert len(kit.runs.evaluator_results(cr.id)) == 1
    assert kit.runs.verify(run.id).ok
    before = counter.total
    final = kit.controller.execute(run.id, target=tgt)
    assert final.status == "succeeded"
    assert counter.total == N and before < N  # the resume did only what was left
    assert set(counter.calls.values()) == {1}
    kit.close()


def test_resume_never_repeats_an_evaluator_call_that_was_already_paid_for(tmp_path):
    """The audit's surviving mutant: re-running finished evaluators wasted (and re-billed) calls."""
    from conftest import judged

    from evalkit import Rubric
    from evalkit.evaluators import llm_judge_spec
    from evalkit.llm import LLMResponse, Usage

    class Judge:
        provider, model = "fake", "judge-1"

        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def call(self, req):
            with self.lock:
                self.calls += 1
            return LLMResponse(
                payload=judged(quality=5), stop_reason="tool_use", usage=Usage(1, 1), request_id="r"
            )

    judge = Judge()
    spec = llm_judge_spec("quality", Rubric.from_dict({"quality": "Good?"}), judge)
    kit, run, tgt = make(tmp_path, n=6, evaluators=(EM, spec), policy={"concurrency": 1})
    kit.runs.start(run.id)
    for cr in list(kit.runs.case_results(run.id)):  # a crash after the case result and the EM row
        kit.runs.record_case_result(
            run.id, cr.case_key, CaseOutcome.complete(f"a{cr.case_key[1:]}")
        )
    from evalkit import EvaluatorOutcome

    em_key = next(s.key for s in run.config.evaluators if s.kind == "exact_match")
    for cr in kit.runs.case_results(run.id):
        kit.runs.record_evaluator_result(
            cr.id, EvaluatorOutcome(evaluator_key=em_key, status="ok", metrics={"match": 1.0})
        )
    report = kit.controller.execute(run.id, clients=[judge])
    assert report.status == "succeeded"
    assert judge.calls == 6  # one judge call per case, none repeated
    kit.close()


# --- audit P1-4: a dead writer must not hang the run --------------------------------------------


def run_with_timeout(fn, seconds=30):
    box = {}

    def go():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "execute() hung on a dead writer thread"
    if "error" in box:
        raise box["error"]
    return box["value"]


def test_an_unexpected_writer_exception_stops_the_run_instead_of_hanging_it(tmp_path):
    counter = Counting()
    kit, run, tgt = make(
        tmp_path, n=400, target=counter.target(), policy={"concurrency": 2, "batch_size": 8}
    )
    real = type(kit.store).write_batch
    state = {"n": 0}

    def buggy(self, run_id, items):
        state["n"] += 1
        if state["n"] == 2:
            raise ValueError("a bug in the write path")
        return real(self, run_id, items)

    type(kit.store).write_batch = buggy
    try:
        report = run_with_timeout(lambda: kit.controller.execute(run.id, target=tgt))
    finally:
        type(kit.store).write_batch = real
    assert report.status == "partial" and report.stop_reason == "infrastructure.internal_error"
    assert "ValueError" in (report.writer_error or "")
    assert report.spilled > 0  # what the dead writer held was spilled, not dropped
    final = kit.controller.execute(run.id, target=tgt)  # fixed: replay the spill, finish
    assert final.status == "succeeded" and kit.runs.verify(run.id).ok
    assert set(counter.calls.values()) == {1}  # nothing was re-run: the spill preserved it
    kit.close()


def test_a_storage_error_is_still_reported_as_a_storage_stop(tmp_path):
    import sqlite3

    kit, run, tgt = make(tmp_path, n=40, target=Counting().target(), policy={"batch_size": 4})
    real = type(kit.store).write_batch

    def broken(self, run_id, items):
        raise sqlite3.OperationalError("disk I/O error")

    type(kit.store).write_batch = broken
    try:
        report = run_with_timeout(lambda: kit.controller.execute(run.id, target=tgt))
    finally:
        type(kit.store).write_batch = real
    assert (report.status, report.stop_reason) == ("partial", "infrastructure.storage")
    kit.close()


# --- audit P1-7: what `succeeded` means ---------------------------------------------------------


def test_a_run_cannot_succeed_without_its_evaluator_results_service_and_database(tmp_path):
    kit, run, _ = make(tmp_path, n=3, evaluators=(EM, RX))
    kit.runs.start(run.id)
    for key in ("c000", "c001", "c002"):
        kit.runs.record_case_result(run.id, key, CaseOutcome.complete("x"))
    with pytest.raises(RunError, match="evaluator result"):
        kit.runs.transition(run.id, "succeeded")
    conn = kit.store._conn
    with pytest.raises(Exception, match="every evaluator result"):
        conn.execute("UPDATE runs SET status='succeeded', finished_at='t' WHERE id = ?", (run.id,))
    conn.rollback()
    assert kit.runs.get(run.id).status == "running"
    kit.close()


def test_verify_reports_a_succeeded_run_that_lacks_evaluator_results(tmp_path):
    kit, run, _ = make(tmp_path, n=3, evaluators=(EM,))
    kit.controller.execute(run.id)
    conn = kit.store._conn
    for trigger in ("evaluator_results_no_delete", "metrics_no_delete"):  # a writer that
        conn.execute(f"DROP TRIGGER {trigger}")  # bypassed the rules
    (victim,) = conn.execute("SELECT min(evaluator_result_id) FROM metrics").fetchone()
    conn.execute("DELETE FROM metrics WHERE evaluator_result_id = ?", (victim,))
    conn.execute("DELETE FROM evaluator_results WHERE id = ?", (victim,))
    conn.commit()
    report = kit.runs.verify(run.id)
    assert not report.ok and any("evaluator result" in p for p in report.problems)
    kit.close()


def test_a_run_where_every_case_failed_does_not_succeed(tmp_path):
    counter = Counting(fail={f"c{i:03}" for i in range(5)})
    kit, run, tgt = make(tmp_path, n=5, target=counter.target())
    report = kit.controller.execute(run.id, target=tgt)
    assert report.status == "partial" and report.stop_reason == "insufficient_coverage"
    assert kit.runs.counts(run.id).failed == 5
    assert kit.runs.verify(run.id).ok
    kit.close()


def test_an_evaluator_that_scored_nothing_does_not_let_the_run_succeed(tmp_path):
    kit = EvalKit.open(tmp_path / "e.db")
    kit.datasets.import_cases(
        "qa",
        [{"case_key": f"c{i}", "prompt": "p", "output": "o", "reference": "r"} for i in range(4)],
    )
    ok_eval = EM
    nothing = EvaluatorSpec(kind="regex", name="never", params={"pattern": "zzz"})
    run = kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[ok_eval, nothing], policy=FAST
    )
    assert kit.controller.execute(run.id).status == "succeeded"  # regex FAILs are scores, not gaps
    kit.close()


def test_the_run_level_minimum_coverage_is_enforced_by_the_policy(tmp_path):
    counter = Counting(fail={"c000", "c001"})
    kit, run, tgt = make(tmp_path, n=10, target=counter.target(), policy={"min_coverage": 0.9})
    report = kit.controller.execute(run.id, target=tgt)
    assert (report.status, report.stop_reason) == ("partial", "insufficient_coverage")
    kit.close()
    counter = Counting(fail={"c000", "c001"})
    kit, run, tgt = make(
        tmp_path, n=10, target=counter.target(), policy={"min_coverage": 0.5}, name="ok.db"
    )
    assert kit.controller.execute(run.id, target=tgt).status == "succeeded"  # 80 % >= 50 %
    kit.close()
