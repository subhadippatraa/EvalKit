"""More seams of the P1.1 executor and analysis, exercised directly: the writer's fallback paths,
a crashing worker, a failing snapshot, an interrupted run whose stop cannot be recorded, and the
report sections for unpaired and not-applicable cases."""

import sqlite3
import threading

import pytest

from evalkit import DuplicateResultError, EvalKit, EvaluatorSpec, RunError
from evalkit.compare import Gates, MetricGate, compare, evaluate_gates
from evalkit.engine import ResultWriter, _Exec
from evalkit.report import render_report
from evalkit.runs import CaseOutcome, WriteItem
from evalkit.targets import CallableTarget, PrecomputedTarget, TargetOutput

EM = EvaluatorSpec(kind="exact_match", name="em")
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}


def dataset(kit, n=8, name="qa"):
    kit.datasets.import_cases(
        name,
        [
            {"case_key": f"c{i:03}", "prompt": f"q{i}", "reference": f"a{i}", "output": f"a{i}"}
            for i in range(n)
        ],
    )


@pytest.fixture
def kit(tmp_path):
    k = EvalKit.open(tmp_path / "s.db")
    yield k
    k.close()


def item(key="c000"):
    return WriteItem("case", key, CaseOutcome.complete("x"))


class Store:
    """A store double whose write_batch is scripted; spills go to a directory (or nowhere)."""

    def __init__(self, tmp_path=None, script=()):
        self.spill_dir = tmp_path
        self.script, self.calls = list(script), []

    def write_batch(self, run_id, items):
        self.calls.append([i.case_key for i in items])
        action = self.script.pop(0) if self.script else None
        if isinstance(action, BaseException):
            raise action


def writer(store, **kw):
    return ResultWriter(
        store, "run-1", batch_size=kw.pop("batch_size", 10), batch_ms=5, depth=4, **kw
    )


# --- the writer's fallback paths ---------


def test_one_bad_group_does_not_lose_the_others_in_its_batch(tmp_path):
    store = Store(tmp_path / "s", script=[RunError("bad group"), None, RunError("bad again"), None])
    w = writer(store)
    for key in ("a", "b", "c"):
        w.put(item(key))
    w.close()
    assert w.failed is not None  # the second group failed on its own retry: stop and spill the rest
    assert w.written + w.spilled >= 2 and store.calls[0] == ["a", "b", "c"]


def test_a_result_that_is_already_stored_is_counted_not_an_error(tmp_path):
    store = Store(
        tmp_path / "s", script=[RunError("batch failed"), DuplicateResultError("dup"), None]
    )
    w = writer(store)
    w.put(item("a"))
    w.put(item("b"))
    w.close()
    assert w.duplicates == 1 and w.written == 1 and w.failed is None


def test_a_spill_that_also_fails_is_reported_as_lost_and_never_raises(tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")  # mkdir under a file fails with an OSError
    store = Store(blocked / "spill", script=[sqlite3.OperationalError("disk I/O error")] * 10)
    w = writer(store, retries=0)
    w.put(item("a"))
    w.close()
    assert w.failed is not None and w.lost == 1 and w.spilled == 0


def test_a_result_offered_after_the_writer_closed_is_spilled(tmp_path):
    store = Store(tmp_path / "s")
    w = writer(store)
    w.close()
    assert w.put(item("late")) is False
    assert (tmp_path / "s" / "run-1.jsonl").is_file() and w.spilled == 1


def test_a_full_queue_at_close_does_not_block_the_shutdown(tmp_path):
    store = Store(tmp_path / "s")
    w = writer(store, batch_size=1)
    for i in range(6):
        w.put(item(f"c{i}"))
    w.close()
    assert w.written == 6


# --- executor seams ---------


def test_a_crashing_worker_stops_the_run_as_an_internal_error(kit, monkeypatch):
    dataset(kit)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)
    real = _Exec._process

    def boom(self, unit):
        if unit.case_key == "c003":
            raise RuntimeError("bug in a worker")
        return real(self, unit)

    monkeypatch.setattr(_Exec, "_process", boom)
    report = kit.controller.execute(run.id, overrides={"concurrency": 1})
    assert report.worker_errors == 1
    assert (report.status, report.stop_reason) == ("partial", "infrastructure.internal_error")


def test_a_failing_summary_snapshot_never_loses_the_execution_report(kit, monkeypatch):
    dataset(kit)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)

    def broken(run_id):
        raise sqlite3.OperationalError("snapshot failed")

    monkeypatch.setattr(kit, "snapshot", broken)
    assert kit.controller.execute(run.id).status == "succeeded"


def test_an_interrupted_run_whose_stop_cannot_be_recorded_still_raises_the_interrupt(
    kit, monkeypatch
):
    dataset(kit)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)

    real = kit.runs.transition

    def cannot(run_id, status, **k):
        if status == "running":  # starting works; recording the stop does not
            return real(run_id, status, **k)
        raise sqlite3.OperationalError("database is gone")

    monkeypatch.setattr(kit.runs, "transition", cannot)

    def interrupt(n):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        kit.controller.execute(run.id, on_progress=interrupt)


def test_a_run_created_without_evaluators_never_succeeds(kit):
    dataset(kit)
    run = kit.runs.create("qa", {"policy": FAST})  # (the controller refuses this at preflight)
    kit.runs.plan(run.id)
    report = kit.controller.execute(run.id)
    assert (report.status, report.stop_reason) == ("partial", "incomplete")


def test_the_controller_refuses_a_run_with_no_evaluators_at_preflight(kit):
    from evalkit.engine import PreflightError

    dataset(kit)
    with pytest.raises(PreflightError, match="at least one evaluator"):
        kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[], policy=FAST)


def test_execution_settings_given_at_execute_time_are_validated(kit):
    dataset(kit)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)
    with pytest.raises(RunError, match="invalid execution policy"):
        kit.controller.execute(run.id, overrides={"concurrency": 0})
    with pytest.raises(ValueError, match="min_coverage"):
        from evalkit.engine import ExecPolicy

        ExecPolicy(min_coverage=1.5)


def test_a_rejected_case_at_the_very_end_of_a_run_is_recorded_not_dropped(tmp_path):
    """Nothing after it proves the failure was the case's own, but the run did not stop for it
    either, so it is what it looks like: that case's failure."""
    from conftest import judged

    from evalkit import EvalFailure, Rubric
    from evalkit.evaluators import llm_judge_spec
    from evalkit.llm import LLMResponse, Usage
    from evalkit.store import _seeded_order

    kit = EvalKit.open(tmp_path / "e.db")
    keys = sorted((f"c{i:03}" for i in range(10)), key=lambda k: _seeded_order(0, k))
    last = keys[-1]
    kit.datasets.import_cases(
        "qa",
        [{"case_key": k, "prompt": "q", "output": "BAD" if k == last else "fine"} for k in keys],
    )

    class Judge:
        provider, model = "fake", "j"

        def call(self, req):
            if "BAD" in req.user:
                raise EvalFailure("evaluator", "bad_request", "400", provider="fake")
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=Usage(1, 1))

    spec = llm_judge_spec("quality", Rubric.from_dict({"quality": "ok?"}), Judge())
    run = kit.controller.create(
        "qa", target=PrecomputedTarget(), evaluators=[spec], policy={**FAST, "concurrency": 1}
    )
    report = kit.controller.execute(run.id, clients=[Judge()])
    assert report.status == "succeeded"
    assert [f.case_key for f in kit.runs.failures(run.id)] == [last]
    kit.close()


# --- report and gate sections ---------


def cites(inp):
    return TargetOutput("answer [1]", retrieved=["d1"])


def uncited(inp):
    return TargetOutput("answer", retrieved=["d1"])


def test_the_report_shows_unpaired_cases_and_the_not_applicable_share(kit):
    def cases(n, prompt_for):
        return [
            {
                "case_key": f"k{i:02}",
                "prompt": prompt_for(i),
                "reference": "r",
                "retrieved": ["d1"],
                "relevance": {"d1": 1},
            }
            for i in range(n)
        ]

    kit.datasets.import_cases("rag", cases(40, lambda i: f"q{i}"))
    kit.datasets.import_cases("rag", cases(36, lambda i: f"q{i}" if i < 30 else "changed"))
    cite = EvaluatorSpec(kind="citation_check", name="cite")
    runs = []
    for ref, fn, fp in (("rag@1", cites, "A"), ("rag@2", uncited, "B")):
        target = CallableTarget(fn, name="rag", fingerprint=fp)
        run = kit.controller.create(ref, target=target, evaluators=[cite], policy=FAST)
        kit.controller.execute(run.id, target=target)
        runs.append(run)
    cmp = compare(kit, runs[0].id, runs[1].id)
    assert cmp.unpaired == {"input_changed": 6, "only_baseline": 4, "only_candidate": 0}
    html = render_report(kit, runs[1].id, comparison=cmp)
    assert "Cases not paired" in html and "input_changed" in html
    assert "Not-applicable share of the paired cases" in html
    assert "<h3>Not applicable</h3>" in html


def test_gate_settings_are_validated_and_selectors_that_match_nothing_fail(kit):
    with pytest.raises(ValueError, match="allow_unpinned"):
        Gates(allow_unpinned="yes")
    dataset(kit, 35)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)
    kit.controller.execute(run.id)
    cmp = compare(kit, run.id, run.id)
    gates = Gates(
        metrics=(MetricGate("exact_match:em.match", "higher", delta=0.02),),
        must_pass={"llm_judge:nothing": 0.0},
        max_truncated_share=0.5,
    )
    result = evaluate_gates(kit, run.id, gates, cmp)
    assert any(
        f["code"] == "must_pass" and "no evaluator matches" in f["message"] for f in result.failures
    )
    assert not any(f["code"] == "truncated_share" for f in result.failures)  # 0% is under the cap


def test_evaluators_present_in_only_the_baseline_are_reported(kit):
    dataset(kit, 35)
    rx = EvaluatorSpec(kind="regex", name="digits", params={"pattern": r"\d"})
    a = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM, rx], policy=FAST)
    b = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM], policy=FAST)
    kit.controller.execute(a.id)
    kit.controller.execute(b.id)
    cmp = compare(kit, a.id, b.id)
    assert any("exists only in the baseline" in i for i in cmp.informational)


def test_two_threads_queueing_groups_are_all_written_exactly_once(tmp_path):
    store = Store(tmp_path / "s")
    w = writer(store, batch_size=5)

    def produce(prefix):
        for i in range(50):
            w.put(item(f"{prefix}{i}"), item(f"{prefix}{i}-skipped"))

    threads = [threading.Thread(target=produce, args=(p,)) for p in "xy"]
    [t.start() for t in threads]
    [t.join() for t in threads]
    w.close()
    written = [k for batch in store.calls for k in batch]
    assert len(written) == 200 and len(set(written)) == 200
    for batch in store.calls:  # a group is never split: an item and its -skipped are adjacent
        for i, key in enumerate(batch):
            if key.endswith("-skipped"):
                assert i > 0 and batch[i - 1] == key.removesuffix("-skipped")
