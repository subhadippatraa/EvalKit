"""The executor, end to end: dataset -> run -> target -> evaluators -> results, and every way it
can go wrong (partial failure, retry, cancel, crash, storage failure) without losing or inventing
a result."""

import hashlib
import json
import sqlite3
import threading
import time

import pytest
from conftest import judged

from evalkit import (
    EvalFailure,
    EvalKit,
    EvaluatorSpec,
    FailureClass,
    Rubric,
    RunError,
)
from evalkit.calls import CancelToken
from evalkit.engine import ExecPolicy, PreflightError
from evalkit.evaluators import llm_judge_spec, resolve
from evalkit.llm import LLMResponse, Usage
from evalkit.runs import CaseOutcome
from evalkit.targets import (
    CallableTarget,
    ModelTarget,
    PrecomputedTarget,
    ReuseTarget,
    TargetOutput,
)

N = 12
EM = EvaluatorSpec(kind="exact_match", name="em")
RX = EvaluatorSpec(kind="regex", name="digits", params={"pattern": r"\d"})
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}


def dataset(kit, n=N, name="qa", **extra):
    cases = [
        {"case_key": f"c{i:03}", "prompt": f"q{i}", "reference": f"a{i}", "output": f"a{i}"} | extra
        for i in range(n)
    ]
    kit.datasets.import_cases(name, cases)


def answer(inp):
    return "a" + inp.prompt[1:]


def create(kit, target=None, evaluators=(EM,), policy=None, ref="qa", **kw):
    return kit.controller.create(
        ref,
        target=target or PrecomputedTarget(),
        evaluators=list(evaluators),
        policy={**FAST, **(policy or {})},
        **kw,
    )


@pytest.fixture
def ds(kit):
    dataset(kit)
    return kit


class Judge:
    provider, model = "fake", "judge-1"

    def __init__(self, responder=None):
        self.calls, self.responder = 0, responder
        self.lock = threading.Lock()

    def call(self, req):
        with self.lock:
            self.calls += 1
            n = self.calls
        if self.responder:
            r = self.responder(n, req)
            if isinstance(r, BaseException):
                raise r
            return r
        return LLMResponse(
            payload=judged(quality=5),
            stop_reason="tool_use",
            usage=Usage(10, 2),
            request_id=f"r{n}",
        )


RUBRIC = Rubric.from_dict({"quality": "Good?"})


def judge_spec(client):
    return llm_judge_spec("quality", RUBRIC, client)


# --- the happy path ---------------------------------------------------------------------------


def test_dataset_to_run_to_target_to_evaluator_to_results(ds):
    run = create(ds)
    assert run.status == "created" and ds.runs.counts(run.id).pending == N
    report = ds.controller.execute(run.id)
    assert (report.status, report.stop_reason, report.units) == ("succeeded", None, N)
    counts = ds.runs.counts(run.id)
    assert (counts.complete, counts.failed, counts.pending, counts.missing) == (N, 0, 0, 0)
    assert counts.evaluator_results == {EM.key if False else resolve(EM).key: {"ok": N}}
    for cr in ds.runs.case_results(run.id):
        (er,) = ds.runs.evaluator_results(cr.id)
        assert (er.status, er.verdict, er.metrics) == ("ok", "PASS", {"match": 1.0})
    assert ds.runs.verify(run.id).ok
    assert ds.runs.get(run.id).status == "succeeded"


def test_scores_reflect_the_data_not_the_engine(kit):
    dataset(kit, 6)
    kit.datasets.import_cases(
        "mixed",
        [
            {"case_key": "right", "prompt": "p", "reference": "x", "output": "x"},
            {"case_key": "wrong", "prompt": "p", "reference": "x", "output": "y"},
        ],
    )
    run = create(kit, ref="mixed")
    kit.controller.execute(run.id)
    scores = {
        cr.case_key: kit.runs.evaluator_results(cr.id)[0].metrics["match"]
        for cr in kit.runs.case_results(run.id)
    }
    assert scores == {"right": 1.0, "wrong": 0.0}  # a wrong answer is a real zero, not a failure


def test_multiple_evaluators_are_independent_and_all_recorded(ds):
    judge = Judge()
    run = create(ds, evaluators=[EM, RX, judge_spec(judge)])
    ds.controller.execute(run.id, clients=[judge])
    counts = ds.runs.counts(run.id)
    assert len(counts.evaluator_results) == 3
    assert all(v == {"ok": N} for v in counts.evaluator_results.values())
    cr = next(iter(ds.runs.case_results(run.id)))
    results = {r.evaluator_key.split(":")[0]: r for r in ds.runs.evaluator_results(cr.id)}
    assert set(results) == {"exact_match", "regex", "llm_judge"}
    assert (
        results["llm_judge"].metrics["score"] == 1.0
        and results["llm_judge"].attempts[0].input_tokens == 10
    )
    assert results["exact_match"].attempts == []  # deterministic: no external call
    assert judge.calls == N


def test_a_callable_target_produces_outputs_and_retrieved_docs(kit):
    kit.datasets.import_cases(
        "rag",
        [
            {"case_key": "a", "prompt": "q", "relevance": {"d1": 2}},
            {"case_key": "b", "prompt": "q", "relevance": {"d9": 1}},
        ],
    )
    t = CallableTarget(
        lambda i: TargetOutput("answer", retrieved=["d1", "d2"], usage=Usage(7, 3)),
        name="rag@1",
        fingerprint="v1",
    )
    ev = EvaluatorSpec(kind="retrieval", name="ret", params={"k": [2]})
    run = create(kit, t, [ev], ref="rag")
    kit.controller.execute(run.id, target=t)
    a = kit.runs.case_result(run.id, "a")
    assert (a.output, a.retrieved) == ("answer", ["d1", "d2"])
    assert (a.attempts[0].provider, a.attempts[0].input_tokens, a.attempts[0].output_tokens) == (
        "callable",
        7,
        3,
    )
    assert a.duration_ms is not None and a.started_at <= a.finished_at
    by_case = {
        r.case_key: kit.runs.evaluator_results(r.id)[0] for r in kit.runs.case_results(run.id)
    }
    assert by_case["a"].metrics["recall@2"] == 1.0 and by_case["b"].metrics["recall@2"] == 0.0


def test_targets_never_see_reference_or_labels(kit):
    seen = []

    def spy(inp):
        seen.append(inp)
        return "x"

    kit.datasets.import_cases(
        "d",
        [{"case_key": "k", "prompt": "p", "reference": "SECRET", "relevance": {"d": 1},
          "tags": ["t"], "metadata": {"m": 1}, "output": "PRECOMPUTED"}],
    )  # fmt: skip
    t = CallableTarget(spy, name="spy", fingerprint="1")
    run = create(kit, t, [RX], ref="d")
    kit.controller.execute(run.id, target=t)
    (inp,) = seen
    assert inp.provided is None and "SECRET" not in repr(inp) and "PRECOMPUTED" not in repr(inp)


def test_a_model_target_calls_its_client_and_records_usage(ds):
    class Model:
        provider, model = "fake", "m1"

        def __init__(self):
            self.requests = []

        def call(self, req):
            self.requests.append(req)
            return LLMResponse(text="a" + req.user[1:], stop_reason="end_turn", usage=Usage(5, 2))

    client = Model()
    t = ModelTarget(client, "{prompt}")
    run = create(ds, t)
    report = ds.controller.execute(run.id, target=t)
    assert report.status == "succeeded" and len(client.requests) == N
    assert {r.role for r in client.requests} == {"target"}
    assert report.spend.input_tokens == 5 * N and report.spend.output_tokens == 2 * N
    assert ds.runs.counts(run.id).evaluator_results[resolve(EM).key] == {"ok": N}


# --- failures are attributed where they happen ------------------------------------------------


def test_a_target_exception_is_a_target_failure_and_its_evaluators_are_skipped_not_scored(ds):
    def flaky(inp):
        if inp.case_key in ("c003", "c007"):
            raise RuntimeError("pipeline crashed")
        return answer(inp)

    t = CallableTarget(flaky, name="flaky", fingerprint="1")
    run = create(ds, t, [EM, RX])
    report = ds.controller.execute(run.id, target=t)
    assert report.status == "succeeded"  # the run finished; two of its cases failed
    counts = ds.runs.counts(run.id)
    assert (counts.complete, counts.failed) == (N - 2, 2)
    for key in ("c003", "c007"):
        cr = ds.runs.case_result(run.id, key)
        assert (cr.status, cr.output, cr.failure.failure_class, cr.failure.kind) == (
            "failed", None, FailureClass.TARGET, "exception",
        )  # fmt: skip
        ers = ds.runs.evaluator_results(cr.id)
        assert {e.status for e in ers} == {"skipped"} and all(not e.metrics for e in ers)
    assert ds.runs.failure_counts(run.id)[0].count == 2
    assert ds.runs.verify(run.id).ok


def test_a_slow_target_times_out_retries_once_and_keeps_the_evidence(kit):
    dataset(kit, 2)

    def slow(inp):
        time.sleep(0.5)
        return "late"

    t = CallableTarget(slow, name="slow", fingerprint="1")
    run = create(kit, t, policy={"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 0.05}})
    kit.controller.execute(run.id, target=t)
    cr = kit.runs.case_result(run.id, "c000")
    assert cr.failure.kind == "timeout" and cr.failure.failure_class is FailureClass.TARGET
    assert [(a.n, a.error_kind) for a in cr.attempts] == [(1, "timeout"), (2, "timeout")]
    assert all(a.duration_ms >= 45 for a in cr.attempts) and cr.duration_ms >= 90


def test_a_judge_that_fails_twice_is_an_evaluator_failure_and_the_target_is_untouched(ds):
    judge = Judge(lambda n, req: LLMResponse(payload={"bogus": 1}, stop_reason="tool_use"))
    run = create(ds, evaluators=[EM, judge_spec(judge)])
    ds.controller.execute(run.id, clients=[judge])
    counts = ds.runs.counts(run.id)
    assert counts.complete == N and counts.failed == 0
    jkey = judge_spec(judge).key
    assert counts.evaluator_results[jkey] == {"failed": N}
    assert counts.evaluator_results[resolve(EM).key] == {"ok": N}  # the other evaluator is fine
    cr = ds.runs.case_result(run.id, "c000")
    (failed,) = [e for e in ds.runs.evaluator_results(cr.id) if e.evaluator_key == jkey]
    assert (failed.failure.failure_class, failed.failure.kind) == (
        FailureClass.EVALUATOR,
        "invalid_output",
    )
    assert [a.n for a in failed.attempts] == [1, 2] and all(a.raw for a in failed.attempts)
    assert all(f.failure.failure_class is FailureClass.EVALUATOR for f in ds.runs.failures(run.id))


def test_a_judge_that_recovers_on_retry_records_both_attempts(ds):
    def responder(n, req):
        if n == 1:
            return EvalFailure("infrastructure", "rate_limited", "429", retry_after_s=0.001)
        return LLMResponse(payload=judged(quality=4), stop_reason="tool_use")

    judge = Judge(responder)
    dataset(ds, 1, name="one")
    run = create(ds, evaluators=[judge_spec(judge)], ref="one")
    ds.controller.execute(run.id, clients=[judge], overrides={"concurrency": 1})
    cr = ds.runs.case_result(run.id, "c000")
    (er,) = ds.runs.evaluator_results(cr.id)
    assert er.status == "ok" and [(a.n, a.outcome) for a in er.attempts] == [
        (1, "failed"),
        (2, "ok"),
    ]


def test_retry_exhaustion_is_an_infrastructure_failure_not_a_quality_failure(kit):
    dataset(kit, 2)
    judge = Judge(lambda n, req: EvalFailure("infrastructure", "provider_unavailable", "503"))
    run = create(kit, evaluators=[judge_spec(judge)], policy={"breaker_threshold": 1000})
    kit.controller.execute(run.id, clients=[judge], overrides={"concurrency": 1})
    cr = kit.runs.case_result(run.id, "c000")
    (er,) = kit.runs.evaluator_results(cr.id)
    assert (er.failure.failure_class, er.failure.kind) == (
        FailureClass.INFRA,
        "provider_unavailable",
    )
    assert len(er.attempts) == 4 and er.metrics == {} and er.verdict is None  # 4 = max_attempts
    assert cr.status == "complete"  # the target's answer stands


def test_evaluator_bugs_and_nonfinite_metrics_become_internal_errors_never_scores(ds, monkeypatch):
    from evalkit import evaluators

    def nan_metric(self, inp, call):
        return evaluators.EvalOutcome({"match": float("nan")}, "PASS")

    monkeypatch.setattr(evaluators.ExactMatch, "evaluate", nan_metric)
    run = create(ds)
    ds.controller.execute(run.id)
    (er,) = ds.runs.evaluator_results(ds.runs.case_result(run.id, "c000").id)
    assert er.status == "failed" and er.failure.kind == "internal_error"
    assert er.failure.failure_class is FailureClass.EVALUATOR and er.metrics == {}

    def crash(self, inp, call):
        raise KeyError("bug")

    monkeypatch.setattr(evaluators.ExactMatch, "evaluate", crash)
    run2 = create(ds)
    ds.controller.execute(run2.id)
    (er2,) = ds.runs.evaluator_results(ds.runs.case_result(run2.id, "c000").id)
    assert er2.failure.kind == "internal_error" and "KeyError" in er2.failure.message


def test_a_bug_in_the_target_stage_is_infrastructure_internal_error_not_a_target_failure(
    ds, monkeypatch
):
    from evalkit import targets

    monkeypatch.setattr(targets.PrecomputedTarget, "generate", lambda *a: 1 / 0)
    run = create(ds)
    ds.controller.execute(run.id)
    cr = ds.runs.case_result(run.id, "c000")
    assert (cr.failure.failure_class, cr.failure.kind) == (FailureClass.INFRA, "internal_error")


def test_an_oversized_target_output_is_a_contract_violation(kit):
    from evalkit import Limits

    dataset(kit, 1)
    small = EvalKit(kit.store, Limits(max_field_bytes=10))
    t = CallableTarget(lambda i: "x" * 50, name="big", fingerprint="1")
    run = small.controller.create("qa", target=t, evaluators=[EM], policy=FAST)
    small.controller.execute(run.id, target=t)
    cr = kit.runs.case_result(run.id, "c000")
    assert (cr.failure.failure_class, cr.failure.kind) == (
        FailureClass.TARGET,
        "contract_violation",
    )


# --- stopping: systemic, breaker, budget, cancel ----------------------------------------------


def test_a_systemic_failure_stops_the_run_at_once_as_failed_with_results_kept(kit):
    dataset(kit, 40)
    judge = Judge(lambda n, req: EvalFailure("infrastructure", "auth", "AccessDenied"))
    run = create(kit, evaluators=[EM, judge_spec(judge)], policy={"concurrency": 2})
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("failed", "infrastructure.auth")
    assert judge.calls <= 2 + 4 + 2  # nowhere near 40: in-flight units only, never retried
    counts = kit.runs.counts(run.id)
    assert counts.pending > 0 and counts.complete < 40
    assert kit.runs.verify(run.id).ok


def test_resuming_after_the_cause_is_fixed_finishes_only_what_is_left(kit):
    dataset(kit, 20)
    bad = Judge(lambda n, req: EvalFailure("infrastructure", "auth", "denied"))
    run = create(kit, evaluators=[judge_spec(bad)], policy={"concurrency": 1})
    kit.controller.execute(run.id, clients=[bad])
    assert kit.runs.get(run.id).status == "failed"
    before = {r.case_key: r.id for r in kit.runs.case_results(run.id) if r.status != "pending"}
    good = Judge()
    report = kit.controller.execute(run.id, clients=[good])  # same provider/model: the key matches
    assert report.status == "succeeded"
    counts = kit.runs.counts(run.id)
    assert counts.pending == 0 and counts.complete == 20
    after = {r.case_key: r.id for r in kit.runs.case_results(run.id)}
    assert all(after[k] == v for k, v in before.items())  # recorded results were not redone
    # cases whose judge auth failed once keep that failure (write-once): visible, not retried
    failed = kit.runs.failures(run.id, failure_class="infrastructure")
    assert 0 < len(failed) <= 3 and all(f.failure.kind == "auth" for f in failed)
    assert kit.runs.verify(run.id).ok


def test_the_breaker_stops_the_run_partial_after_consecutive_infra_failures(kit):
    dataset(kit, 60)
    judge = Judge(lambda n, req: EvalFailure("infrastructure", "provider_unavailable", "503"))
    retry = {**FAST["retry"], "max_attempts": 1}
    run = create(
        kit,
        evaluators=[judge_spec(judge)],
        policy={"concurrency": 1, "breaker_threshold": 5, "retry": retry},
    )
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "provider_outage")
    assert judge.calls < 15 and kit.runs.counts(run.id).pending > 30


def test_a_budget_stops_dispatch_and_the_overshoot_is_bounded(kit):
    dataset(kit, 50)
    judge = Judge()
    run = create(
        kit, evaluators=[judge_spec(judge)], policy={"concurrency": 2, "budget": {"max_calls": 10}}
    )
    report = kit.controller.execute(run.id, clients=[judge])
    assert (report.status, report.stop_reason) == ("partial", "budget")
    window = 2 * 2
    assert 10 <= judge.calls <= 10 + window  # spend <= budget + window x unit cost
    assert kit.runs.counts(run.id).pending == 50 - judge.calls


def test_a_budgeted_run_resumes_with_a_raised_budget_without_changing_its_identity(kit):
    dataset(kit, 30)
    judge = Judge()
    run = create(
        kit,
        evaluators=[judge_spec(judge)],
        policy={"concurrency": 1, "budget": {"max_tokens": 100}},
    )
    first = kit.controller.execute(run.id, clients=[judge])
    assert first.stop_reason == "budget"
    second = kit.controller.execute(
        run.id, clients=[judge], overrides={"budget": {"max_tokens": 10_000}}
    )
    assert second.status == "succeeded"
    assert kit.runs.get(run.id).identity_hash == run.identity_hash and judge.calls == 30


def test_cancellation_stops_dispatch_keeps_completed_work_and_resume_finishes(ds):
    token = CancelToken()
    run = create(ds, policy={"concurrency": 1})
    report = ds.controller.execute(
        run.id, token=token, on_progress=lambda n: token.cancel("user") if n == 4 else None
    )
    assert (report.status, report.stop_reason) == ("cancelled", "user")
    done = ds.runs.counts(run.id)
    assert 4 <= done.complete < N and done.pending == N - done.complete
    resumed = ds.controller.execute(run.id)
    assert resumed.status == "succeeded" and ds.runs.counts(run.id).complete == N
    assert ds.runs.verify(run.id).ok
    rows = ds.store._conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT case_id) FROM case_results"
    ).fetchone()
    assert tuple(rows) == (N, N)  # exactly one result per case


def test_cancelling_before_starting_dispatches_nothing(ds):
    token = CancelToken()
    token.cancel("never mind")
    run = create(ds)
    report = ds.controller.execute(run.id, token=token)
    assert report.status == "cancelled" and report.units == 0
    assert ds.runs.counts(run.id).pending == N


# --- resume and crash recovery ---------------------------------------------------------------


def test_a_crash_between_the_case_result_and_its_evaluators_is_repaired_on_resume(ds):
    run = create(ds, evaluators=[EM, RX])
    ds.runs.start(run.id)
    ds.runs.record_case_result(run.id, "c000", CaseOutcome.complete("a0"))  # process died here
    key_em = resolve(EM).key
    from evalkit import EvaluatorOutcome

    cr = ds.runs.case_result(run.id, "c000")
    ds.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(evaluator_key=key_em, status="ok", metrics={"match": 1.0}, verdict="PASS"),
    )
    report = ds.controller.execute(run.id)
    assert report.status == "succeeded" and report.units == N
    assert {e.evaluator_key.split(":")[0] for e in ds.runs.evaluator_results(cr.id)} == {
        "exact_match",
        "regex",
    }
    counts = ds.runs.counts(run.id)
    assert all(sum(v.values()) == N for v in counts.evaluator_results.values())
    assert ds.runs.verify(run.id).ok


def test_executing_a_succeeded_run_is_refused(ds):
    run = create(ds)
    ds.controller.execute(run.id)
    with pytest.raises(RunError, match="already succeeded"):
        ds.controller.execute(run.id)


# --- ordering ---------------------------------------------------------------------------------


def processed_order(kit, run):
    rows = kit.store._conn.execute(
        "SELECT c.case_key FROM case_results r JOIN cases c ON c.id = r.case_id "
        "WHERE r.run_id = ? AND r.status != 'pending' ORDER BY r.finished_at, r.id",
        (run.id,),
    )
    return {r[0] for r in rows}


def expected_first(keys, seed, k):
    def h(key):
        return int.from_bytes(hashlib.sha256(f"{seed}\x00{key}".encode()).digest()[:8], "big") >> 1

    return set(sorted(keys, key=h)[:k])


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_units_are_processed_in_seeded_hash_order_so_a_partial_run_is_a_sample(kit, seed):
    dataset(kit, 40)
    keys = [f"c{i:03}" for i in range(40)]
    # the precomputed target makes no calls, so use a callable to spend budget
    t = CallableTarget(lambda i: "x", name="t", fingerprint="1")
    run = create(kit, t, policy={"concurrency": 1, "seed": seed, "budget": {"max_calls": 10}})
    report = kit.controller.execute(run.id, target=t)
    assert report.stop_reason == "budget"
    done = processed_order(kit, run)
    assert 10 <= len(done) <= 10 + 2  # the budget check happens at dispatch: window overshoot
    assert done == expected_first(keys, seed, len(done))  # ...and what ran is a hash-order prefix


def test_a_different_seed_samples_different_cases_and_the_same_seed_repeats(kit):
    dataset(kit, 40)
    t = CallableTarget(lambda i: "x", name="t", fingerprint="1")

    def first_ten(seed):
        r = create(kit, t, policy={"concurrency": 1, "seed": seed, "budget": {"max_calls": 10}})
        kit.controller.execute(r.id, target=t)
        return processed_order(kit, r)

    assert first_ten(1) == first_ten(1) and first_ten(1) != first_ten(2)


def test_the_processing_order_is_stored_and_matches_the_hash(ds):
    run = create(ds, policy={"seed": 5})
    rows = ds.store._conn.execute(
        "SELECT c.case_key, r.ord FROM case_results r JOIN cases c ON c.id = r.case_id "
        "WHERE r.run_id = ?",
        (run.id,),
    ).fetchall()
    for key, ord_ in rows:
        digest = hashlib.sha256(f"5\x00{key}".encode()).digest()
        assert ord_ == int.from_bytes(digest[:8], "big") >> 1


# --- storage failure: paid results are never lost ---------------------------------------------


def test_a_storage_failure_spills_results_stops_the_run_and_resume_replays_them(tmp_path):
    kit = EvalKit.open(tmp_path / "e.db")
    dataset(kit, 20)
    t = CallableTarget(lambda i: "a" + i.prompt[1:], name="t", fingerprint="1")
    run = create(kit, t, policy={"concurrency": 1, "batch_size": 4})
    real = type(kit.store).write_batch
    state = {"calls": 0}

    def flaky(self, run_id, items):
        state["calls"] += 1
        if state["calls"] >= 3:
            raise sqlite3.OperationalError("disk I/O error")
        return real(self, run_id, items)

    type(kit.store).write_batch = flaky
    try:
        report = kit.controller.execute(run.id, target=t)
    finally:
        type(kit.store).write_batch = real
    assert (report.status, report.stop_reason) == ("partial", "infrastructure.storage")
    assert report.spilled > 0
    spill = kit.store.spill_dir / f"{run.id}.jsonl"
    assert spill.is_file() and oct(spill.stat().st_mode & 0o777) == "0o600"
    lines = [json.loads(line) for line in spill.read_text().splitlines()]
    assert len(lines) == report.spilled and {"kind", "case_key", "outcome", "attempts"} <= set(
        lines[0]
    )

    final = kit.controller.execute(run.id, target=t)  # replays the spill, then finishes
    assert final.status == "succeeded" and not spill.exists()
    rows = kit.store._conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT case_id) FROM case_results"
    ).fetchone()
    assert tuple(rows) == (20, 20)  # nothing lost, nothing duplicated
    assert kit.runs.verify(run.id).ok
    kit.close()


def test_a_bad_spill_file_is_reported_and_kept(tmp_path):
    kit = EvalKit.open(tmp_path / "e.db")
    dataset(kit, 3)
    run = create(kit)
    kit.store.spill_dir.mkdir(parents=True)
    path = kit.store.spill_dir / f"{run.id}.jsonl"
    path.write_text("not json\n")
    with pytest.raises(RunError, match="not a valid spilled result"):
        kit.controller.execute(run.id)
    assert path.exists()
    kit.close()


def test_an_in_memory_store_keeps_spilled_results_on_the_writer():
    from evalkit.engine import ResultWriter
    from evalkit.runs import WriteItem

    kit = EvalKit.open(":memory:")
    dataset(kit, 2)
    run = create(kit)
    kit.runs.start(run.id)
    w = ResultWriter(kit.store, run.id, batch_size=2, batch_ms=10, depth=4, retries=0)
    kit.store.write_batch = lambda *a: (_ for _ in ()).throw(sqlite3.OperationalError("locked"))
    item = WriteItem("case", "c000", CaseOutcome.complete("x"))
    w.put(item)
    time.sleep(0.2)
    assert w.failed is not None and w.put(item) is False
    w.close()
    assert len(w.spilled_items) == 2 and w.written == 0


# --- preflight --------------------------------------------------------------------------------


def test_preflight_refuses_a_run_whose_cases_lack_what_an_evaluator_needs(kit):
    kit.datasets.import_cases(
        "d",
        [
            {"case_key": "a", "prompt": "p", "output": "x", "reference": "x"},
            {"case_key": "b", "prompt": "p", "output": "x"},
            {"case_key": "c", "prompt": "p", "output": "x"},
        ],
    )
    with pytest.raises(PreflightError) as info:
        create(kit, ref="d")
    e = info.value.report.errors[0]
    assert (e.code, e.count, e.examples) == ("missing_field", 2, ("b", "c"))
    assert kit.runs.list() == []  # nothing was created


def test_on_missing_not_applicable_turns_the_error_into_a_warning_and_a_not_applicable_result(kit):
    kit.datasets.import_cases(
        "d",
        [
            {"case_key": "a", "prompt": "p", "output": "x", "reference": "x"},
            {"case_key": "b", "prompt": "p", "output": "x"},
        ],
    )
    run = create(kit, ref="d", policy={"on_missing": "not_applicable"})
    kit.controller.execute(run.id)
    b = kit.runs.evaluator_results(kit.runs.case_result(run.id, "b").id)[0]
    assert (b.status, b.metrics) == ("not_applicable", {}) and "reference" in b.detail["reason"]
    warnings = [i for i in run.environment["preflight"]["issues"] if i["level"] == "warning"]
    assert any(w["code"] == "missing_field" for w in warnings)


def test_a_precomputed_target_needs_every_case_to_carry_an_output(kit):
    kit.datasets.import_cases("d", [{"case_key": "a", "prompt": "p", "reference": "x"}])
    with pytest.raises(PreflightError, match="output"):
        create(kit, ref="d")


def test_preflight_warns_about_unpinned_callables_contamination_and_lint(kit):
    kit.datasets.import_cases(
        "d",
        [
            {
                "case_key": "a",
                "prompt": "What is the capital of France exactly?",
                "reference": "Paris",
            },
            {
                "case_key": "b",
                "prompt": "What is the capital of France exactly?",
                "reference": "Lyon",
            },
        ],
    )

    class M:
        provider, model = "fake", "m"

        def call(self, req):
            return LLMResponse(text="x")

    leaky = ModelTarget(M(), "Example: What is the capital of France exactly? -> Paris\n{prompt}")
    report = kit.controller.preflight("d", leaky, [RX])
    codes = {i.code for i in report.issues}
    assert {"contamination", "lint.duplicate_prompt"} <= codes and report.ok
    unpinned = kit.controller.preflight("d", CallableTarget(lambda i: "x"), [RX])
    assert "unpinned_target" in {i.code for i in unpinned.issues}
    assert report.estimate == {"target_calls": 2, "judge_calls": 0}


def test_preflight_reports_a_bad_evaluator_config_and_a_bad_policy_before_creating_anything(ds):
    bad = EvaluatorSpec(kind="regex", name="r", params={"pattern": "("})
    with pytest.raises(PreflightError) as info:
        create(ds, evaluators=[bad], policy={"concurrency": 0})
    codes = [(i.code, i.message[:20]) for i in info.value.report.errors]
    assert len(codes) == 2 and all(c == "bad_config" for c, _ in codes)
    assert ds.runs.list() == []


def test_the_preflight_report_is_stored_on_the_run(ds):
    run = create(ds)
    pre = run.environment["preflight"]
    assert pre["ok"] and pre["cases"] == N and pre["dataset"] == "qa@1"


# --- reuse and target binding ----------------------------------------------------------------


def test_reuse_rescoring_copies_outputs_and_inherits_target_failures(ds):
    def flaky(inp):
        if inp.case_key == "c002":
            raise RuntimeError("boom")
        return answer(inp)

    t = CallableTarget(flaky, name="f", fingerprint="1")
    first = create(ds, t, [EM])
    ds.controller.execute(first.id, target=t)
    second = create(ds, ReuseTarget(first.id), [EM, RX])
    report = ds.controller.execute(second.id)
    assert report.status == "succeeded"
    a, b = ds.runs.case_result(first.id, "c001"), ds.runs.case_result(second.id, "c001")
    assert a.output == b.output and b.attempts == []  # no target call, zero target cost
    inherited = ds.runs.case_result(second.id, "c002")
    assert (inherited.failure.failure_class, inherited.failure.kind) == (
        FailureClass.TARGET,
        "exception",
    )
    assert "inherited from source run" in inherited.failure.message
    assert {e.status for e in ds.runs.evaluator_results(inherited.id)} == {"skipped"}
    assert ds.runs.get(second.id).config.target.identity == {"source_run_id": first.id}


def test_reuse_refuses_a_source_of_another_dataset_version_or_an_unfinished_one(ds):
    src = create(ds)  # planned but never executed: pending
    with pytest.raises(PreflightError, match="not finished"):
        create(ds, ReuseTarget(src.id))
    dataset(ds, 3, name="other")
    with pytest.raises(PreflightError, match="not this run's dataset version"):
        create(ds, ReuseTarget(src.id), ref="other")
    with pytest.raises(PreflightError, match="not found"):
        create(ds, ReuseTarget("no-such-run"))


def test_the_supplied_target_must_match_the_frozen_one(ds):
    t1 = CallableTarget(answer, name="t", fingerprint="v1")
    run = create(ds, t1)
    with pytest.raises(RunError, match="cannot be persisted"):
        ds.controller.execute(run.id)
    with pytest.raises(RunError, match="does not match the run's frozen target"):
        ds.controller.execute(run.id, target=CallableTarget(answer, name="t", fingerprint="v2"))
    assert ds.runs.get(run.id).status == "created"  # nothing changed
    assert ds.controller.execute(run.id, target=t1).status == "succeeded"


def test_a_missing_llm_client_is_a_config_error_before_any_state_change(ds):
    judge = Judge()
    run = create(ds, evaluators=[judge_spec(judge)])
    from evalkit import ConfigError

    with pytest.raises(ConfigError, match="no LLM client"):
        ds.controller.execute(run.id)
    assert ds.runs.get(run.id).status == "created"


def test_a_stale_judge_spec_is_refused_instead_of_silently_judging_differently(ds, monkeypatch):
    judge = Judge()
    run = create(ds, evaluators=[judge_spec(judge)])
    from evalkit import evaluators

    monkeypatch.setattr(evaluators, "PROMPT_VERSION", "changed-judge-prompt")
    from evalkit import ConfigError

    with pytest.raises(ConfigError, match="not in canonical form"):
        ds.controller.execute(run.id, clients=[judge])


# --- concurrency, backpressure, throughput ----------------------------------------------------


def test_in_flight_units_never_exceed_the_window(kit):
    dataset(kit, 60)
    active, peak, lock = 0, 0, threading.Lock()

    def slow(inp):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return "a"

    t = CallableTarget(slow, name="s", fingerprint="1")
    run = create(kit, t, policy={"concurrency": 3, "window": 5})
    kit.controller.execute(run.id, target=t)
    assert 2 <= peak <= 3  # the pool bounds concurrent calls; the window bounds queued units


def test_a_thousand_units_persist_exactly_once_with_three_evaluators(kit):
    dataset(kit, 1000)
    run = create(
        kit, evaluators=[EM, RX, EvaluatorSpec(kind="regex", name="a", params={"pattern": "a"})]
    )
    t0 = time.monotonic()
    report = kit.controller.execute(run.id)
    elapsed = time.monotonic() - t0
    assert report.status == "succeeded" and report.units == 1000
    c = kit.store._conn
    assert c.execute("SELECT COUNT(*), COUNT(DISTINCT case_id) FROM case_results").fetchone()[
        :
    ] == (1000, 1000)
    assert c.execute(
        "SELECT COUNT(*), COUNT(DISTINCT case_result_id || evaluator_key) FROM evaluator_results"
    ).fetchone()[:] == (3000, 3000)
    assert c.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 3000
    assert elapsed < 30 and kit.runs.verify(run.id).ok


def test_progress_is_reported_once_per_unit(ds):
    seen = []
    ds.controller.execute(create(ds).id, on_progress=seen.append)
    assert sorted(seen) == list(range(1, N + 1))


def test_the_rate_limit_spaces_calls(kit):
    dataset(kit, 6)
    t = CallableTarget(lambda i: "x", name="t", fingerprint="1")
    run = create(kit, t, policy={"concurrency": 2, "max_calls_per_s": 20})
    t0 = time.monotonic()
    kit.controller.execute(run.id, target=t)
    assert time.monotonic() - t0 >= 5 / 20 - 0.02


# --- policy -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"concurrency": 0},
        {"concurrency": 65},
        {"concurrency": True},
        {"window": 1, "concurrency": 4},
        {"batch_size": 0},
        {"seed": "1"},
        {"on_missing": "ignore"},
        {"max_calls_per_s": 0},
        {"unknown": 1},
        {"retry": {"nope": 1}},
        {"budget": {"max_dollars": 1}},
        {"breaker_threshold": 0},
    ],
)
def test_invalid_execution_policies_are_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        ExecPolicy.from_mapping(bad)


def test_the_policy_never_affects_run_identity(kit):
    dataset(kit, 3)
    a = create(kit, policy={"concurrency": 1})
    b = create(kit, policy={"concurrency": 32, "seed": 9})
    assert a.identity_hash == b.identity_hash and a.exec_hash != b.exec_hash


def test_execution_policy_defaults():
    p = ExecPolicy()
    assert (p.concurrency, p.in_flight, p.batch_size, p.on_missing) == (4, 8, 100, "abort")
    assert ExecPolicy(concurrency=3, window=10).in_flight == 10


def test_a_systemic_failure_outranks_a_simultaneous_cancellation(kit):
    dataset(kit, 20)
    judge = Judge(lambda n, req: EvalFailure("infrastructure", "auth", "denied"))
    run = create(kit, evaluators=[judge_spec(judge)], policy={"concurrency": 1})
    token = CancelToken()
    report = kit.controller.execute(
        run.id, clients=[judge], token=token, on_progress=lambda n: token.cancel("user")
    )
    assert (report.status, report.stop_reason) == ("failed", "infrastructure.auth")


def test_a_silently_lost_evaluator_write_never_yields_a_succeeded_run(kit):
    """Defence in depth: even if a write vanished without an error, `succeeded` is only claimed
    when every evaluator has a result for every finished case."""
    dataset(kit, 6)
    run = create(kit, evaluators=[EM, RX])
    store = type(kit.store)
    real = store.write_batch

    def lossy(self, run_id, items):
        kept = [i for i in items if not (i.kind == "evaluator" and i.case_key == "c002")]
        return real(self, run_id, kept)

    store.write_batch = lossy
    try:
        report = kit.controller.execute(run.id)
    finally:
        store.write_batch = real
    assert (report.status, report.stop_reason) == ("partial", "incomplete")
    assert kit.controller.execute(run.id).status == "succeeded"  # resume fills the gap
    assert kit.runs.verify(run.id).ok
