"""CaseResult, EvaluatorResult and their relationships; failure classes; partial failures."""

from datetime import timedelta

import pytest
from conftest import EM, JUDGE, RUN_CASES, case, complete, make_run, ok, utc_now
from pydantic import ValidationError

from evalkit import (
    CaseOutcome,
    DuplicateResultError,
    EvalFailure,
    EvaluatorOutcome,
    Failure,
    RunError,
)

EM_KEY, JUDGE_KEY = EM.key, JUDGE.key


def tfail(kind="exception", msg="boom", **kw):
    return CaseOutcome.fail(Failure(failure_class="target", kind=kind, message=msg), **kw)


def efail(key=EM_KEY, cls="evaluator", kind="invalid_output", msg="bad"):
    return EvaluatorOutcome(
        evaluator_key=key,
        status="failed",
        failure=Failure(failure_class=cls, kind=kind, message=msg),
    )


def skipped(key=EM_KEY):
    return EvaluatorOutcome(evaluator_key=key, status="skipped", detail={"reason": "target failed"})


# --- CaseOutcome: an output or a failure, never both, never neither --------------------------


def test_a_case_outcome_is_an_output_xor_a_failure():
    assert CaseOutcome.complete("text").output == "text"
    with pytest.raises(ValidationError, match="exactly one"):
        CaseOutcome()
    with pytest.raises(ValidationError, match="exactly one"):
        CaseOutcome(
            output="x", failure=Failure(failure_class="target", kind="timeout", message="m")
        )


def test_an_empty_output_is_a_real_output_not_a_failure():
    assert CaseOutcome.complete("").output == ""


def test_a_failed_outcome_carries_no_retrieved_docs_and_a_bad_clock_is_refused():
    with pytest.raises(ValidationError, match="retrieved"):
        tfail(retrieved=["d1"])
    now = utc_now()
    with pytest.raises(ValidationError, match="before started_at"):
        CaseOutcome.complete("x", started_at=now, finished_at=now - timedelta(seconds=1))
    with pytest.raises(ValidationError):  # naive datetimes are ambiguous
        CaseOutcome.complete("x", started_at=now.replace(tzinfo=None))


def test_an_evaluator_class_failure_is_not_a_case_failure():
    with pytest.raises(ValidationError, match="a case failure is one of"):
        CaseOutcome.fail(Failure(failure_class="evaluator", kind="refused", message="m"))


def test_lossless_outputs_and_surrogates(kit, run):
    weird = "  leading\r\n\ttabs é́ \U0001f600 <<<EVALKIT:x>>> \x00 end  "
    kit.runs.record_case_result(run.id, "a", CaseOutcome.complete(weird, retrieved=["d2", "d1"]))
    got = kit.runs.case_result(run.id, "a")
    assert got.output == weird and got.retrieved == ["d2", "d1"]  # order kept, nothing normalized
    with pytest.raises(ValidationError, match="unpaired surrogate"):
        CaseOutcome.complete("bad \ud800")


def test_output_size_is_limited(kit, run):
    from evalkit import EvalKit, Limits

    small = EvalKit(kit.store, Limits(max_field_bytes=10))
    with pytest.raises(RunError, match="too large"):
        small.runs.record_case_result(run.id, "a", CaseOutcome.complete("x" * 11))
    small.runs.record_case_result(run.id, "a", CaseOutcome.complete("x" * 10))


# --- planning and recording case results -----------------------------------------------------


def test_plan_creates_one_pending_result_per_case_and_is_idempotent(kit, run):
    assert kit.runs.plan(run.id) == 3
    assert kit.runs.plan(run.id) == 0
    assert [(r.case_key, r.status) for r in kit.runs.case_results(run.id)] == [
        ("a", "pending"), ("b", "pending"), ("c", "pending"),
    ]  # fmt: skip
    counts = kit.runs.counts(run.id)
    assert (counts.total_cases, counts.pending, counts.missing) == (3, 3, 0)


def test_plan_only_fills_gaps_and_never_touches_recorded_results(kit, run):
    complete(kit, run, "b", "kept")
    assert kit.runs.plan(run.id) == 2
    assert kit.runs.case_result(run.id, "b").output == "kept"


def test_recording_completes_the_pending_result_instead_of_adding_a_second(kit, run, store):
    kit.runs.plan(run.id)
    before = store._conn.execute("SELECT id FROM case_results WHERE status='pending'").fetchall()
    done = complete(kit, run, "a")
    assert done.id in {r[0] for r in before}
    assert store._conn.execute("SELECT COUNT(*) FROM case_results").fetchone()[0] == 3
    assert (done.status, done.output, done.case_key) == ("complete", "out", "a")


def test_a_case_that_was_never_planned_can_still_be_recorded(kit, run):
    r = complete(kit, run, "c")
    assert kit.runs.counts(run.id).missing == 2 and r.status == "complete"


def test_timing_is_preserved(kit, run):
    start = utc_now() - timedelta(seconds=2)
    end = utc_now()
    kit.runs.record_case_result(
        run.id, "a", CaseOutcome.complete("o", started_at=start, finished_at=end, duration_ms=2000)
    )
    got = kit.runs.case_result(run.id, "a")
    assert (got.started_at, got.finished_at, got.duration_ms) == (start, end, 2000)


def test_a_case_result_needs_a_running_run(kit):
    run = make_run(kit, EM)  # created, not started
    with pytest.raises(RunError, match="created; results need a running run"):
        complete(kit, run, "a")
    kit.runs.start(run.id)
    complete(kit, run, "a")
    kit.runs.transition(run.id, "partial", stop_reason="x")
    with pytest.raises(RunError, match="partial; results need a running run"):
        complete(kit, run, "b")


def test_planning_is_refused_once_a_run_has_ended(kit, run):
    kit.runs.transition(run.id, "failed", stop_reason="x")
    with pytest.raises(RunError, match="cannot plan a failed run"):
        kit.runs.plan(run.id)


# --- one result per case per run -------------------------------------------------------------


def test_a_second_result_for_the_same_case_is_refused_and_the_first_is_kept(kit, run):
    complete(kit, run, "a", "first")
    with pytest.raises(DuplicateResultError, match="already has a result"):
        complete(kit, run, "a", "second")
    with pytest.raises(DuplicateResultError):
        kit.runs.record_case_result(run.id, "a", tfail())
    assert kit.runs.case_result(run.id, "a").output == "first"


def test_a_failed_result_is_final_too(kit, run):
    kit.runs.record_case_result(run.id, "a", tfail(kind="timeout"))
    with pytest.raises(DuplicateResultError):
        complete(kit, run, "a")
    assert kit.runs.case_result(run.id, "a").failure.kind == "timeout"


def test_the_same_case_in_two_runs_has_two_independent_results(kit):
    r1 = make_run(kit, EM, start=True)
    r2 = make_run(kit, EM, start=True)
    complete(kit, r1, "a", "one")
    complete(kit, r2, "a", "two")
    assert kit.runs.case_result(r1.id, "a").output == "one"
    assert kit.runs.case_result(r2.id, "a").output == "two"


# --- results reference cases of the run's dataset version ------------------------------------


def test_a_case_from_another_version_or_dataset_is_refused(kit, run):
    kit.datasets.import_cases("qa", [case("a", output="v2"), case("only-in-v2")])
    kit.datasets.import_cases("other", [case("foreign")])
    with pytest.raises(RunError, match="not part of run .* dataset version"):
        complete(kit, run, "only-in-v2")  # exists in qa@2, but this run is pinned to qa@1
    with pytest.raises(RunError, match="not part of run"):
        complete(kit, run, "foreign")
    with pytest.raises(RunError, match="not part of run"):
        complete(kit, run, "never-existed")
    assert kit.runs.counts(run.id).complete == 0


def test_the_database_refuses_a_case_of_another_version(kit, run, store):
    kit.datasets.import_cases("qa", [case("a", output="v2")])
    foreign_case = store._conn.execute(
        "SELECT c.id FROM cases c JOIN dataset_versions v ON v.id = c.dataset_version_id "
        "WHERE v.version_no = 2"
    ).fetchone()[0]
    from conftest import refuses

    for case_id in (foreign_case, "no-such-case"):
        refuses(
            store,
            "INSERT INTO case_results (id, run_id, case_id, status, created_at) "
            "VALUES ('x', ?, ?, 'pending', 't')",
            (run.id, case_id),
            match="not part of the run's dataset version",
        )
    refuses(
        store,
        "INSERT INTO case_results (id, run_id, case_id, status, created_at) "
        "VALUES ('x', 'no-such-run', ?, 'pending', 't')",
        (foreign_case,),
        match="not part of the run's dataset version|not accepting",
    )


def test_a_pinned_run_keeps_seeing_its_own_cases_after_a_new_dataset_version(kit, run):
    kit.datasets.import_cases("qa", [case("z-new")])
    assert kit.runs.plan(run.id) == 3
    assert [r.case_key for r in kit.runs.case_results(run.id)] == RUN_CASES


# --- evaluator results: relationships --------------------------------------------------------


def test_an_evaluator_result_belongs_to_its_case_result_and_carries_metrics(kit, run):
    cr = complete(kit, run, "a")
    out = EvaluatorOutcome(
        evaluator_key=JUDGE_KEY,
        status="ok",
        verdict="PASS",
        metrics={"score": 0.75, "criterion.correctness": 4, "criterion.clarity": 3.5},
        detail={"reasoning": {"correctness": "fine"}},
        duration_ms=1200,
    )
    er = kit.runs.record_evaluator_result(cr.id, out)
    assert (er.case_result_id, er.run_id, er.evaluator_key) == (cr.id, run.id, JUDGE_KEY)
    assert (er.status, er.verdict, er.score) == ("ok", "PASS", 0.75)
    assert er.metrics == {"score": 0.75, "criterion.correctness": 4.0, "criterion.clarity": 3.5}
    assert er.detail == {"reasoning": {"correctness": "fine"}} and er.duration_ms == 1200
    assert kit.runs.evaluator_results(cr.id) == [er]


def test_one_case_result_can_have_one_result_per_evaluator_ordered_by_key(kit, run):
    cr = complete(kit, run, "a")
    kit.runs.record_evaluator_result(cr.id, ok(JUDGE_KEY, 0.5))
    kit.runs.record_evaluator_result(cr.id, ok(EM_KEY, 1.0))
    assert [r.evaluator_key for r in kit.runs.evaluator_results(cr.id)] == sorted(
        [EM_KEY, JUDGE_KEY]
    )


def test_a_second_result_from_the_same_evaluator_is_refused(kit, run):
    cr = complete(kit, run, "a")
    kit.runs.record_evaluator_result(cr.id, ok(EM_KEY, 1.0))
    with pytest.raises(DuplicateResultError):
        kit.runs.record_evaluator_result(cr.id, ok(EM_KEY, 0.0))
    (only,) = kit.runs.evaluator_results(cr.id)
    assert only.score == 1.0


def test_an_evaluator_result_cannot_exist_without_its_case_result(kit, run, store):
    with pytest.raises(RunError, match="case result 'nope' not found"):
        kit.runs.record_evaluator_result("nope", ok(EM_KEY))
    from conftest import refuses

    refuses(
        store,
        "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
        "created_at) VALUES ('x', 'nope', ?, ?, 'not_applicable', 't')",
        (run.id, EM_KEY),
    )
    assert store._conn.execute("SELECT COUNT(*) FROM evaluator_results").fetchone()[0] == 0


def test_an_evaluator_the_run_does_not_have_is_refused(kit, run):
    cr = complete(kit, run, "a")
    with pytest.raises(RunError, match="not one of run .* evaluators"):
        kit.runs.record_evaluator_result(cr.id, ok("regex:other:000000000000"))
    assert kit.runs.evaluator_results(cr.id) == []


def test_the_database_refuses_an_unregistered_evaluator_and_a_wrong_run(kit, run, store):
    from conftest import refuses

    cr = complete(kit, run, "a")
    other = make_run(kit, EM, start=True)
    sql = (
        "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
        "detail_json, created_at) VALUES ('x', ?, ?, ?, 'not_applicable', '{}', 't')"
    )
    refuses(store, sql, (cr.id, run.id, "regex:ghost:000000000000"), match="FOREIGN KEY")
    refuses(store, sql, (cr.id, other.id, EM_KEY), match="carry its case result's run")


def test_the_evaluator_results_of_other_case_results_are_not_mixed_up(kit, run):
    a, b = complete(kit, run, "a"), complete(kit, run, "b")
    kit.runs.record_evaluator_result(a.id, ok(EM_KEY, 1.0))
    kit.runs.record_evaluator_result(b.id, ok(EM_KEY, 0.0))
    assert [r.score for r in kit.runs.evaluator_results(a.id)] == [1.0]
    assert [r.score for r in kit.runs.evaluator_results(b.id)] == [0.0]


# --- evaluator outcome coherence -------------------------------------------------------------


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"status": "ok"}, "at least one metric"),
        ({"status": "ok", "metrics": {"score": 1}, "failure": True}, "no failure"),
        ({"status": "not_applicable"}, r"needs detail\['reason'\]"),
        ({"status": "skipped"}, r"needs detail\['reason'\]"),
        ({"status": "not_applicable", "detail": {"reason": ""}}, r"needs detail"),
        (
            {"status": "not_applicable", "detail": {"reason": "r"}, "metrics": {"score": 0}},
            "no metrics",
        ),
        (
            {"status": "skipped", "detail": {"reason": "r"}, "verdict": "FAIL"},
            "no metrics and no verdict",
        ),
        ({"status": "failed"}, "needs a failure"),
        ({"status": "failed", "failure": True, "metrics": {"score": 0}}, "no metrics"),
        ({"status": "failed", "failure": True, "verdict": "FAIL"}, "no metrics and no verdict"),
    ],
)
def test_incoherent_evaluator_outcomes_are_refused(fields, match):
    fields = dict(fields)
    if fields.get("failure"):
        fields["failure"] = Failure(failure_class="evaluator", kind="refused", message="m")
    with pytest.raises(ValidationError, match=match):
        EvaluatorOutcome(evaluator_key="k", **fields)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "1", None, [1]])
def test_metric_values_must_be_finite_real_numbers(value):
    with pytest.raises(ValidationError):
        EvaluatorOutcome(evaluator_key="k", status="ok", metrics={"score": value})


@pytest.mark.parametrize("name", ["", " ", "has space", "-lead", "x" * 129, "a/b"])
def test_metric_names_are_validated(name):
    with pytest.raises(ValidationError, match="invalid metric name"):
        EvaluatorOutcome(evaluator_key="k", status="ok", metrics={name: 1.0})


def test_metric_names_from_the_design_are_valid_and_there_is_a_cap():
    names = ["score", "recall@5", "ndcg@10", "mrr", "criterion.correctness", "hit@1"]
    out = EvaluatorOutcome(evaluator_key="k", status="ok", metrics=dict.fromkeys(names, 0.5))
    assert set(out.metrics) == set(names)
    with pytest.raises(ValidationError, match="at most"):
        EvaluatorOutcome(evaluator_key="k", status="ok", metrics={f"m{i}": 1.0 for i in range(257)})


def test_an_unknown_verdict_and_non_json_detail_are_refused():
    with pytest.raises(ValidationError):
        EvaluatorOutcome(evaluator_key="k", status="ok", metrics={"s": 1}, verdict="MAYBE")
    for detail in ({"x": float("nan")}, {"x": object()}, {1: "k"}):
        with pytest.raises(ValidationError):
            EvaluatorOutcome(evaluator_key="k", status="ok", metrics={"s": 1}, detail=detail)


def test_uncertain_is_a_verdict_not_a_failure(kit, run):
    cr = complete(kit, run, "a")
    out = EvaluatorOutcome(
        evaluator_key=JUDGE_KEY, status="ok", verdict="UNCERTAIN", metrics={"score": 0.5}
    )
    assert kit.runs.record_evaluator_result(cr.id, out).verdict == "UNCERTAIN"


def test_a_verdict_is_optional_for_an_ok_result(kit, run):
    cr = complete(kit, run, "a")
    out = EvaluatorOutcome(evaluator_key=EM_KEY, status="ok", metrics={"recall@5": 0.4})
    er = kit.runs.record_evaluator_result(cr.id, out)
    assert er.verdict is None and er.score is None and er.metrics == {"recall@5": 0.4}


def test_detail_size_is_limited(kit, run):
    from evalkit import EvalKit, Limits

    cr = complete(kit, run, "a")
    small = EvalKit(kit.store, Limits(max_field_bytes=100))
    big = EvaluatorOutcome(
        evaluator_key=EM_KEY, status="ok", metrics={"score": 1}, detail={"x": "y" * 200}
    )
    with pytest.raises(RunError, match="detail is too large"):
        small.runs.record_evaluator_result(cr.id, big)


def test_a_not_applicable_result_is_stored_without_a_metric_or_a_score(kit, run):
    cr = complete(kit, run, "a")
    na = EvaluatorOutcome(
        evaluator_key=EM_KEY, status="not_applicable", detail={"reason": "no relevant documents"}
    )
    er = kit.runs.record_evaluator_result(cr.id, na)
    assert (er.status, er.metrics, er.score, er.verdict) == ("not_applicable", {}, None, None)
    assert er.detail["reason"] == "no relevant documents"


# --- target failures are never scores --------------------------------------------------------


def test_a_target_failure_is_a_case_result_failure_with_no_evaluator_score(kit, run):
    cr = kit.runs.record_case_result(run.id, "a", tfail("timeout", "no answer in 30s"))
    assert (cr.status, cr.output) == ("failed", None)
    assert (cr.failure.failure_class.value, cr.failure.kind) == ("target", "timeout")
    assert kit.runs.evaluator_results(cr.id) == []  # nothing was scored


def test_a_failed_case_result_takes_only_skipped_evaluator_results(kit, run):
    cr = kit.runs.record_case_result(run.id, "a", tfail())
    for outcome in (ok(EM_KEY, 0.0), efail(), _na()):
        with pytest.raises(RunError, match="does not fit a failed case result"):
            kit.runs.record_evaluator_result(cr.id, outcome)
    er = kit.runs.record_evaluator_result(cr.id, skipped(EM_KEY))
    assert (er.status, er.metrics, er.failure, er.verdict) == ("skipped", {}, None, None)
    assert er.detail["reason"] == "target failed"


def _na(key=EM_KEY):
    return EvaluatorOutcome(evaluator_key=key, status="not_applicable", detail={"reason": "r"})


def test_a_complete_case_result_cannot_take_skipped_and_a_pending_one_takes_nothing(kit, run):
    kit.runs.plan(run.id)
    (pending,) = [r for r in kit.runs.case_results(run.id) if r.case_key == "a"]
    with pytest.raises(RunError, match="does not fit a pending case result"):
        kit.runs.record_evaluator_result(pending.id, ok(EM_KEY))
    done = complete(kit, run, "a")
    assert done.id == pending.id
    with pytest.raises(RunError, match="does not fit a complete case result"):
        kit.runs.record_evaluator_result(done.id, skipped())


def test_the_database_enforces_the_same_rules_on_a_raw_insert(kit, run, store):
    from conftest import refuses

    failed = kit.runs.record_case_result(run.id, "a", tfail())
    done = complete(kit, run, "b")
    kit.runs.plan(run.id)
    pending = kit.runs.case_result(run.id, "c")
    sql = (
        "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
        "verdict, detail_json, created_at) VALUES ('x', ?, ?, ?, ?, NULL, '{}', 't')"
    )
    match = "does not fit its case result"
    refuses(store, sql, (failed.id, run.id, EM_KEY, "ok"), match=match)
    refuses(store, sql, (failed.id, run.id, EM_KEY, "not_applicable"), match=match)
    refuses(store, sql, (done.id, run.id, EM_KEY, "skipped"), match=match)
    refuses(store, sql, (pending.id, run.id, EM_KEY, "not_applicable"), match=match)


# --- failure classes, end to end -------------------------------------------------------------

CASE_LEVEL = [
    ("input", "invalid_case"), ("input", "oversize"), ("input", "missing_field"),
    ("target", "exception"), ("target", "timeout"), ("target", "contract_violation"),
    ("target", "blocked"), ("target", "empty_output"),
    ("infrastructure", "storage"), ("infrastructure", "budget_exceeded"),
    ("infrastructure", "cancelled"), ("infrastructure", "auth"),
]  # fmt: skip
EVALUATOR_LEVEL = [
    ("input", "oversize"), ("input", "missing_field"),
    ("evaluator", "invalid_output"), ("evaluator", "truncated"), ("evaluator", "refused"),
    ("evaluator", "bad_request"), ("evaluator", "internal_error"),
    ("infrastructure", "rate_limited"), ("infrastructure", "provider_unavailable"),
    ("infrastructure", "timeout"), ("infrastructure", "connection"),
    ("infrastructure", "quota_exhausted"),
]  # fmt: skip


@pytest.mark.parametrize("cls,kind", CASE_LEVEL)
def test_each_case_level_failure_class_and_kind_round_trips(kit, run, cls, kind):
    f = Failure(failure_class=cls, kind=kind, message=f"{cls} {kind}")
    cr = kit.runs.record_case_result(run.id, "a", CaseOutcome.fail(f))
    got = kit.runs.case_result(run.id, "a")
    assert got.failure == f == cr.failure and got.status == "failed"
    (rec,) = kit.runs.failures(run.id, failure_class=cls)
    assert (rec.scope, rec.case_key, rec.evaluator_key, rec.failure) == ("case", "a", None, f)


@pytest.mark.parametrize("cls,kind", EVALUATOR_LEVEL)
def test_each_evaluator_level_failure_class_and_kind_round_trips(kit, run, cls, kind):
    cr = complete(kit, run, "a")
    f = Failure(failure_class=cls, kind=kind, message=f"{cls} {kind}")
    out = EvaluatorOutcome(evaluator_key=JUDGE_KEY, status="failed", failure=f)
    kit.runs.record_evaluator_result(cr.id, out)
    (got,) = kit.runs.evaluator_results(cr.id)
    assert (got.status, got.failure, got.metrics, got.verdict) == ("failed", f, {}, None)
    (rec,) = kit.runs.failures(run.id, failure_class=cls)
    assert (rec.scope, rec.evaluator_key, rec.failure) == ("evaluator", JUDGE_KEY, f)


def test_retryable_is_recorded_as_given_including_false_on_a_retryable_kind(kit, run):
    f = Failure(failure_class="infrastructure", kind="rate_limited", message="m", retryable=False)
    kit.runs.record_case_result(run.id, "a", CaseOutcome.fail(f))
    assert kit.runs.case_result(run.id, "a").failure.retryable is False
    g = Failure(failure_class="target", kind="timeout", message="m")
    kit.runs.record_case_result(run.id, "b", CaseOutcome.fail(g))
    assert kit.runs.case_result(run.id, "b").failure.retryable is True


def test_an_evaluator_result_cannot_carry_a_target_failure(kit, run):
    cr = complete(kit, run, "a")
    target = Failure(failure_class="target", kind="exception", message="m")
    with pytest.raises(ValidationError, match="evaluator-result failure is one of"):
        EvaluatorOutcome(evaluator_key=EM_KEY, status="failed", failure=target)
    assert kit.runs.evaluator_results(cr.id) == []


def test_the_database_refuses_misplaced_and_incomplete_failure_columns(kit, run, store):
    from conftest import refuses

    cr = complete(kit, run, "a")
    er = (
        "INSERT INTO evaluator_results (id, case_result_id, run_id, evaluator_key, status, "
        "failure_class, failure_kind, failure_message, retryable, created_at) "
        "VALUES ('x', ?, ?, ?, 'failed', ?, ?, ?, ?, 't')"
    )
    args = (cr.id, run.id, EM_KEY)
    refuses(store, er, (*args, "target", "exception", "m", 0))  # target is not an evaluator class
    refuses(store, er, (*args, None, "exception", "m", 0))  # NULL class must not slip through
    refuses(store, er, (*args, "made_up", "x", "m", 0))
    refuses(store, er, (*args, "evaluator", None, "m", 0))
    refuses(store, er, (*args, "evaluator", "refused", None, 0))
    refuses(store, er, (*args, "evaluator", "refused", "m", None))
    cr_sql = (
        "INSERT INTO case_results (id, run_id, case_id, status, failure_class, failure_kind, "
        "failure_message, retryable, finished_at, created_at) "
        "SELECT 'y', ?, c.id, 'failed', ?, ?, ?, ?, 't', 't' FROM cases c WHERE c.case_key = 'b'"
    )
    refuses(store, cr_sql, (run.id, "evaluator", "refused", "m", 0))  # not a case class
    refuses(store, cr_sql, (run.id, None, "exception", "m", 0))
    refuses(store, cr_sql, (run.id, "target", "exception", None, 0))
    refuses(store, cr_sql, (run.id, "target", "exception", "m", None))


def test_failures_can_be_separated_by_class_and_counted(kit, run):
    kit.runs.record_case_result(run.id, "a", tfail("timeout"))
    kit.runs.record_evaluator_result(kit.runs.case_result(run.id, "a").id, skipped(EM_KEY))
    b = complete(kit, run, "b")
    kit.runs.record_evaluator_result(b.id, efail(EM_KEY, "evaluator", "refused"))
    kit.runs.record_evaluator_result(b.id, efail(JUDGE_KEY, "infrastructure", "rate_limited"))
    c = complete(kit, run, "c")
    kit.runs.record_evaluator_result(c.id, ok(EM_KEY, 0.0))  # a *real* zero: quality, no failure

    def classes(name):
        return [
            (f.scope, f.case_key, f.failure.kind)
            for f in kit.runs.failures(run.id, failure_class=name)
        ]

    assert classes("target") == [("case", "a", "timeout")]
    assert classes("evaluator") == [("evaluator", "b", "refused")]
    assert classes("infrastructure") == [("evaluator", "b", "rate_limited")]
    assert classes("input") == []
    assert len(kit.runs.failures(run.id)) == 3
    assert [
        (x.scope, x.failure_class, x.kind, x.count) for x in kit.runs.failure_counts(run.id)
    ] == [
        ("case", "target", "timeout", 1),
        ("evaluator", "evaluator", "refused", 1),
        ("evaluator", "infrastructure", "rate_limited", 1),
    ]
    assert kit.runs.failures(run.id, failure_class="evaluator")[0].failure.retryable is False


def test_failure_queries_validate_their_arguments(kit, run):
    with pytest.raises(RunError, match="unknown failure class"):
        kit.runs.failures(run.id, failure_class="model")
    with pytest.raises(RunError, match="limit"):
        kit.runs.failures(run.id, limit=0)
    assert kit.runs.failures(run.id) == []


def test_a_failure_raised_as_eval_failure_is_recorded_like_a_failure(kit, run):
    try:
        raise EvalFailure("target", "contract_violation", "adapter returned a list")
    except EvalFailure as e:
        kit.runs.record_case_result(run.id, "a", CaseOutcome.fail(e))
    assert kit.runs.case_result(run.id, "a").failure.kind == "contract_violation"


# --- partial failures ------------------------------------------------------------------------


def test_a_run_with_mixed_outcomes_keeps_every_kind_of_result_distinct(kit):
    run = make_run(kit, EM, JUDGE, start=True)
    kit.datasets  # noqa: B018 - clarity: cases a, b, c
    a = complete(kit, run, "a")
    kit.runs.record_evaluator_result(a.id, ok(EM_KEY, 1.0))
    kit.runs.record_evaluator_result(
        a.id, efail(JUDGE_KEY, "infrastructure", "provider_unavailable")
    )
    b = kit.runs.record_case_result(run.id, "b", tfail("blocked", "guardrail"))
    kit.runs.record_evaluator_result(b.id, skipped(EM_KEY))
    kit.runs.record_evaluator_result(b.id, skipped(JUDGE_KEY))
    counts = kit.runs.counts(run.id)
    assert (counts.complete, counts.failed, counts.missing) == (1, 1, 1)
    assert counts.evaluator_results == {
        EM_KEY: {"ok": 1, "skipped": 1},
        JUDGE_KEY: {"failed": 1, "skipped": 1},
    }
    stopped = kit.runs.transition(run.id, "partial", stop_reason="deadline")
    assert stopped.status == "partial"
    assert kit.runs.verify(run.id).ok
    # resume finishes only what is missing; recorded results are untouched
    kit.runs.start(run.id)
    c = complete(kit, run, "c")
    kit.runs.record_evaluator_result(c.id, ok(EM_KEY, 1.0))
    assert kit.runs.transition(run.id, "succeeded").status == "succeeded"
    assert kit.runs.case_result(run.id, "b").failure.kind == "blocked"


def test_a_failed_evaluator_on_one_case_does_not_disturb_other_evaluators_or_cases(kit, run):
    a, b = complete(kit, run, "a"), complete(kit, run, "b")
    kit.runs.record_evaluator_result(a.id, efail(EM_KEY))
    kit.runs.record_evaluator_result(a.id, ok(JUDGE_KEY, 0.8))
    kit.runs.record_evaluator_result(b.id, ok(EM_KEY, 1.0))
    assert {r.evaluator_key: r.status for r in kit.runs.evaluator_results(a.id)} == {
        EM_KEY: "failed",
        JUDGE_KEY: "ok",
    }
    assert [r.status for r in kit.runs.evaluator_results(b.id)] == ["ok"]


def test_case_results_stream_in_case_key_order_filtered_by_status(kit, run):
    kit.runs.plan(run.id)
    complete(kit, run, "c")
    kit.runs.record_case_result(run.id, "a", tfail())
    assert [r.case_key for r in kit.runs.case_results(run.id, status="pending")] == ["b"]
    assert [r.case_key for r in kit.runs.case_results(run.id, status="failed")] == ["a"]
    assert [r.case_key for r in kit.runs.case_results(run.id, batch_size=1)] == ["a", "b", "c"]
    with pytest.raises(RunError, match="unknown case result status"):
        kit.runs.case_results(run.id, status="done")
    with pytest.raises(RunError, match="batch_size"):
        kit.runs.case_results(run.id, batch_size=0)


def test_unknown_case_and_missing_result_lookups(kit, run):
    assert kit.runs.case_result(run.id, "a") is None  # no result yet
    assert kit.runs.case_result(run.id, "not-a-case") is None
