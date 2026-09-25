"""Attempts: every external call is preserved with its evidence (P0 guarantees) and its owner."""

import hashlib
import json
from datetime import timedelta

import pytest
from conftest import EM, JUDGE, complete, ok, refuses, utc_now
from pydantic import ValidationError

from evalkit import (
    CaseOutcome,
    EvalFailure,
    EvaluatorOutcome,
    Failure,
    JudgeOutputError,
    RunAttempt,
    RunError,
)
from evalkit.evaluator import _attempt as p0_attempt
from evalkit.limits import MAX_ERROR_CHARS, MAX_EVIDENCE_BYTES

NOW = utc_now()


def tf(kind="timeout", msg="slow"):
    return Failure(failure_class="target", kind=kind, message=msg)


def ef(cls="evaluator", kind="invalid_output", msg="bad"):
    return Failure(failure_class=cls, kind=kind, message=msg)


def attempts_in(store, column, owner_id):
    rows = store._conn.execute(
        f"SELECT * FROM attempts WHERE {column} = ? ORDER BY n", (owner_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# --- the P0 guarantees, unchanged -------------------------------------------------------------


def test_a_successful_attempt_keeps_only_a_hash_never_the_content():
    a = RunAttempt.succeeded(1, NOW, 12, evidence={"score": 5}, provider="bedrock", model="m")
    assert (a.outcome, a.raw, a.raw_truncated, a.error, a.error_class) == (
        "ok",
        None,
        False,
        None,
        None,
    )
    assert a.raw_sha256 == hashlib.sha256(json.dumps({"score": 5}).encode()).hexdigest()


def test_a_failed_attempt_keeps_the_rejected_output_verbatim_with_its_hash():
    payload = {"correctness": {"score": 9, "reasoning": "r"}}
    a = RunAttempt.failed(1, NOW, 30, ef(), evidence=payload)
    assert json.loads(a.raw) == payload
    assert a.raw_sha256 == hashlib.sha256(a.raw.encode()).hexdigest() and not a.raw_truncated


def test_huge_evidence_is_truncated_on_a_character_boundary_and_flagged():
    text = "é" * (MAX_EVIDENCE_BYTES // 2 + 500)  # 2 bytes each, over the limit
    a = RunAttempt.failed(1, NOW, 1, ef(), evidence=text)
    assert a.raw_truncated and len(a.raw.encode("utf-8")) <= MAX_EVIDENCE_BYTES
    a.raw.encode("utf-8")  # still valid text
    full = json.dumps(text, ensure_ascii=False)
    assert a.raw_sha256 == hashlib.sha256(full.encode()).hexdigest()  # the hash is of everything


@pytest.mark.parametrize(
    "evidence",
    [
        {"a": 1},
        "plain text",
        "\ud800 lone surrogate",
        ["x" * 10],
        {"nested": {"k": [1, 2, {"z": None}]}},
        12345,
        "x" * (MAX_EVIDENCE_BYTES + 10),
        object(),
    ],
    ids=["dict", "text", "surrogate", "list", "nested", "int", "huge", "unserializable"],
)
def test_run_attempts_capture_evidence_exactly_as_the_p0_path_does(evidence):
    """One implementation serves both: the P0 single-record attempt and a run attempt agree on
    kept text, hash and truncation for the same input."""
    exc = JudgeOutputError("bad output", kind="invalid_json", raw=evidence)
    p0 = p0_attempt(1, NOW, 0.01, exc, None)
    run_attempt = RunAttempt.failed(1, NOW, 10, ef(), evidence=evidence)
    assert (run_attempt.raw, run_attempt.raw_sha256, run_attempt.raw_truncated) == (
        p0.raw, p0.raw_sha256, p0.raw_truncated,
    )  # fmt: skip
    ok_p0 = p0_attempt(1, NOW, 0.01, None, evidence)
    ok_run = RunAttempt.succeeded(1, NOW, 10, evidence=evidence)
    assert (ok_run.raw, ok_run.raw_sha256) == (ok_p0.raw, ok_p0.raw_sha256)


def test_error_text_is_scrubbed_clipped_and_made_encodable():
    long = "denied for arn:aws:iam::123456789012:role/x sk-abcdefghijklmnopqrstuvwx " + "y" * 5000
    a = RunAttempt.failed(1, NOW, 1, ef(msg=long))
    assert "123456789012" not in a.error and "sk-abcdef" not in a.error
    assert len(a.error) == MAX_ERROR_CHARS
    direct = RunAttempt(
        n=1, outcome="failed", started_at=NOW, duration_ms=1, error_class="target",
        error_kind="timeout", error="secret=hunter2hunter2 \ud800",
    )  # fmt: skip
    assert "hunter2" not in direct.error
    direct.error.encode("utf-8")


def test_an_eval_failure_donates_its_provider_status_and_request_id():
    exc = EvalFailure(
        "infrastructure", "rate_limited", "429", provider="bedrock", http_status=429,
        request_id="req-9",
    )  # fmt: skip
    a = RunAttempt.failed(2, NOW, 50, exc, model="claude")
    assert (a.provider, a.http_status, a.request_id, a.error_type) == (
        "bedrock", 429, "req-9", "EvalFailure",
    )  # fmt: skip
    assert (a.error_class.value, a.error_kind, a.error, a.model) == (
        "infrastructure", "rate_limited", "429", "claude",
    )  # fmt: skip


# --- model coherence -------------------------------------------------------------------------


def make(**kw):
    base = {"n": 1, "outcome": "ok", "started_at": NOW, "duration_ms": 1}
    return RunAttempt(**(base | kw))


def test_direct_construction_is_held_to_the_same_rules():
    make()
    failed = {"outcome": "failed", "error_class": "target", "error_kind": "timeout"}
    make(**failed)
    bad = [
        ({"n": 0}, "greater than or equal"),
        ({"duration_ms": -1}, "greater than or equal"),
        ({"outcome": "failed"}, "needs error_class and error_kind"),
        ({"error_kind": "timeout"}, "no error"),
        ({"raw": "x", "raw_sha256": "a" * 64}, "no error and no kept evidence"),
        ({**failed, "raw": "x"}, "needs the hash"),
        ({**failed, "raw": "x", "raw_sha256": "zz"}, "64 lowercase hex"),
        ({**failed, "raw": "x" * (MAX_EVIDENCE_BYTES + 1), "raw_sha256": "a" * 64}, "longer than"),
        ({**failed, "raw_truncated": True}, "no evidence is kept"),
        ({**failed, "error_kind": "auth"}, "unknown failure kind"),  # auth is infrastructure
        ({"started_at": NOW.replace(tzinfo=None)}, "timezone"),
        ({"http_status": 99}, "greater than or equal"),
        ({"http_status": 600}, "less than or equal"),
        ({"input_tokens": -1}, "greater than or equal"),
        ({"n": "1"}, "valid integer"),
        ({"surprise": 1}, "Extra inputs"),
    ]
    for fields, match in bad:
        with pytest.raises(ValidationError, match=match):
            make(**fields)


# --- preservation with results ---------------------------------------------------------------


def test_every_attempt_of_a_case_result_is_stored_in_order_with_all_its_fields(kit, run, store):
    t0 = NOW - timedelta(seconds=10)
    attempts = [
        RunAttempt.failed(
            1,
            t0,
            800,
            EvalFailure("target", "timeout", "no reply", provider="http", request_id="r1"),
            evidence={"partial": "..."},
        ),
        RunAttempt.succeeded(
            2,
            t0 + timedelta(seconds=1),
            1500,
            provider="http",
            model="rag-v3",
            http_status=200,
            request_id="r2",
            input_tokens=120,
            output_tokens=45,
        ),
    ]
    cr = kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("answer"), attempts=attempts)
    assert cr.attempts == attempts  # returned as recorded
    assert kit.runs.case_result(run.id, "a").attempts == attempts  # and as read back
    rows = attempts_in(store, "case_result_id", cr.id)
    assert [(r["n"], r["outcome"]) for r in rows] == [(1, "failed"), (2, "ok")]
    assert (rows[1]["input_tokens"], rows[1]["output_tokens"], rows[1]["request_id"]) == (
        120,
        45,
        "r2",
    )
    assert rows[0]["evaluator_result_id"] is None  # owned by the case result alone


def test_a_retry_success_is_distinguishable_from_a_first_try_success(kit, run):
    once = complete(kit, run, "a")
    twice = kit.runs.record_case_result(
        run.id, "b", CaseOutcome.complete("x"),
        attempts=[RunAttempt.failed(1, NOW, 5, tf("timeout")), RunAttempt.succeeded(2, NOW, 5)],
    )  # fmt: skip
    assert (len(once.attempts), len(twice.attempts)) == (0, 2)
    assert [a.outcome for a in twice.attempts] == ["failed", "ok"]


def test_a_failed_case_result_keeps_the_attempts_that_led_to_it(kit, run):
    attempts = [
        RunAttempt.failed(i, NOW, 100 * i, tf("timeout"), evidence=f"try {i}") for i in (1, 2, 3)
    ]
    cr = kit.runs.record_case_result(
        run.id, "a", CaseOutcome.fail(tf("timeout", "gave up after 3")), attempts=attempts
    )
    assert [json.loads(a.raw) for a in cr.attempts] == ["try 1", "try 2", "try 3"]
    assert [a.duration_ms for a in cr.attempts] == [100, 200, 300]


def test_evaluator_attempts_belong_to_the_evaluator_result_not_the_case_result(kit, run, store):
    cr = complete(kit, run, "a")
    calls = [
        RunAttempt.failed(1, NOW, 20, ef(), evidence={"score": 99}, provider="bedrock", model="j"),
        RunAttempt.succeeded(2, NOW, 30, provider="bedrock", model="j", input_tokens=900),
    ]
    er = kit.runs.record_evaluator_result(cr.id, ok(JUDGE.key, 0.9), attempts=calls)
    assert er.attempts == calls
    assert kit.runs.evaluator_results(cr.id)[0].attempts == calls
    assert kit.runs.case_result(run.id, "a").attempts == []  # target attempts stay separate
    rows = attempts_in(store, "evaluator_result_id", er.id)
    assert len(rows) == 2 and all(r["case_result_id"] is None for r in rows)


def test_a_failed_evaluator_result_keeps_its_attempts_too(kit, run):
    cr = complete(kit, run, "a")
    calls = [
        RunAttempt.failed(1, NOW, 10, ef("infrastructure", "rate_limited", "429")),
        RunAttempt.failed(2, NOW, 10, ef("infrastructure", "rate_limited", "429")),
        RunAttempt.failed(3, NOW, 10, ef("infrastructure", "provider_unavailable", "503")),
    ]
    out = EvaluatorOutcome(
        evaluator_key=JUDGE.key,
        status="failed",
        failure=ef("infrastructure", "provider_unavailable", "503"),
    )
    er = kit.runs.record_evaluator_result(cr.id, out, attempts=calls)
    assert [a.error_kind for a in er.attempts] == [
        "rate_limited",
        "rate_limited",
        "provider_unavailable",
    ]


def test_attempts_do_not_need_to_match_the_final_outcome(kit, run):
    """A failed attempt followed by an ok result is the normal retry story."""
    cr = complete(kit, run, "a")
    er = kit.runs.record_evaluator_result(
        cr.id,
        ok(EM.key),
        attempts=[RunAttempt.failed(1, NOW, 1, ef("infrastructure", "timeout", "t"))],
    )
    assert er.status == "ok" and er.attempts[0].outcome == "failed"


def test_a_result_with_no_external_call_has_no_attempts(kit, run):
    cr = complete(kit, run, "a")  # e.g. a deterministic evaluator or a precomputed output
    er = kit.runs.record_evaluator_result(cr.id, ok(EM.key))
    assert cr.attempts == [] and er.attempts == []


# --- numbering, ownership, class fit ---------------------------------------------------------


@pytest.mark.parametrize("numbers", [[2], [1, 3], [2, 1], [1, 1], [0, 1]])
def test_attempts_must_be_numbered_from_one_without_gaps(kit, run, numbers):
    attempts = [RunAttempt.succeeded(n, NOW, 1) if n >= 1 else None for n in numbers]
    if None in attempts:
        with pytest.raises(ValidationError):
            RunAttempt.succeeded(0, NOW, 1)
        return
    with pytest.raises(RunError, match="numbered 1, 2, 3"):
        kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("x"), attempts=attempts)
    assert kit.runs.case_result(run.id, "a") is None  # nothing was stored


def test_a_target_call_cannot_fail_as_the_evaluator_nor_the_reverse(kit, run):
    cr = complete(kit, run, "b")
    with pytest.raises(RunError, match="a target call cannot fail with class 'evaluator'"):
        kit.runs.record_case_result(
            run.id, "a", CaseOutcome.complete("x"), attempts=[RunAttempt.failed(1, NOW, 1, ef())]
        )
    with pytest.raises(RunError, match="an evaluator call cannot fail with class 'target'"):
        kit.runs.record_evaluator_result(
            cr.id, ok(EM.key), attempts=[RunAttempt.failed(1, NOW, 1, tf())]
        )
    assert kit.runs.case_result(run.id, "a") is None and kit.runs.evaluator_results(cr.id) == []


def test_infrastructure_and_input_failures_can_fail_either_kind_of_call(kit, run):
    cr = complete(kit, run, "a")
    oversize = Failure(failure_class="input", kind="oversize", message="m")
    for cls, kind in (("infrastructure", "timeout"), ("input", "oversize")):
        f = Failure(failure_class=cls, kind=kind, message="m")
        kit.runs.record_case_result(
            run.id, "b" if cls == "input" else "c", CaseOutcome.complete("x"),
            attempts=[RunAttempt.failed(1, NOW, 1, f)],
        )  # fmt: skip
    kit.runs.record_evaluator_result(
        cr.id, ok(EM.key),
        attempts=[RunAttempt.failed(1, NOW, 1, oversize)],
    )  # fmt: skip


def test_too_many_attempts_are_refused(kit, run):
    many = [RunAttempt.succeeded(n, NOW, 1) for n in range(1, 102)]
    with pytest.raises(RunError, match="at most 100 attempts"):
        kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("x"), attempts=many)
    exactly = many[:100]
    assert (
        len(
            kit.runs.record_case_result(
                run.id, "a", CaseOutcome.complete("x"), attempts=exactly
            ).attempts
        )
        == 100
    )


def test_the_database_requires_exactly_one_owner_and_a_running_run(kit, run, store):
    cr = complete(kit, run, "a")
    er = kit.runs.record_evaluator_result(cr.id, ok(EM.key))
    base = (
        "INSERT INTO attempts (id, case_result_id, evaluator_result_id, n, started_at, "
        "duration_ms, outcome) VALUES (?, ?, ?, ?, 't', 1, 'ok')"
    )
    refuses(store, base, ("x", None, None, 1))  # no owner
    refuses(store, base, ("x", cr.id, er.id, 1))  # two owners
    refuses(store, base, ("x", "ghost", None, 1))  # unknown owner
    refuses(store, base, ("x", None, "ghost", 1))
    kit.runs.transition(run.id, "partial", stop_reason="x")
    refuses(store, base, ("y", cr.id, None, 7), match="running run")  # runs that ended are closed


def test_the_database_refuses_duplicate_attempt_numbers_per_owner(kit, run, store):
    cr = kit.runs.record_case_result(
        run.id, "a", CaseOutcome.complete("x"), attempts=[RunAttempt.succeeded(1, NOW, 1)]
    )
    refuses(
        store,
        "INSERT INTO attempts (id, case_result_id, n, started_at, duration_ms, outcome) "
        "VALUES ('dup', ?, 1, 't', 1, 'ok')",
        (cr.id,),
        match="UNIQUE",
    )


def test_the_database_refuses_incoherent_attempt_columns(kit, run, store):
    cr = complete(kit, run, "a")
    sql = (
        "INSERT INTO attempts (id, case_result_id, n, started_at, duration_ms, outcome, "
        "error_class, error_kind, raw_payload, raw_sha256, http_status, input_tokens) "
        "VALUES ('x', ?, 1, 't', 1, ?, ?, ?, ?, ?, ?, ?)"
    )
    good = (cr.id, "failed", "target", "timeout", None, None, None, None)
    refuses(store, sql, (cr.id, "failed", None, "timeout", None, None, None, None))  # NULL class
    refuses(store, sql, (cr.id, "failed", "made_up", "timeout", None, None, None, None))
    refuses(store, sql, (cr.id, "failed", "target", None, None, None, None, None))
    refuses(store, sql, (cr.id, "ok", "target", "timeout", None, None, None, None))
    refuses(
        store, sql, (cr.id, "ok", None, None, "raw text", "a" * 64, None, None)
    )  # ok keeps no raw
    refuses(store, sql, (cr.id, "failed", "target", "timeout", "raw", None, None, None))  # no hash
    refuses(store, sql, (cr.id, "failed", "target", "timeout", None, None, 99, None))
    refuses(store, sql, (cr.id, "failed", "target", "timeout", None, None, None, -5))
    refuses(
        store,
        sql,
        (cr.id, "failed", "evaluator", "refused", None, None, None, None),
        match="does not fit",
    )
    er = kit.runs.record_evaluator_result(cr.id, ok(EM.key))
    refuses(
        store,
        "INSERT INTO attempts (id, evaluator_result_id, n, started_at, duration_ms, outcome, "
        "error_class, error_kind) VALUES ('y', ?, 1, 't', 1, 'failed', 'target', 'timeout')",
        (er.id,),
        match="does not fit",
    )
    assert good  # the well-formed shape is exercised by the service tests


def test_attempts_are_write_once_at_the_database(kit, run, store):
    cr = kit.runs.record_case_result(
        run.id,
        "a",
        CaseOutcome.complete("x"),
        attempts=[RunAttempt.failed(1, NOW, 1, tf(), evidence="e")],
    )
    refuses(
        store,
        "UPDATE attempts SET outcome = 'ok', error_class = NULL, error_kind = NULL, "
        "raw_payload = NULL",
        match="write-once",
    )
    refuses(store, "UPDATE attempts SET duration_ms = 0", match="write-once")
    refuses(store, "DELETE FROM attempts", match="permanent")
    assert kit.runs.case_result(run.id, "a").attempts[0].raw == json.dumps("e")
    assert cr.attempts[0].n == 1


# --- atomicity -------------------------------------------------------------------------------


def test_a_failing_attempt_rolls_the_whole_case_result_back(kit, run, store, monkeypatch):
    kit.runs.plan(run.id)
    good = RunAttempt.succeeded(1, NOW, 1)
    dup_id = RunAttempt.succeeded(2, NOW, 1).model_copy(update={"id": good.id})  # PK collision
    with pytest.raises(RunError):
        kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("x"), attempts=[good, dup_id])
    got = kit.runs.case_result(run.id, "a")
    assert got.status == "pending" and got.output is None and got.attempts == []
    assert store._conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert not store._conn.in_transaction  # and the store is usable afterwards
    complete(kit, run, "a")


def test_a_failing_attempt_rolls_back_an_evaluator_result_and_its_metrics(kit, run, store):
    cr = complete(kit, run, "a")
    a = RunAttempt.succeeded(1, NOW, 1)
    clash = RunAttempt.succeeded(2, NOW, 1).model_copy(update={"id": a.id})
    out = EvaluatorOutcome(
        evaluator_key=JUDGE.key, status="ok", verdict="PASS", metrics={"score": 1, "c.x": 2}
    )
    with pytest.raises(RunError):
        kit.runs.record_evaluator_result(cr.id, out, attempts=[a, clash])
    conn = store._conn
    assert conn.execute("SELECT COUNT(*) FROM evaluator_results").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 0
    assert kit.runs.evaluator_results(cr.id) == []
    kit.runs.record_evaluator_result(cr.id, out, attempts=[a])  # the same call now works


def test_an_interrupted_write_leaves_no_partial_state(kit, run, store, monkeypatch):
    class Boom(Exception):
        pass

    def explode(*args, **kwargs):
        raise Boom

    monkeypatch.setattr(type(store), "_insert_attempts", explode)
    with pytest.raises(Boom):
        kit.runs.record_case_result(
            run.id, "a", CaseOutcome.complete("x"), attempts=[RunAttempt.succeeded(1, NOW, 1)]
        )
    monkeypatch.undo()
    assert kit.runs.case_result(run.id, "a") is None
    assert not store._conn.in_transaction
    assert complete(kit, run, "a").status == "complete"


def test_a_run_that_ended_meanwhile_rejects_late_results_atomically(kit, run):
    kit.runs.transition(run.id, "cancelled", stop_reason="user")
    with pytest.raises(RunError, match="results need a running run"):
        kit.runs.record_case_result(
            run.id, "a", CaseOutcome.complete("x"), attempts=[RunAttempt.succeeded(1, NOW, 1)]
        )
    assert kit.runs.counts(run.id).complete == 0


def test_circular_evidence_is_still_captured_as_text():
    loop: list = []
    loop.append(loop)
    a = RunAttempt.failed(1, NOW, 1, ef(), evidence=loop)
    assert a.raw and a.raw_sha256 and not a.raw_truncated
