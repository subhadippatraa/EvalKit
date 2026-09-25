"""F-13: a judged (paid-for) result is never silently lost when persistence fails."""

import os
import sqlite3
import stat

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import ConfigError, Evaluator, JudgeError, StoreError
from evalkit.cli import main
from evalkit.store import SQLiteStore


class DiskFull(sqlite3.OperationalError):
    pass


def break_saves(monkeypatch, store, exc=None):
    real = store.save
    state = {"broken": True}

    def save(result):
        if state["broken"]:
            raise exc or DiskFull("database or disk is full")
        return real(result)

    monkeypatch.setattr(store, "save", save)
    return state


def evaluate(evaluator):
    return evaluator.evaluate("Explain DI.", "DI is ...", criteria=CRITERIA)


def test_a_failed_save_raises_store_error_carrying_the_complete_result(store, monkeypatch):
    break_saves(monkeypatch, store)
    with pytest.raises(StoreError) as info:
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=4)), store))
    e = info.value
    assert (
        e.result.status == "ok" and e.result.overall_score == 0.875 and e.result.verdict == "PASS"
    )
    assert e.evaluation_id == e.result.id
    assert isinstance(e.__cause__, DiskFull)  # the underlying error is chained, not swallowed
    assert e.result.id in str(e) and "disk is full" in str(e)


def test_store_error_is_still_an_sqlite3_error_for_existing_callers(store, monkeypatch):
    break_saves(monkeypatch, store)
    with pytest.raises(sqlite3.Error):  # pre-P0 contract: "sqlite3.Error propagates"
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store))


def test_the_result_is_spilled_to_disk_and_recoverable(store, monkeypatch):
    state = break_saves(monkeypatch, store)
    ev = Evaluator(FakeJudge(judged(correctness=5, clarity=4)), store)
    with pytest.raises(StoreError) as info:
        evaluate(ev)
    lost = info.value.result
    spill = info.value.spill_path
    assert spill.exists() and spill.parent == store.spill_dir and store.get(lost.id) is None
    if os.name == "posix":
        assert stat.S_IMODE(spill.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.spill_dir.stat().st_mode) == 0o700

    state["broken"] = False  # the database is healthy again
    report = ev.recover_spilled()
    assert report.recovered == [lost.id] and report.failed == {}
    assert store.get(lost.id) == lost  # exactly what the judge produced, attempts included
    assert not spill.exists()
    assert ev.recover_spilled().recovered == []  # idempotent


def test_recovering_a_result_that_was_in_fact_saved_is_harmless(store, monkeypatch):
    state = break_saves(monkeypatch, store)
    ev = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store)
    with pytest.raises(StoreError) as info:
        evaluate(ev)
    state["broken"] = False
    store.save(info.value.result)  # e.g. a caller retried info.value.result themselves
    assert ev.recover_spilled().recovered == [info.value.result.id]
    assert len(store.list()) == 1 and not info.value.spill_path.exists()


def test_a_judge_failure_that_cannot_be_saved_keeps_both_facts(store, monkeypatch):
    state = break_saves(monkeypatch, store)
    ev = Evaluator(FakeJudge(JudgeError("Bedrock ThrottlingException: slow down")), store)
    with pytest.raises(StoreError) as info:
        evaluate(ev)
    e = info.value
    assert e.result.status == "error" and "ThrottlingException" in e.result.error
    assert isinstance(e.judge_error, JudgeError) and "judge failed" in str(e)
    state["broken"] = False
    ev.recover_spilled()
    assert store.get(e.result.id).status == "error"


def test_unreadable_spill_files_are_reported_and_kept_while_others_recover(store, monkeypatch):
    state = break_saves(monkeypatch, store)
    ev = Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store)
    with pytest.raises(StoreError) as info:
        evaluate(ev)
    (store.spill_dir / "corrupt.json").write_text("{not json")
    state["broken"] = False

    report = ev.recover_spilled()
    assert report.recovered == [info.value.result.id]
    assert list(report.failed) == ["corrupt.json"] and (store.spill_dir / "corrupt.json").exists()


def test_a_hostile_result_id_cannot_escape_the_spill_directory(store):
    from evalkit.models import CriterionScore, EvaluationResult, Rubric

    r = Rubric.from_dict({"a": "A?"})
    result = EvaluationResult(
        id="../../escape", status="ok", prompt="p", model_output="o", rubric=r,
        rubric_version=r.version, judge_provider="f", judge_model="m", judge_temperature=0.0,
        judge_prompt_version="v", scores={"a": CriterionScore(reasoning="r", score=4)},
        overall_score=0.75, verdict="PASS",
    )  # fmt: skip
    path = store.spill_result(result)
    assert path.parent == store.spill_dir and ".." not in path.name


def test_when_the_spill_itself_fails_the_result_is_still_on_the_exception(store, monkeypatch):
    break_saves(monkeypatch, store)
    monkeypatch.setattr(
        store, "spill_result", lambda result: (_ for _ in ()).throw(OSError("read-only fs"))
    )
    with pytest.raises(StoreError) as info:
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store))
    assert info.value.spill_path is None and info.value.result.status == "ok"
    assert "writing the recovery file also failed" in str(info.value)


def test_a_custom_store_without_spill_support_still_returns_the_result_on_the_error():
    class BrokenStore:
        def save(self, result):
            raise RuntimeError("backend down")

        def get(self, i):
            return None

        def list(self, tag=None, limit=20):
            return []

        def add_review(self, review):
            pass

    with pytest.raises(StoreError) as info:
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=5)), BrokenStore()))
    assert info.value.spill_path is None and info.value.result.status == "ok"
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(ConfigError, match="spill recovery"):
        Evaluator(None, BrokenStore()).recover_spilled()


def test_an_in_memory_store_has_nowhere_to_spill_but_still_reports_the_result(monkeypatch):
    store = SQLiteStore(":memory:")
    break_saves(monkeypatch, store)
    with pytest.raises(StoreError) as info:
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store))
    assert info.value.spill_path is None and info.value.result.status == "ok"
    assert store.recover_spilled().recovered == []


def test_a_rubric_registration_failure_happens_before_any_paid_call(store, monkeypatch):
    monkeypatch.setattr(
        store, "register_rubric", lambda *a: (_ for _ in ()).throw(DiskFull("full"))
    )
    judge = FakeJudge(judged(correctness=5, clarity=5))
    with pytest.raises(sqlite3.OperationalError):
        evaluate(Evaluator(judge, store))
    assert judge.calls == []


def test_opening_a_store_with_waiting_spill_files_warns(store, monkeypatch, caplog):
    import logging

    break_saves(monkeypatch, store)
    with pytest.raises(StoreError):
        evaluate(Evaluator(FakeJudge(judged(correctness=5, clarity=5)), store))
    path = store.path
    with caplog.at_level(logging.WARNING, logger="evalkit.store"):
        SQLiteStore(path).close()
    assert "unsaved evaluation results" in caplog.text


def test_cli_reports_the_evaluation_id_and_spill_path_and_can_recover(
    tmp_path, monkeypatch, capsys
):
    import json

    db = tmp_path / "cli.db"
    store = SQLiteStore(db)
    state = break_saves(monkeypatch, store)
    judge = FakeJudge(judged(correctness=5, clarity=5))
    monkeypatch.setattr(
        Evaluator,
        "from_env",
        classmethod(lambda cls, with_judge=True: cls(judge if with_judge else None, store)),
    )
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"prompt": "p", "model_output": "o", "criteria": CRITERIA}))

    assert main(["run", "--input", str(inp)]) == 1
    err = capsys.readouterr().err
    assert "could not be saved" in err and "evaluation_id:" in err and "spilled_to:" in err
    evaluation_id = err.split("evaluation_id: ")[1].split()[0]

    state["broken"] = False
    assert main(["recover"]) == 0
    assert json.loads(capsys.readouterr().out) == {"recovered": [evaluation_id], "failed": {}}
    assert store.get(evaluation_id).status == "ok"
    assert main(["recover"]) == 0
    capsys.readouterr()

    (store.spill_dir).mkdir(exist_ok=True)
    (store.spill_dir / "bad.json").write_text("nope")
    assert main(["recover"]) == 1  # failures are a non-zero exit


def test_cli_reports_other_database_errors_cleanly(monkeypatch, store, capsys):
    monkeypatch.setattr(
        Evaluator, "from_env", classmethod(lambda cls, with_judge=True: cls(None, store))
    )
    monkeypatch.setattr(
        store, "get", lambda i: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error"))
    )
    assert main(["get", "x"]) == 1
    assert "database error: disk I/O error" in capsys.readouterr().err
