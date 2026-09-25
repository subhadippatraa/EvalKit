import pytest

from evalkit.store import SQLiteStore

CRITERIA = {"correctness": "Is it factually correct?", "clarity": "Is it clear?"}


def judged(**scores: int) -> dict:
    """Raw judge payload for the given criterion scores."""
    return {name: {"reasoning": f"{name} looks fine", "score": s} for name, s in scores.items()}


class FakeJudge:
    """Test Judge: returns scripted raw payloads or raises scripted exceptions, in order."""

    provider = "fake"
    model = "fake-model"
    temperature = 0.0
    prompt_version = "fakeversion1"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def judge(self, prompt, model_output, reference_output, context, rubric):
        self.calls.append((prompt, model_output, reference_output, context, rubric))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def store(tmp_path):
    s = SQLiteStore(tmp_path / "evalkit.db")
    yield s
    s.close()


@pytest.fixture
def kit(store):
    from evalkit import EvalKit

    return EvalKit(store)


def case(key="c1", **fields):
    """A valid dataset case dict; override or add fields."""
    return {"case_key": key, "prompt": f"prompt for {key}"} | fields


# --- runs ------------------------------------------------------------------------------------

RUN_CASES = ["a", "b", "c"]


def utc_now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


def spec(kind="exact_match", name="em", **params):
    from evalkit import EvaluatorSpec

    return EvaluatorSpec(kind=kind, name=name, params=params)


EM = spec()
JUDGE = spec("llm_judge", "quality", model="judge-1", rubric="r1")


def make_run(kit, *evaluators, ref="qa", start=False, **kw):
    """A run over dataset `ref` (imported on first use) with the given evaluator specs."""
    from evalkit import RunConfig

    if not any(d.name == ref.split("@")[0] for d in kit.datasets.list()):
        kit.datasets.import_cases(
            ref.split("@")[0], [case(k, output=f"out-{k}") for k in RUN_CASES]
        )
    config = kw.pop("config", None) or RunConfig(evaluators=list(evaluators))
    run = kit.runs.create(ref, config, **kw)
    return kit.runs.start(run.id) if start else run


@pytest.fixture
def run(kit):
    """A *running* run over dataset qa (cases a, b, c) with two evaluators (EM, JUDGE)."""
    return make_run(kit, EM, JUDGE, start=True)


def ok(key, score=1.0, **kw):
    from evalkit import EvaluatorOutcome

    return EvaluatorOutcome(
        evaluator_key=key, status="ok", verdict="PASS", metrics={"score": score}, **kw
    )


def complete(kit, run, case_key="a", output="out"):
    """Record a complete case result; returns it."""
    from evalkit import CaseOutcome

    return kit.runs.record_case_result(run.id, case_key, CaseOutcome.complete(output))


def refuses(store, sql, params=(), match=None):
    """Assert the database itself rejects `sql` (CHECK / trigger / FK), leaving nothing open."""
    import sqlite3

    conn = store._conn
    try:
        with pytest.raises(sqlite3.IntegrityError, match=match):
            conn.execute(sql, params)
    finally:
        if conn.in_transaction:
            conn.rollback()
