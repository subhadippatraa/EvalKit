import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import (
    ConfigError,
    Criterion,
    EvalKitError,
    Evaluator,
    JudgeError,
    JudgeOutputError,
    JudgeTimeoutError,
    Rubric,
    RubricError,
)


def run(evaluator, **overrides):
    kwargs = dict(prompt="Explain DI.", model_output="DI is ...", criteria=CRITERIA)
    return evaluator.evaluate(**(kwargs | overrides))


def only_row(store):
    (row,) = store.list()
    return store.get(row.id)


# --- success ------------------------------------------------------------------


def test_success_is_scored_and_persisted(store):
    judge = FakeJudge(judged(correctness=5, clarity=3))
    result = run(
        Evaluator(judge, store),
        reference_output="gold",
        metadata={"app": "ragforge", "model": "gpt-x"},
        tags=["regression", "regression", "nightly"],
    )
    assert result.status == "ok" and result.error is None
    assert result.overall_score == 0.75 and result.verdict == "PASS"
    assert result.judge_provider == "fake" and result.judge_prompt_version == "fakeversion1"
    assert result.rubric_version == result.rubric.version
    assert result.tags == ["regression", "nightly"]
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert judge.calls[0][2] == "gold"
    assert store.get(result.id) == result


def test_explicit_rubric_object_and_dict(store):
    rubric = Rubric(
        criteria=[Criterion(name="acc", description="Accurate?", scale=(0, 10), weight=2)],
        threshold=0.9,
        version="acc-v1",
    )
    r1 = Evaluator(FakeJudge(judged(acc=8)), store).evaluate("p", "o", rubric=rubric)
    assert (r1.overall_score, r1.verdict, r1.rubric_version) == (0.8, "FAIL", "acc-v1")
    r2 = Evaluator(FakeJudge(judged(acc=10)), store).evaluate("p", "o", rubric=rubric.model_dump())
    assert r2.verdict == "PASS" and r2.rubric == rubric


def test_works_without_store():
    result = run(Evaluator(FakeJudge(judged(correctness=1, clarity=1))))
    assert result.verdict == "FAIL"


# --- invalid input: RubricError, judge not called, nothing persisted -----------


@pytest.mark.parametrize(
    "overrides",
    [
        {"criteria": None},  # neither criteria nor rubric
        {"rubric": Rubric.from_dict({"a": "A"})},  # both
        {"criteria": {}},
        {"criteria": ["correctness"]},
        {"criteria": {"a": ""}},
        {
            "criteria": None,
            "rubric": {"criteria": [{"name": "a", "description": "A", "weight": 0}]},
        },
        {"prompt": None},
        {"model_output": 42},
        {"metadata": {"bad": object()}},
        {"tags": "not-a-list"},
    ],
)
def test_invalid_input(store, overrides):
    judge = FakeJudge()
    with pytest.raises(RubricError):
        run(Evaluator(judge, store), **overrides)
    assert judge.calls == []
    assert store.list() == []


# --- retry semantics ------------------------------------------------------------


def test_malformed_then_valid_retries_once(store):
    judge = FakeJudge(judged(correctness=9, clarity=3), judged(correctness=5, clarity=5))
    result = run(Evaluator(judge, store))
    assert len(judge.calls) == 2
    assert result.status == "ok" and result.overall_score == 1.0
    assert only_row(store).status == "ok"


def test_missing_payload_then_valid_retries_once(store):
    judge = FakeJudge(JudgeOutputError("no tool call"), judged(correctness=5, clarity=5))
    assert run(Evaluator(judge, store)).status == "ok"
    assert len(judge.calls) == 2


def test_malformed_twice_raises_and_persists_error(store):
    judge = FakeJudge(judged(correctness=4), judged(correctness=4, clarity=0))
    with pytest.raises(JudgeOutputError) as info:
        run(Evaluator(judge, store))
    assert len(judge.calls) == 2  # never more than one retry
    row = only_row(store)
    assert info.value.evaluation_id == row.id
    assert row.status == "error" and row.error.startswith("JudgeOutputError:")
    assert row.scores == {} and row.overall_score is None and row.verdict is None
    assert row.judge_model == "fake-model" and row.prompt == "Explain DI."


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (JudgeError("Bedrock ThrottlingException: slow down"), JudgeError),
        (JudgeTimeoutError("timed out"), JudgeTimeoutError),
    ],
)
def test_provider_errors_are_not_retried_and_are_persisted(store, exc, expected):
    judge = FakeJudge(exc, judged(correctness=5, clarity=5))
    with pytest.raises(expected) as info:
        run(Evaluator(judge, store))
    assert len(judge.calls) == 1
    row = only_row(store)
    assert row.id == info.value.evaluation_id
    assert row.status == "error" and expected.__name__ in row.error


def test_provider_error_during_retry_is_not_retried(store):
    judge = FakeJudge(judged(correctness=0, clarity=0), JudgeTimeoutError("timed out"))
    with pytest.raises(JudgeTimeoutError):
        run(Evaluator(judge, store))
    assert len(judge.calls) == 2
    assert only_row(store).error.startswith("JudgeTimeoutError:")


def test_unexpected_judge_exception_is_wrapped_and_persisted(store):
    judge = FakeJudge(RuntimeError("bug"))
    with pytest.raises(JudgeError, match="RuntimeError: bug") as info:
        run(Evaluator(judge, store))
    assert isinstance(info.value.__cause__, RuntimeError)
    assert only_row(store).status == "error"


def test_error_without_store_has_no_evaluation_id():
    with pytest.raises(JudgeError) as info:
        run(Evaluator(FakeJudge(JudgeError("down"))))
    assert info.value.evaluation_id is None


# --- human review ---------------------------------------------------------------


def test_multiple_reviews(store):
    evaluator = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store)
    result = run(evaluator)
    r1 = evaluator.review(result.id, "alice", "PASS", score=0.9, comment="good")
    r2 = evaluator.review(result.id, "bob", "FAIL", score=0.4)
    r3 = evaluator.review(result.id, "carol", "PASS")
    got = evaluator.get(result.id)
    assert got.reviews == [r1, r2, r3]
    assert got.reviews[0].score == 0.9 and got.reviews[2].score is None


def test_review_error_row(store):
    evaluator = Evaluator(FakeJudge(JudgeError("down")), store)
    with pytest.raises(JudgeError) as info:
        run(evaluator)
    evaluator.review(info.value.evaluation_id, "alice", "FAIL", comment="provider outage")
    assert len(evaluator.get(info.value.evaluation_id).reviews) == 1


def test_review_validation(store):
    evaluator = Evaluator(FakeJudge(judged(correctness=4, clarity=4)), store)
    result = run(evaluator)
    with pytest.raises(EvalKitError, match="not found"):
        evaluator.review("missing", "alice", "PASS")
    for bad in [
        dict(reviewer="", verdict="PASS"),
        dict(reviewer="a", verdict="pass?"),
        dict(reviewer="a", verdict="PASS", score=1.5),
    ]:
        with pytest.raises(EvalKitError, match="invalid review"):
            evaluator.review(result.id, **bad)
    assert evaluator.get(result.id).reviews == []


def test_store_required_for_review_get_list():
    evaluator = Evaluator(FakeJudge())
    for call in (
        lambda: evaluator.review("x", "a", "PASS"),
        lambda: evaluator.get("x"),
        lambda: evaluator.list(),
    ):
        with pytest.raises(ConfigError):
            call()


def test_get_missing(store):
    with pytest.raises(EvalKitError, match="not found"):
        Evaluator(None, store).get("missing")


def test_evaluate_requires_judge(store):
    with pytest.raises(ConfigError):
        run(Evaluator(None, store))


# --- from_env ---------------------------------------------------------------------


@pytest.fixture
def env(monkeypatch, tmp_path):
    for name in [
        "EVALKIT_JUDGE_PROVIDER",
        "EVALKIT_JUDGE_MODEL",
        "EVALKIT_JUDGE_TEMPERATURE",
        "EVALKIT_JUDGE_TIMEOUT",
        "EVALKIT_JUDGE_API_KEY",
        "EVALKIT_JUDGE_BASE_URL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "env.db"))
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    return monkeypatch


def test_from_env_builds_bedrock_judge_and_sqlite_store(env, tmp_path):
    from evalkit.bedrock import BedrockJudge
    from evalkit.store import SQLiteStore

    env.setenv("EVALKIT_JUDGE_MODEL", "anthropic.some-model")
    env.setenv("EVALKIT_JUDGE_TEMPERATURE", "0.3")
    env.setenv("EVALKIT_JUDGE_TIMEOUT", "15")
    evaluator = Evaluator.from_env()
    assert isinstance(evaluator.judge, BedrockJudge)
    assert (evaluator.judge.model, evaluator.judge.temperature, evaluator.judge.timeout) == (
        "anthropic.some-model",
        0.3,
        15.0,
    )
    assert isinstance(evaluator.store, SQLiteStore)
    assert (tmp_path / "env.db").exists()


def test_from_env_requires_model(env):
    with pytest.raises(ConfigError, match="EVALKIT_JUDGE_MODEL"):
        Evaluator.from_env()


def test_from_env_rejects_unknown_provider(env):
    env.setenv("EVALKIT_JUDGE_MODEL", "m")
    env.setenv("EVALKIT_JUDGE_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="openai"):
        Evaluator.from_env()


def test_from_env_rejects_bad_number(env):
    env.setenv("EVALKIT_JUDGE_MODEL", "m")
    env.setenv("EVALKIT_JUDGE_TIMEOUT", "soon")
    with pytest.raises(ConfigError, match="EVALKIT_JUDGE_TIMEOUT"):
        Evaluator.from_env()


def test_from_env_store_only_needs_no_judge_config(env):
    evaluator = Evaluator.from_env(with_judge=False)
    assert evaluator.judge is None and evaluator.list() == []


def test_from_env_builds_bedrock_openai_judge(env):
    from evalkit.bedrock_openai import DEFAULT_BASE_URL, BedrockOpenAIJudge

    env.setenv("EVALKIT_JUDGE_PROVIDER", "bedrock-openai")
    env.setenv("EVALKIT_JUDGE_MODEL", "openai.gpt-oss-120b")
    env.setenv("EVALKIT_JUDGE_API_KEY", "test-key")
    evaluator = Evaluator.from_env()
    assert isinstance(evaluator.judge, BedrockOpenAIJudge)
    assert evaluator.judge.model == "openai.gpt-oss-120b"
    assert str(evaluator.judge._client.base_url).rstrip("/") == DEFAULT_BASE_URL.rstrip("/")


def test_from_env_bedrock_openai_base_url_override(env):
    env.setenv("EVALKIT_JUDGE_PROVIDER", "bedrock-openai")
    env.setenv("EVALKIT_JUDGE_MODEL", "m")
    env.setenv("EVALKIT_JUDGE_API_KEY", "test-key")
    env.setenv("EVALKIT_JUDGE_BASE_URL", "https://example.test/v1")
    evaluator = Evaluator.from_env()
    assert str(evaluator.judge._client.base_url).rstrip("/") == "https://example.test/v1"


def test_from_env_bedrock_openai_requires_api_key(env):
    env.setenv("EVALKIT_JUDGE_PROVIDER", "bedrock-openai")
    env.setenv("EVALKIT_JUDGE_MODEL", "m")
    with pytest.raises(ConfigError, match="EVALKIT_JUDGE_API_KEY"):
        Evaluator.from_env()
