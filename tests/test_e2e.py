"""End-to-end: input -> BedrockJudge (stubbed) -> result -> persisted -> retrieved -> review."""

from test_bedrock import FakeBedrockClient, converse_response

from evalkit import Evaluator
from evalkit.bedrock import BedrockJudge
from evalkit.judge import PROMPT_VERSION
from evalkit.store import SQLiteStore


def test_end_to_end(tmp_path):
    db = tmp_path / "evalkit.db"
    client = FakeBedrockClient(
        converse_response(
            {
                "correctness": {"reasoning": "Accurate description of DI.", "score": 5},
                "clarity": {"reasoning": "Clear but dense.", "score": 4},
            }
        )
    )
    evaluator = Evaluator(BedrockJudge("my-model", client=client), SQLiteStore(db))

    result = evaluator.evaluate(
        prompt="Explain dependency injection in .NET.",
        model_output="DI in .NET uses IServiceCollection ...",
        reference_output="Constructor injection via the built-in container ...",
        criteria={"correctness": "Is it factually correct?", "clarity": "Is it clear?"},
        metadata={"app": "ragforge", "app_version": "1.4.2", "model": "gpt-x"},
        tags=["regression-suite"],
    )
    assert result.status == "ok" and result.verdict == "PASS"
    assert result.overall_score == 0.875  # (1.0 + 0.75) / 2
    assert result.judge_provider == "bedrock" and result.judge_model == "my-model"
    assert result.judge_prompt_version == PROMPT_VERSION

    # a fresh process/connection sees the persisted result
    reopened = Evaluator(None, SQLiteStore(db))
    stored = reopened.get(result.id)
    assert stored == result
    assert stored.scores["clarity"].reasoning == "Clear but dense."
    assert [r.id for r in reopened.list(tag="regression-suite")] == [result.id]

    reopened.review(result.id, "alice", "PASS", score=0.9, comment="agree")
    reopened.review(result.id, "bob", "FAIL", score=0.5, comment="too dense")
    reviews = reopened.get(result.id).reviews
    assert [(r.reviewer, r.verdict, r.score) for r in reviews] == [
        ("alice", "PASS", 0.9),
        ("bob", "FAIL", 0.5),
    ]
