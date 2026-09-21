from __future__ import annotations

import os
import time
from typing import Any

from pydantic import ValidationError

from evalkit.errors import ConfigError, EvalKitError, JudgeError, JudgeOutputError, RubricError
from evalkit.judge import Judge
from evalkit.models import EvaluationResult, Review, Rubric, format_validation_error
from evalkit.store import Store


class Evaluator:
    def __init__(self, judge: Judge | None, store: Store | None = None):
        self.judge = judge
        self.store = store

    @classmethod
    def from_env(cls, *, with_judge: bool = True) -> Evaluator:
        """Judge (from EVALKIT_JUDGE_PROVIDER) + SQLiteStore from EVALKIT_* env vars.

        with_judge=False builds a store-only evaluator (get/list/review) that needs no
        judge/AWS configuration.
        """
        from evalkit.store import SQLiteStore

        judge = None
        if with_judge:
            provider = os.environ.get("EVALKIT_JUDGE_PROVIDER") or "bedrock"
            model = os.environ.get("EVALKIT_JUDGE_MODEL")
            if not model:
                raise ConfigError("EVALKIT_JUDGE_MODEL is required")
            temperature = _float_env("EVALKIT_JUDGE_TEMPERATURE", 0.0)
            timeout = _float_env("EVALKIT_JUDGE_TIMEOUT", 60.0)

            if provider == "bedrock":
                from evalkit.bedrock import BedrockJudge  # keep boto3 out of the core import path

                judge = BedrockJudge(model=model, temperature=temperature, timeout=timeout)
            elif provider == "bedrock-openai":
                # Bedrock's OpenAI-compatible gateway; keep the `openai` SDK out of the core
                # import path the same way boto3 is kept out for 'bedrock'.
                from evalkit.bedrock_openai import DEFAULT_BASE_URL, BedrockOpenAIJudge

                api_key = os.environ.get("EVALKIT_JUDGE_API_KEY")
                if not api_key:
                    raise ConfigError(
                        "EVALKIT_JUDGE_API_KEY is required for provider 'bedrock-openai'"
                    )
                judge = BedrockOpenAIJudge(
                    model=model,
                    api_key=api_key,
                    base_url=os.environ.get("EVALKIT_JUDGE_BASE_URL") or DEFAULT_BASE_URL,
                    temperature=temperature,
                    timeout=timeout,
                )
            else:
                raise ConfigError(
                    f"unsupported EVALKIT_JUDGE_PROVIDER {provider!r}; "
                    "supported: 'bedrock', 'bedrock-openai'"
                )
        return cls(judge, SQLiteStore(os.environ.get("EVALKIT_DB_PATH") or "./evalkit.db"))

    def evaluate(
        self,
        prompt: str,
        model_output: str,
        reference_output: str | None = None,
        context: str | None = None,
        criteria: dict[str, str] | None = None,
        rubric: Rubric | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> EvaluationResult:
        if self.judge is None:
            raise ConfigError("no judge configured")
        judge = self.judge

        # 1. validate inputs before the judge; invalid input is never persisted
        try:
            if (criteria is None) == (rubric is None):
                raise RubricError("pass exactly one of `criteria` or `rubric`")
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise RubricError("`criteria` must be a dict of {name: description}")
                rubric = Rubric.from_dict(criteria)
            elif not isinstance(rubric, Rubric):
                rubric = Rubric.model_validate(rubric)
            result = EvaluationResult(
                status="ok",
                prompt=prompt,
                model_output=model_output,
                reference_output=reference_output,
                context=context,
                rubric=rubric,
                rubric_version=rubric.version,
                judge_provider=judge.provider,
                judge_model=judge.model,
                judge_temperature=judge.temperature,
                judge_prompt_version=judge.prompt_version,
                metadata=metadata or {},
                tags=tags or [],
            )
        except ValidationError as e:
            raise RubricError(f"invalid input: {format_validation_error(e)}") from e

        # 2. judge + score; retry once only on malformed output
        error: JudgeError | None = None
        start = time.perf_counter()
        for _ in range(2):
            try:
                raw = judge.judge(prompt, model_output, reference_output, context, rubric)
                result.scores, result.overall_score, result.verdict = rubric.score(raw)
                error = None
                break
            except JudgeOutputError as e:
                error = e
            except JudgeError as e:
                error = e
                break
            except Exception as e:  # a buggy custom judge must still produce an error row
                error = JudgeError(f"judge raised {type(e).__name__}: {e}")
                error.__cause__ = e
                break
        result.latency_ms = round((time.perf_counter() - start) * 1000)

        if error is not None:
            result.status = "error"
            result.error = f"{type(error).__name__}: {error}"

        # 3. persist ok and judge-stage error rows alike
        if self.store is not None:
            try:
                self.store.save(result)
            except Exception as store_exc:
                # sqlite3.Error still propagates as the raised exception, per architecture;
                # chain to the judge failure that triggered it (if any) so it isn't lost
                raise store_exc from error
        if error is not None:
            error.evaluation_id = result.id if self.store is not None else None
            raise error
        return result

    def review(
        self,
        evaluation_id: str,
        reviewer: str,
        verdict: str,
        score: float | None = None,
        comment: str | None = None,
    ) -> Review:
        store = self._require_store()
        try:
            review = Review(
                evaluation_id=evaluation_id,
                reviewer=reviewer,
                verdict=verdict,
                score=score,
                comment=comment,
            )
        except ValidationError as e:
            raise EvalKitError(f"invalid review: {format_validation_error(e)}") from e
        if store.get(evaluation_id) is None:
            raise EvalKitError(f"evaluation {evaluation_id!r} not found")
        store.add_review(review)
        return review

    def get(self, evaluation_id: str) -> EvaluationResult:
        result = self._require_store().get(evaluation_id)
        if result is None:
            raise EvalKitError(f"evaluation {evaluation_id!r} not found")
        return result

    def list(self, tag: str | None = None, limit: int = 20) -> list[EvaluationResult]:
        return self._require_store().list(tag=tag, limit=limit)

    def _require_store(self) -> Store:
        if self.store is None:
            raise ConfigError("no store configured")
        return self.store


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {value!r}") from e
