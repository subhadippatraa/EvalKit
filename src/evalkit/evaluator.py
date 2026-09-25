from __future__ import annotations

import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalkit.errors import (
    ConfigError,
    EvalKitError,
    JudgeError,
    JudgeOutputError,
    JudgeTimeoutError,
    RubricError,
    ScoringError,
    StoreError,
)
from evalkit.evidence import capture_evidence, clip
from evalkit.judge import Judge
from evalkit.limits import (
    MAX_ERROR_CHARS,
    MAX_TAG_CHARS,
    Limits,
)
from evalkit.models import (
    Attempt,
    AttemptOutcome,
    EvaluationResult,
    Review,
    Rubric,
    format_validation_error,
)
from evalkit.store import Store

UNEXPECTED = "unexpected_error"  # JudgeError.kind for a non-evalkit exception raised by a judge


class Evaluator:
    def __init__(
        self, judge: Judge | None, store: Store | None = None, limits: Limits | None = None
    ):
        self.judge = judge
        self.store = store
        self.limits = limits or Limits()

    @classmethod
    def from_env(cls, *, with_judge: bool = True) -> Evaluator:
        """Judge (from EVALKIT_JUDGE_PROVIDER) + SQLiteStore from EVALKIT_* env vars.

        with_judge=False builds a store-only evaluator (get/list/review) that needs no
        judge/AWS configuration.
        """
        from evalkit.store import SQLiteStore, default_db_path

        judge = None
        if with_judge:
            provider = os.environ.get("EVALKIT_JUDGE_PROVIDER") or "bedrock"
            model = os.environ.get("EVALKIT_JUDGE_MODEL")
            if not model:
                raise ConfigError("EVALKIT_JUDGE_MODEL is required")
            temperature = _float_env("EVALKIT_JUDGE_TEMPERATURE", 0.0, minimum=0.0)
            timeout = _float_env("EVALKIT_JUDGE_TIMEOUT", 60.0, positive=True)

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
        return cls(judge, SQLiteStore(default_db_path()))

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

        # 1. validate inputs before the judge; invalid input is never persisted and never paid for
        try:
            if (criteria is None) == (rubric is None):
                raise RubricError("pass exactly one of `criteria` or `rubric`")
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise RubricError("`criteria` must be a dict of {name: description}")
                rubric = Rubric.from_dict(criteria)
            elif not isinstance(rubric, Rubric):
                rubric = Rubric.model_validate(rubric)
            self._check_sizes(prompt, model_output, reference_output, context, tags)
            common: dict[str, Any] = dict(
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
            # a draft is an (unjudged) error result: it validates the inputs, then supplies the
            # id and start time of the real result built after judging
            draft = EvaluationResult(status="error", error="not yet judged", **common)
            self._check_metadata_size(draft.metadata)
        except ValidationError as e:
            raise RubricError(f"invalid input: {format_validation_error(e)}") from e
        assert rubric.version is not None
        # bind the rubric version to its content BEFORE paying for a judge call
        register = getattr(self.store, "register_rubric", None)
        if register is not None:
            register(rubric.version, rubric.content_hash)

        # 2. judge + score; retry once only on malformed output. Every attempt is recorded.
        attempts: list[Attempt] = []
        scored = None
        error: JudgeError | None = None
        start = time.perf_counter()
        for n in (1, 2):
            started_at, t0, raw, exc = datetime.now(UTC), time.perf_counter(), None, None
            try:
                raw = judge.judge(prompt, model_output, reference_output, context, rubric)
                scored = rubric.score(raw)
            except JudgeError as e:
                exc = e
            except Exception as e:  # a buggy custom judge must still produce an error row
                exc = JudgeError(f"judge raised {type(e).__name__}: {e}", kind=UNEXPECTED)
                exc.__cause__ = e
            attempts.append(_attempt(n, started_at, time.perf_counter() - t0, exc, raw))
            error = exc
            if not isinstance(exc, JudgeOutputError):
                break  # success, or a failure that is never retried
        latency_ms = round((time.perf_counter() - start) * 1000)

        result = None
        if error is None:
            assert scored is not None
            scores, overall, verdict = scored
            try:
                result = EvaluationResult(
                    id=draft.id,
                    created_at=draft.created_at,
                    status="ok",
                    scores=scores,
                    overall_score=overall,
                    verdict=verdict,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    **common,
                )
            except ValidationError as e:  # a scoring bug must not lose the (paid) attempt evidence
                error = ScoringError(f"inconsistent result: {format_validation_error(e)}")
        if result is None:
            assert error is not None
            result = EvaluationResult(
                id=draft.id,
                created_at=draft.created_at,
                status="error",
                error=clip(f"{type(error).__name__}: {error}", MAX_ERROR_CHARS),
                latency_ms=latency_ms,
                attempts=attempts,
                **common,
            )

        # 3. persist ok and judge-stage error rows alike; a judged result is never silently lost
        if self.store is not None:
            try:
                self.store.save(result)
            except Exception as store_exc:
                raise self._store_error(result, store_exc, error) from store_exc
        if error is not None:
            error.evaluation_id = result.id if self.store is not None else None
            raise error
        return result

    def _store_error(
        self, result: EvaluationResult, cause: Exception, judge_error: JudgeError | None
    ) -> StoreError:
        spill = getattr(self.store, "spill_result", None)
        path: Path | None = None
        spill_note = "; the result is available as StoreError.result"
        if spill is not None:
            try:
                path = spill(result)
                spill_note = f"; the result was written to {path} (recover with recover_spilled())"
            except Exception as spill_exc:
                spill_note = (
                    f"; writing the recovery file also failed ({type(spill_exc).__name__}: "
                    f"{spill_exc}); the result is available as StoreError.result"
                )
        outcome = "judged ok" if result.status == "ok" else f"judge failed ({result.error})"
        err = StoreError(
            f"evaluation {result.id} ({outcome}) could not be saved: "
            f"{type(cause).__name__}: {cause}{spill_note}",
            result=result,
            spill_path=path,
        )
        err.judge_error = judge_error  # the judge failure, if that is what was being saved
        return err

    def _check_sizes(
        self,
        prompt: Any,
        model_output: Any,
        reference_output: Any,
        context: Any,
        tags: Any,
    ) -> None:
        fields = {
            "prompt": prompt,
            "model_output": model_output,
            "reference_output": reference_output,
            "context": context,
        }
        for name, value in fields.items():
            if not isinstance(value, str):
                continue  # wrong types are reported by model validation
            try:
                size = len(value) if len(value) > self.limits.max_field_bytes else None
                if size is None:
                    size = len(value.encode("utf-8"))
            except UnicodeEncodeError as e:
                raise RubricError(
                    f"`{name}` is not valid Unicode text (unpaired surrogate): {e.reason}"
                ) from e
            if size > self.limits.max_field_bytes:
                raise RubricError(
                    f"`{name}` is too large ({size} bytes; limit {self.limits.max_field_bytes})"
                )
        if isinstance(tags, list):
            if len(tags) > self.limits.max_tags:
                raise RubricError(f"too many tags ({len(tags)}; limit {self.limits.max_tags})")
            if any(isinstance(t, str) and len(t) > MAX_TAG_CHARS for t in tags):
                raise RubricError(f"a tag is longer than {MAX_TAG_CHARS} characters")

    def _check_metadata_size(self, metadata: dict[str, Any]) -> None:
        size = len(json.dumps(metadata, allow_nan=False).encode("utf-8", "replace"))
        if size > self.limits.max_metadata_bytes:
            raise RubricError(
                f"`metadata` is too large ({size} bytes as JSON; "
                f"limit {self.limits.max_metadata_bytes})"
            )

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

    def list_page(
        self, tag: str | None = None, limit: int = 20, cursor: str | None = None
    ) -> tuple[list[EvaluationResult], str | None]:
        """Like list(), plus a `next_cursor` to continue from (None on the last page)."""
        store = self._require_store()
        page = getattr(store, "list_page", None)
        if page is None:
            raise ConfigError("this store does not support pagination cursors")
        return page(tag=tag, limit=limit, cursor=cursor)

    def recover_spilled(self):
        """Save results that a failed store.save() left in the spill directory."""
        recover = getattr(self._require_store(), "recover_spilled", None)
        if recover is None:
            raise ConfigError("this store does not support spill recovery")
        return recover()

    def _require_store(self) -> Store:
        if self.store is None:
            raise ConfigError("no store configured")
        return self.store


def _float_env(
    name: str, default: float, *, minimum: float | None = None, positive: bool = False
) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        number = float(value)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {value!r}") from e
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be a finite number, got {value!r}")
    if positive and number <= 0:
        raise ConfigError(f"{name} must be > 0, got {value!r}")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value!r}")
    return number


def _outcome(exc: JudgeError) -> AttemptOutcome:
    if isinstance(exc, JudgeOutputError):
        return "invalid_output"
    if isinstance(exc, JudgeTimeoutError):
        return "timeout"
    if isinstance(exc, ScoringError):
        return "scoring_error"
    return "unexpected_error" if exc.kind == UNEXPECTED else "provider_error"


def _attempt(
    n: int, started_at: datetime, seconds: float, exc: JudgeError | None, raw: Any
) -> Attempt:
    """Evidence for one judge call. The rejected/partial output is kept (truncated) only for
    failed attempts; a successful attempt keeps just a hash, its content is the stored scores."""
    evidence = raw if raw is not None else getattr(exc, "raw", None)
    text, sha, truncated = capture_evidence(evidence, keep_text=exc is not None)
    return Attempt(
        n=n,
        outcome="ok" if exc is None else _outcome(exc),
        started_at=started_at,
        duration_ms=round(seconds * 1000),
        error_type=None if exc is None else type(exc).__name__,
        error=None if exc is None else clip(str(exc), MAX_ERROR_CHARS),
        kind=getattr(exc, "kind", None),
        raw=text,
        raw_sha256=sha,
        raw_truncated=truncated,
    )
