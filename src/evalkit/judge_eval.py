"""Judge semantics over an `LLMClient` (docs/TARGET-ARCHITECTURE.md §6.4).

Owns what the transport does not: building the judge request (prompt, forced tool, schema) and
deciding what a *response* means -- a usable payload, or a classified failure. The legacy `Judge`
protocol classes (`BedrockJudge`, `BedrockOpenAIJudge`) are thin adapters built on the same two
functions, so the single-record path and run evaluation cannot disagree about a response.
"""

from __future__ import annotations

from typing import Any

from evalkit.errors import JudgeError, JudgeOutputError, JudgeTimeoutError
from evalkit.failures import EvalFailure, FailureClass
from evalkit.judge import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    output_schema,
    render_prompt,
)
from evalkit.llm import (
    STOP_CONTENT_FILTER,
    STOP_CONTEXT_WINDOW,
    STOP_GUARDRAIL,
    STOP_MALFORMED,
    STOP_MAX_TOKENS,
    LLMClient,
    LLMRequest,
    LLMResponse,
    ToolSpec,
)
from evalkit.models import Rubric

# ponytail: fixed output budget; very large rubrics may truncate -> evaluator.truncated
MAX_TOKENS = 4096

# normalized stop reason -> (failure class, kind, legacy sub-kind, what happened)
_STOP_FAILURES = {
    STOP_MAX_TOKENS: (FailureClass.EVALUATOR, "truncated", "truncated", "was truncated"),
    STOP_GUARDRAIL: (FailureClass.EVALUATOR, "refused", "refused", "was blocked by a guardrail"),
    STOP_CONTENT_FILTER: (
        FailureClass.EVALUATOR,
        "refused",
        "refused",
        "was blocked by a content filter",
    ),  # fmt: skip
    STOP_MALFORMED: (
        FailureClass.EVALUATOR,
        "invalid_output",
        "malformed_output",
        "was malformed",
    ),  # fmt: skip
    STOP_CONTEXT_WINDOW: (
        FailureClass.INPUT,
        "oversize",
        "context_window",
        "exceeded the model context window",
    ),  # fmt: skip
}


def judge_tool(rubric: Rubric) -> ToolSpec:
    return ToolSpec(TOOL_NAME, TOOL_DESCRIPTION, output_schema(rubric))


def build_request(
    prompt: str,
    model_output: str,
    reference_output: str | None,
    context: str | None,
    rubric: Rubric,
    *,
    temperature: float,
    timeout_s: float,
    max_tokens: int = MAX_TOKENS,
    sample_index: int = 0,
) -> LLMRequest:
    return LLMRequest(
        system=SYSTEM_PROMPT,
        user=render_prompt(prompt, model_output, reference_output, context, rubric),
        tool=judge_tool(rubric),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        sample_index=sample_index,
        role="evaluator",
    )


def judge_payload(resp: LLMResponse) -> dict[str, Any]:
    """The judge's structured answer, or an `EvalFailure` (with the provider output as `raw`
    evidence). Does not validate against the rubric: that is `Rubric.score`."""
    if resp.payload_error is not None:
        raise EvalFailure(
            FailureClass.EVALUATOR, "invalid_output", resp.payload_error,
            subkind=resp.payload_error_kind, raw=resp.raw,
        )  # fmt: skip
    signal = _STOP_FAILURES.get(resp.stop_reason)
    if signal is not None:
        cls, kind, subkind, what = signal
        suffix = f" ({resp.provider_stop})" if resp.provider_stop else ""
        raise EvalFailure(
            cls, kind, f"judge response {what}{suffix}", subkind=subkind, raw=resp.raw
        )
    if resp.payload is None:
        raise EvalFailure(
            FailureClass.EVALUATOR,
            "invalid_output",
            f"judge response contained no {TOOL_NAME} tool call",
            subkind="no_tool_call",
            raw=resp.raw,
        )
    return resp.payload


def _legacy(failure: EvalFailure) -> JudgeError:
    """The pre-taxonomy exception for a classified failure (same message, `.failure` attached)."""
    if failure.failure_class is FailureClass.INFRA and failure.kind == "timeout":
        return JudgeTimeoutError(str(failure), failure=failure.failure)
    if failure.failure_class is FailureClass.EVALUATOR and failure.kind == "invalid_output":
        return JudgeOutputError(
            str(failure), kind=failure.subkind, raw=failure.raw, failure=failure.failure
        )
    if failure.subkind is not None:  # truncated / refused / context_window
        return JudgeOutputError(
            str(failure), kind=failure.subkind, raw=failure.raw, failure=failure.failure
        )
    return JudgeError(str(failure), failure=failure.failure)


class ClientJudge:
    """The legacy `Judge` protocol over any `LLMClient` (used by `BedrockJudge` and friends)."""

    prompt_version = PROMPT_VERSION

    def __init__(self, client: LLMClient, temperature: float, timeout: float):
        self._llm = client
        self.temperature = temperature
        self.timeout = timeout

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def model(self) -> str:
        return self._llm.model

    def judge(
        self,
        prompt: str,
        model_output: str,
        reference_output: str | None,
        context: str | None,
        rubric: Rubric,
    ) -> dict[str, Any]:
        request = build_request(
            prompt, model_output, reference_output, context, rubric,
            temperature=self.temperature, timeout_s=self.timeout,
        )  # fmt: skip
        try:
            return judge_payload(self._llm.call(request))
        except EvalFailure as f:
            raise _legacy(f) from f
