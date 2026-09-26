"""LLM transport contract (docs/TARGET-ARCHITECTURE.md §6.4): one provider request per call.

A client owns the provider request, the response, usage, latency, provider metadata and
provider errors -- nothing about what the response *means*. Judge semantics (prompt, scoring)
live in `evalkit.judge_eval`; a model target uses the same clients. No SDK is imported here;
SDKs stay confined to `bedrock.py` and `bedrock_openai.py`.

A client raises a classified `EvalFailure` for anything that stopped it getting a response
(throttling, outage, timeout, auth, a rejected request). A response that arrives but is not what
the caller wanted (truncated, blocked, no tool call) is *returned* with a normalized
`stop_reason`; the caller decides what that means for its role (a truncated judge answer is an
evaluator failure, a truncated target answer is still an answer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from evalkit.failures import EvalFailure, FailureClass

# Provider stop reasons, normalized. Anything unrecognized is kept verbatim in provider_stop.
STOP_END = "end_turn"
STOP_TOOL = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_GUARDRAIL = "guardrail_intervened"
STOP_CONTENT_FILTER = "content_filtered"
STOP_MALFORMED = "malformed"
STOP_CONTEXT_WINDOW = "model_context_window_exceeded"
STOP_OTHER = "other"


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValueError(f"{name} must be a non-negative integer or None, got {v!r}")


@dataclass(frozen=True)
class ToolSpec:
    """A forced tool: the model must answer by calling it, with input matching `schema`."""

    name: str
    description: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    tool: ToolSpec | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: float = 60.0
    sample_index: int = 0  # varies for independent samples; part of any future cache key
    # who is calling decides how a provider *rejection* is attributed (design 7.4): a judge with a
    # bad model id is an evaluator problem, a target with one is a configuration problem
    role: Literal["evaluator", "target"] = "evaluator"


@dataclass(frozen=True)
class LLMResponse:
    payload: dict[str, Any] | None = None  # the forced tool's input, when there was one
    text: str | None = None  # free text (a model target's answer)
    usage: Usage = field(default_factory=Usage)
    request_id: str | None = None
    provider_latency_ms: int | None = None
    stop_reason: str = STOP_OTHER  # normalized (STOP_* above)
    provider_stop: str | None = None  # as the provider said it, e.g. "stopReason=max_tokens"
    payload_error: str | None = None  # the tool call arrived but was unusable (e.g. bad JSON)
    payload_error_kind: str | None = None  # its provider-detected sub-case (invalid_json, ...)
    raw: Any = None  # provider output as plain data, kept only as failure evidence


class LLMClient(Protocol):
    """Transport only. Must not retry (SDK retries are off; EvalKit owns retry policy)."""

    provider: str
    model: str

    def call(self, req: LLMRequest) -> LLMResponse: ...


# A provider that rejects a request because THIS input is too long is describing the case, not the
# configuration: every other case would have been accepted. Providers say so in prose (Bedrock's
# ValidationException) or with a code (OpenAI-compatible `context_length_exceeded`). This is a
# heuristic over documented phrasings, deliberately narrow, and it has NOT been verified against a
# live provider (docs/EVALUATION-METHODOLOGY.md). A phrasing it misses is treated as a rejected
# request, which is stopped only when it repeats (calls.RunGuard), never on first sight.
_OVERSIZE = re.compile(
    r"(input|prompt|request|message)s?\b.{0,40}\btoo (long|large|big)"
    r"|too many (input |prompt )?tokens"
    r"|(maximum|max(imum)?) (context|input|prompt)( length| window| size| tokens)?"
    r"|context (length|window) (exceeded|limit)|exceeds? the (model'?s? )?(context|token)",
    re.IGNORECASE,
)
_OVERSIZE_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})


def looks_oversize(message: str, code: str | None = None) -> bool:
    return code in _OVERSIZE_CODES or bool(_OVERSIZE.search(message))


def rejected_request(
    role: str, provider: str, message: str, *, code: str | None = None, **meta: Any
) -> EvalFailure:
    """The provider refused the request itself (400 / unknown model). Who is to blame depends on
    what was wrong (design 7.4): an input too long for the model is `input.oversize`, the case's
    own problem; anything else says the configuration is wrong -- every call would fail, so it is
    systemic (stopped on repetition) -- and depends on the caller's role."""
    if looks_oversize(message, code):
        return EvalFailure(FailureClass.INPUT, "oversize", message, provider=provider, **meta)
    if role == "target":
        return EvalFailure(FailureClass.INPUT, "bad_config", message, provider=provider, **meta)
    return EvalFailure(FailureClass.EVALUATOR, "bad_request", message, provider=provider, **meta)
