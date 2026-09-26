"""Bedrock via its OpenAI-compatible endpoint (chat completions + function calling).

Added on top of the original v1 architecture, which specified only `bedrock.py`
(boto3 Converse). Kept isolated the same way: this is the only module that imports
`openai`. `BedrockOpenAIClient` is the transport; `BedrockOpenAIJudge` is the legacy `Judge`
adapter over it. Requires the `bedrock-openai` extra (`pip install evalkit[bedrock-openai]`).
"""

from typing import Any

from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import OpenAI

from evalkit import safejson
from evalkit.failures import EvalFailure, FailureClass
from evalkit.judge_eval import MAX_TOKENS, ClientJudge
from evalkit.llm import (
    STOP_CONTENT_FILTER,
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_TOOL,
    LLMRequest,
    LLMResponse,
    Usage,
    rejected_request,
)

__all__ = ["DEFAULT_BASE_URL", "MAX_TOKENS", "BedrockOpenAIClient", "BedrockOpenAIJudge"]

DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/v1"
_FINISH_REASONS = {
    "stop": STOP_END,
    "tool_calls": STOP_TOOL,
    "function_call": STOP_TOOL,
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_CONTENT_FILTER,
}
_PROVIDER = "bedrock-openai"


def _evidence(message: Any) -> Any:
    """The assistant message as plain data, for the attempt record when the response is unusable."""
    dump = getattr(message, "model_dump", None)
    return dump(mode="json") if callable(dump) else None


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _status_failure(e: OpenAIAPIStatusError, role: str) -> EvalFailure:
    status = getattr(e, "status_code", None)
    text = f"Bedrock (OpenAI-compatible) API error: {e}"
    extra = {"http_status": status, "request_id": getattr(e, "request_id", None)}
    if status in (400, 404, 422):
        return rejected_request(role, _PROVIDER, text, code=getattr(e, "code", None), **extra)
    if status in (401, 403):
        return EvalFailure(FailureClass.INFRA, "auth", text, provider=_PROVIDER, **extra)
    if status == 429:
        try:
            retry = float(e.response.headers.get("retry-after"))
        except (AttributeError, TypeError, ValueError):
            retry = None
        return EvalFailure(
            FailureClass.INFRA,
            "rate_limited",
            text,
            provider=_PROVIDER,
            retry_after_s=retry,
            **extra,
        )
    return EvalFailure(
        FailureClass.INFRA, "provider_unavailable", text, provider=_PROVIDER, **extra
    )


class BedrockOpenAIClient:
    provider = _PROVIDER

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: Any = None,
    ):
        self.model = model
        self.timeout = timeout
        self._client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # evalkit owns retries (OD-1); no SDK-level retries here either
            default_headers={"OpenAI-Project": "default"},
        )

    def call(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "timeout": req.timeout_s,  # the run's request timeout, not the client's default
        }
        if req.tool is not None:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": req.tool.name,
                        "description": req.tool.description,
                        "parameters": req.tool.schema,
                    },
                }
            ]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": req.tool.name}}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIAPITimeoutError as e:
            raise EvalFailure(
                FailureClass.INFRA, "timeout",
                f"Bedrock (OpenAI-compatible) request timed out: {e}", provider=_PROVIDER,
            ) from e  # fmt: skip
        except OpenAIAPIConnectionError as e:
            raise EvalFailure(
                FailureClass.INFRA, "connection",
                f"Bedrock (OpenAI-compatible) connection failed: {e}", provider=_PROVIDER,
            ) from e  # fmt: skip
        except OpenAIAPIStatusError as e:
            raise _status_failure(e, req.role) from e
        except OpenAIAPIError as e:
            raise EvalFailure(
                FailureClass.INFRA, "provider_unavailable",
                f"Bedrock (OpenAI-compatible) API error: {e}", provider=_PROVIDER,
            ) from e  # fmt: skip
        return self._parse(response, req)

    def _parse(self, response: Any, req: LLMRequest) -> LLMResponse:
        usage_obj = getattr(response, "usage", None)
        usage = Usage(
            _count(getattr(usage_obj, "prompt_tokens", None)),
            _count(getattr(usage_obj, "completion_tokens", None)),
        )
        request_id = getattr(response, "id", None)
        request_id = request_id if isinstance(request_id, str) else None
        if not response.choices:
            return LLMResponse(
                usage=usage,
                request_id=request_id,
                payload_error="judge response contained no choices",
                payload_error_kind="no_choices",
            )
        choice = response.choices[0]
        finish = choice.finish_reason
        common: dict[str, Any] = {
            "usage": usage,
            "request_id": request_id,
            "stop_reason": _FINISH_REASONS.get(finish, STOP_OTHER),
            "provider_stop": None if finish is None else f"finish_reason={finish}",
            "text": choice.message.content if isinstance(choice.message.content, str) else None,
        }
        raw = _evidence(choice.message)
        if req.tool is not None:
            for call in choice.message.tool_calls or []:
                if call.function.name == req.tool.name:
                    try:
                        payload = safejson.loads(call.function.arguments)
                    except (ValueError, RecursionError) as e:  # NaN/Infinity, duplicate keys, ...
                        return LLMResponse(
                            raw=call.function.arguments,
                            payload_error=f"judge tool arguments were not valid JSON: {e}",
                            payload_error_kind="invalid_json",
                            **common,
                        )
                    return LLMResponse(payload=payload, raw=raw, **common)
        return LLMResponse(raw=raw, **common)


class BedrockOpenAIJudge(ClientJudge):
    """Legacy `Judge` protocol over `BedrockOpenAIClient` (constructor and behaviour unchanged)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client: Any = None,
    ):
        llm = BedrockOpenAIClient(model, api_key, base_url, timeout, client)
        super().__init__(llm, temperature, timeout)
        self._client = llm._client
