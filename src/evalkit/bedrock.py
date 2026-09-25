"""AWS Bedrock (Converse API). The only module that imports boto3/botocore.

`BedrockClient` is the transport (one Converse request per call, SDK retries off, provider errors
classified per design 7.4). `BedrockJudge` is the legacy `Judge` adapter over it.
"""

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError

from evalkit.errors import ConfigError
from evalkit.failures import EvalFailure, FailureClass
from evalkit.judge_eval import MAX_TOKENS, ClientJudge
from evalkit.llm import (
    STOP_CONTENT_FILTER,
    STOP_CONTEXT_WINDOW,
    STOP_END,
    STOP_GUARDRAIL,
    STOP_MALFORMED,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_TOOL,
    LLMRequest,
    LLMResponse,
    Usage,
    rejected_request,
)

__all__ = ["MAX_TOKENS", "BedrockClient", "BedrockJudge"]

_STOP_REASONS = {
    "end_turn": STOP_END,
    "stop_sequence": STOP_END,
    "tool_use": STOP_TOOL,
    "max_tokens": STOP_MAX_TOKENS,
    "guardrail_intervened": STOP_GUARDRAIL,
    "content_filtered": STOP_CONTENT_FILTER,
    "malformed_model_output": STOP_MALFORMED,
    "malformed_tool_use": STOP_MALFORMED,
    "model_context_window_exceeded": STOP_CONTEXT_WINDOW,
}
# design 7.4: Bedrock error code -> (class, kind)
_ERROR_CODES = {
    "ThrottlingException": (FailureClass.INFRA, "rate_limited"),
    "TooManyRequestsException": (FailureClass.INFRA, "rate_limited"),
    "ServiceUnavailableException": (FailureClass.INFRA, "provider_unavailable"),
    "InternalServerException": (FailureClass.INFRA, "provider_unavailable"),
    "ModelNotReadyException": (FailureClass.INFRA, "provider_unavailable"),
    "ModelErrorException": (FailureClass.INFRA, "provider_unavailable"),
    "ModelTimeoutException": (FailureClass.INFRA, "timeout"),
    "AccessDeniedException": (FailureClass.INFRA, "auth"),
    "UnrecognizedClientException": (FailureClass.INFRA, "auth"),
    "ExpiredTokenException": (FailureClass.INFRA, "auth"),
    "InvalidSignatureException": (FailureClass.INFRA, "auth"),
    "ServiceQuotaExceededException": (FailureClass.INFRA, "quota_exhausted"),
}
_REJECTED = {"ValidationException", "ResourceNotFoundException"}


def _retry_after(response: dict[str, Any]) -> float | None:
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value < float("inf") else None


class BedrockClient:
    provider = "bedrock"

    def __init__(
        self,
        model: str,
        timeout: float = 60.0,
        region: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.timeout = timeout
        if client is None:
            config = Config(
                retries={"total_max_attempts": 1, "mode": "standard"},  # no SDK retries
                connect_timeout=timeout,
                read_timeout=timeout,
            )
            try:
                client = boto3.client("bedrock-runtime", region_name=region, config=config)
            except BotoCoreError as e:
                raise ConfigError(f"cannot create Bedrock client: {e}") from e
        self._client = client

    def call(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "modelId": self.model,
            "system": [{"text": req.system}],
            "messages": [{"role": "user", "content": [{"text": req.user}]}],
            "inferenceConfig": {"temperature": req.temperature, "maxTokens": req.max_tokens},
        }
        if req.tool is not None:
            kwargs["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": req.tool.name,
                            "description": req.tool.description,
                            "inputSchema": {"json": req.tool.schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": req.tool.name}},
            }
        try:
            response = self._client.converse(**kwargs)
        except (ReadTimeoutError, ConnectTimeoutError) as e:
            raise EvalFailure(
                FailureClass.INFRA, "timeout",
                f"Bedrock request timed out after {self.timeout}s: {e}", provider=self.provider,
            ) from e  # fmt: skip
        except ClientError as e:
            raise self._classify(e, req.role) from e
        except BotoCoreError as e:
            raise EvalFailure(
                FailureClass.INFRA, "connection", f"Bedrock request failed: {e}",
                provider=self.provider,
            ) from e  # fmt: skip
        return self._parse(response, req)

    def _classify(self, e: ClientError, role: str) -> EvalFailure:
        err = e.response.get("Error", {})
        code, message = err.get("Code", "Unknown"), err.get("Message", str(e))
        meta = e.response.get("ResponseMetadata", {})
        extra = {"http_status": meta.get("HTTPStatusCode"), "request_id": meta.get("RequestId")}
        text = f"Bedrock {code}: {message}"
        if code in _REJECTED:
            return rejected_request(role, self.provider, text, **extra)
        cls, kind = _ERROR_CODES.get(code, (FailureClass.INFRA, "provider_unavailable"))
        return EvalFailure(
            cls, kind, text, provider=self.provider, retry_after_s=_retry_after(e.response), **extra
        )

    def _parse(self, response: dict[str, Any], req: LLMRequest) -> LLMResponse:
        output = response.get("output")
        stop = response.get("stopReason")
        payload, texts = None, []
        for block in (output or {}).get("message", {}).get("content", []):
            if "text" in block:
                texts.append(block["text"])
            tool_use = block.get("toolUse")
            if (
                payload is None
                and tool_use
                and req.tool is not None
                and tool_use.get("name") == req.tool.name
            ):
                payload = tool_use.get("input")
        usage = response.get("usage") or {}
        return LLMResponse(
            payload=payload,
            text="".join(texts) if texts else None,
            usage=Usage(_count(usage.get("inputTokens")), _count(usage.get("outputTokens"))),
            request_id=response.get("ResponseMetadata", {}).get("RequestId"),
            provider_latency_ms=_count((response.get("metrics") or {}).get("latencyMs")),
            stop_reason=_STOP_REASONS.get(stop, STOP_OTHER),
            provider_stop=None if stop is None else f"stopReason={stop}",
            raw=output,
        )


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class BedrockJudge(ClientJudge):
    """Legacy `Judge` protocol over `BedrockClient` (constructor and behaviour unchanged)."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        region: str | None = None,
        client: Any = None,
    ):
        llm = BedrockClient(model, timeout, region, client)
        super().__init__(llm, temperature, timeout)
        self._client = llm._client  # the SDK client, as before the transport split
