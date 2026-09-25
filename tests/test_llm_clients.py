"""The LLM client split: transport (usage, latency, request id, classified errors, evidence) is
separate from judge semantics; legacy Judge adapters behave as before and now carry `.failure`."""

import json
from types import SimpleNamespace

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from conftest import judged
from openai import APIConnectionError, APIStatusError, APITimeoutError

from evalkit import (
    EvalFailure,
    FailureClass,
    JudgeError,
    JudgeOutputError,
    JudgeTimeoutError,
    Rubric,
)
from evalkit.bedrock import BedrockClient, BedrockJudge
from evalkit.bedrock_openai import BedrockOpenAIClient, BedrockOpenAIJudge
from evalkit.judge import TOOL_NAME
from evalkit.judge_eval import build_request, judge_payload
from evalkit.llm import (
    STOP_CONTENT_FILTER,
    STOP_END,
    STOP_MALFORMED,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_TOOL,
    LLMRequest,
    LLMResponse,
    ToolSpec,
    Usage,
)

RUBRIC = Rubric.from_dict({"a": "A?", "b": "B?"})
TOOL = ToolSpec("t", "d", {"type": "object"})


def req(**kw):
    return LLMRequest(system="sys", user="usr", tool=kw.pop("tool", TOOL), **kw)


# --- value objects ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_usage_refuses_non_counts(bad):
    with pytest.raises(ValueError):
        Usage(input_tokens=bad)
    Usage(input_tokens=0, output_tokens=None)


def test_transport_modules_keep_sdks_out_of_the_contract():
    import evalkit.judge_eval as judge_eval
    import evalkit.llm as llm

    for module in (llm, judge_eval):
        source = open(module.__file__).read()
        for sdk in ("import boto3", "from botocore", "import openai", "from openai"):
            assert sdk not in source


# --- Bedrock transport -----------------------------------------------------------------------


class FakeConverse:
    def __init__(self, result):
        self.result, self.calls = result, []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def converse(payload=None, stop="tool_use", **extra):
    content = [{"text": "thinking"}]
    if payload is not None:
        content.append({"toolUse": {"name": TOOL.name, "input": payload}})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": {"inputTokens": 120, "outputTokens": 30},
        "metrics": {"latencyMs": 812},
        "ResponseMetadata": {"RequestId": "req-abc", "HTTPStatusCode": 200},
        **extra,
    }


def bedrock(result):
    fake = FakeConverse(result)
    return BedrockClient("m", client=fake), fake


def test_bedrock_client_returns_payload_usage_latency_and_request_id():
    client, fake = bedrock(converse({"x": 1}))
    resp = client.call(req(temperature=0.3, max_tokens=99))
    assert resp.payload == {"x": 1} and resp.text == "thinking"
    assert resp.usage == Usage(120, 30) and resp.provider_latency_ms == 812
    assert (resp.request_id, resp.stop_reason, resp.provider_stop) == (
        "req-abc", STOP_TOOL, "stopReason=tool_use",
    )  # fmt: skip
    assert resp.raw["message"]["content"][0] == {"text": "thinking"}
    (sent,) = fake.calls
    assert sent["inferenceConfig"] == {"temperature": 0.3, "maxTokens": 99}
    assert sent["toolConfig"]["toolChoice"] == {"tool": {"name": "t"}}
    assert sent["system"] == [{"text": "sys"}]


def test_a_plain_text_request_sends_no_tool():
    client, fake = bedrock(converse(None, stop="end_turn"))
    resp = client.call(req(tool=None))
    assert "toolConfig" not in fake.calls[0]
    assert (resp.text, resp.payload, resp.stop_reason) == ("thinking", None, STOP_END)


@pytest.mark.parametrize(
    "stop,normalized",
    [
        ("max_tokens", STOP_MAX_TOKENS),
        ("content_filtered", STOP_CONTENT_FILTER),
        ("malformed_tool_use", STOP_MALFORMED),
        ("malformed_model_output", STOP_MALFORMED),
        ("brand_new_reason", STOP_OTHER),
        (None, STOP_OTHER),
    ],
)
def test_bedrock_stop_reasons_are_normalized_and_returned_not_raised(stop, normalized):
    client, _ = bedrock(converse(None, stop=stop))
    assert client.call(req()).stop_reason == normalized


def test_missing_usage_and_metadata_do_not_break_the_transport():
    client, _ = bedrock({"output": {"message": {"content": []}}, "stopReason": "end_turn"})
    resp = client.call(req())
    assert resp.usage == Usage(None, None) and resp.request_id is None
    assert resp.provider_latency_ms is None


def code_error(code, status=400, headers=None):
    meta = {"HTTPStatusCode": status, "RequestId": "rid", "HTTPHeaders": headers or {}}
    return ClientError(
        {"Error": {"Code": code, "Message": "m"}, "ResponseMetadata": meta}, "Converse"
    )


@pytest.mark.parametrize(
    "code,cls,kind",
    [
        ("ThrottlingException", "infrastructure", "rate_limited"),
        ("TooManyRequestsException", "infrastructure", "rate_limited"),
        ("ServiceUnavailableException", "infrastructure", "provider_unavailable"),
        ("InternalServerException", "infrastructure", "provider_unavailable"),
        ("ModelNotReadyException", "infrastructure", "provider_unavailable"),
        ("ModelErrorException", "infrastructure", "provider_unavailable"),
        ("ModelTimeoutException", "infrastructure", "timeout"),
        ("AccessDeniedException", "infrastructure", "auth"),
        ("ExpiredTokenException", "infrastructure", "auth"),
        ("ServiceQuotaExceededException", "infrastructure", "quota_exhausted"),
        ("SomethingNew", "infrastructure", "provider_unavailable"),
        ("ValidationException", "evaluator", "bad_request"),
        ("ResourceNotFoundException", "evaluator", "bad_request"),
    ],
)
def test_every_bedrock_error_code_maps_to_the_design_table(code, cls, kind):
    client, _ = bedrock(code_error(code, status=429))
    with pytest.raises(EvalFailure) as info:
        client.call(req())
    f = info.value
    assert (f.failure_class.value, f.kind) == (cls, kind)
    assert f"Bedrock {code}" in str(f) and f.provider == "bedrock"
    assert (f.http_status, f.request_id) == (429, "rid")


@pytest.mark.parametrize("code", ["ValidationException", "ResourceNotFoundException"])
def test_a_rejected_request_is_attributed_by_the_callers_role(code):
    client, _ = bedrock(code_error(code))
    with pytest.raises(EvalFailure) as as_judge:
        client.call(req(role="evaluator"))
    with pytest.raises(EvalFailure) as as_target:
        client.call(req(role="target"))
    assert (as_judge.value.failure_class, as_judge.value.kind) == (
        FailureClass.EVALUATOR,
        "bad_request",
    )
    assert (as_target.value.failure_class, as_target.value.kind) == (
        FailureClass.INPUT,
        "bad_config",
    )
    assert as_judge.value.systemic and as_target.value.systemic  # would fail every case


def test_bedrock_retry_after_is_captured_and_bad_values_ignored():
    client, _ = bedrock(code_error("ThrottlingException", 429, {"retry-after": "2.5"}))
    with pytest.raises(EvalFailure) as info:
        client.call(req())
    assert info.value.retry_after_s == 2.5
    for bad in ("soon", "-1", "inf", ""):
        client, _ = bedrock(code_error("ThrottlingException", 429, {"retry-after": bad}))
        with pytest.raises(EvalFailure) as info:
            client.call(req())
        assert info.value.retry_after_s is None


@pytest.mark.parametrize(
    "exc,kind",
    [
        (ReadTimeoutError(endpoint_url="https://b"), "timeout"),
        (ConnectTimeoutError(endpoint_url="https://b"), "timeout"),
        (EndpointConnectionError(endpoint_url="https://b"), "connection"),
        (BotoCoreError(), "connection"),
    ],
)
def test_bedrock_transport_exceptions_are_classified_and_never_retried(exc, kind):
    client, fake = bedrock(exc)
    with pytest.raises(EvalFailure) as info:
        client.call(req())
    assert (info.value.failure_class, info.value.kind) == (FailureClass.INFRA, kind)
    assert len(fake.calls) == 1  # one request: retry policy belongs to EvalKit, not the client


def test_provider_error_text_is_scrubbed_in_the_failure():
    exc = ClientError(
        {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "user arn:aws:iam::123456789012:user/x",
            }
        },
        "Converse",
    )
    client, _ = bedrock(exc)
    with pytest.raises(EvalFailure) as info:
        client.call(req())
    assert "123456789012" not in str(info.value)


# --- Bedrock via OpenAI-compatible transport -------------------------------------------------


def chat(payload=None, finish="tool_calls", content=None, response_id="chatcmpl-1", usage=True):
    calls = None
    if payload is not None:
        calls = [
            SimpleNamespace(
                id="c", function=SimpleNamespace(name="t", arguments=json.dumps(payload))
            )
        ]
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=calls)
    return SimpleNamespace(
        id=response_id,
        choices=[SimpleNamespace(message=msg, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=7) if usage else None,
    )


class FakeOpenAI:
    def __init__(self, result):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.result = result

    def _create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def oa(result):
    fake = FakeOpenAI(result)
    return BedrockOpenAIClient("m", client=fake), fake


def test_openai_client_returns_payload_usage_and_request_id():
    client, fake = oa(chat({"x": 1}))
    resp = client.call(req(max_tokens=50))
    assert (
        resp.payload == {"x": 1} and resp.usage == Usage(50, 7) and resp.request_id == "chatcmpl-1"
    )
    assert (resp.stop_reason, resp.provider_stop) == (STOP_TOOL, "finish_reason=tool_calls")
    assert (
        fake.calls[0]["max_tokens"] == 50 and fake.calls[0]["tools"][0]["function"]["name"] == "t"
    )


def test_openai_plain_text_request_and_missing_usage():
    client, fake = oa(chat(None, finish="stop", content="hello", usage=False))
    resp = client.call(req(tool=None))
    assert (resp.text, resp.payload, resp.usage, resp.stop_reason) == (
        "hello", None, Usage(None, None), STOP_END,
    )  # fmt: skip
    assert "tools" not in fake.calls[0]


@pytest.mark.parametrize(
    "finish,normalized",
    [("length", STOP_MAX_TOKENS), ("content_filter", STOP_CONTENT_FILTER), ("weird", STOP_OTHER)],
)
def test_openai_finish_reasons_are_normalized(finish, normalized):
    client, _ = oa(chat({"x": 1}, finish=finish))
    assert client.call(req()).stop_reason == normalized


def test_invalid_tool_arguments_are_returned_as_a_payload_error_with_raw_text():
    bad = SimpleNamespace(id="c", function=SimpleNamespace(name="t", arguments="{nope"))
    msg = SimpleNamespace(role="assistant", content=None, tool_calls=[bad])
    client, _ = oa(
        SimpleNamespace(
            id="i", choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None
        )
    )
    resp = client.call(req())
    assert resp.payload is None and resp.raw == "{nope"
    assert "not valid JSON" in resp.payload_error and resp.payload_error_kind == "invalid_json"


def status_error(status, headers=None):
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    return APIStatusError("nope", response=response, body=None)


@pytest.mark.parametrize(
    "status,cls,kind",
    [
        (429, "infrastructure", "rate_limited"),
        (401, "infrastructure", "auth"),
        (403, "infrastructure", "auth"),
        (500, "infrastructure", "provider_unavailable"),
        (503, "infrastructure", "provider_unavailable"),
        (400, "evaluator", "bad_request"),
        (404, "evaluator", "bad_request"),
        (422, "evaluator", "bad_request"),
    ],
)
def test_openai_status_codes_map_to_the_design_table(status, cls, kind):
    client, fake = oa(status_error(status, {"retry-after": "3"}))
    with pytest.raises(EvalFailure) as info:
        client.call(req())
    f = info.value
    assert (f.failure_class.value, f.kind, f.http_status) == (cls, kind, status)
    assert f.retry_after_s == (3.0 if status == 429 else None)
    assert len(fake.calls) == 1


def test_openai_transport_exceptions():
    client, _ = oa(APITimeoutError(request=None))
    with pytest.raises(EvalFailure) as t:
        client.call(req())
    client, _ = oa(APIConnectionError(request=None))
    with pytest.raises(EvalFailure) as c:
        client.call(req())
    assert (t.value.kind, c.value.kind) == ("timeout", "connection")


def test_openai_rejected_request_by_role():
    client, _ = oa(status_error(400))
    with pytest.raises(EvalFailure) as target:
        client.call(req(role="target"))
    assert (target.value.failure_class, target.value.kind) == (FailureClass.INPUT, "bad_config")


# --- judge semantics over a response ---------------------------------------------------------


def test_judge_payload_returns_the_payload_of_a_good_response():
    assert judge_payload(LLMResponse(payload={"a": 1}, stop_reason=STOP_TOOL)) == {"a": 1}


@pytest.mark.parametrize(
    "resp,cls,kind,sub",
    [
        (
            LLMResponse(stop_reason="max_tokens", provider_stop="stopReason=max_tokens"),
            "evaluator",
            "truncated",
            "truncated",
        ),
        (LLMResponse(stop_reason="guardrail_intervened"), "evaluator", "refused", "refused"),
        (LLMResponse(stop_reason="content_filtered"), "evaluator", "refused", "refused"),
        (LLMResponse(stop_reason="malformed"), "evaluator", "invalid_output", "malformed_output"),
        (
            LLMResponse(stop_reason="model_context_window_exceeded"),
            "input",
            "oversize",
            "context_window",
        ),
        (LLMResponse(stop_reason="tool_use"), "evaluator", "invalid_output", "no_tool_call"),
        (
            LLMResponse(payload_error="bad json", payload_error_kind="invalid_json", raw="{x"),
            "evaluator",
            "invalid_output",
            "invalid_json",
        ),
    ],
)
def test_judge_payload_classifies_every_unusable_response(resp, cls, kind, sub):
    with pytest.raises(EvalFailure) as info:
        judge_payload(resp)
    f = info.value
    assert (f.failure_class.value, f.kind, f.subkind) == (cls, kind, sub)
    assert f.raw == resp.raw


def test_a_truncated_response_is_a_failure_even_when_it_carries_a_partial_payload():
    with pytest.raises(EvalFailure, match="truncated"):
        judge_payload(LLMResponse(payload={"a": 1}, stop_reason="max_tokens"))


def test_build_request_is_leak_free_and_forces_the_judge_tool():
    r = build_request("P", "OUT", "REF", None, RUBRIC, temperature=0.1, timeout_s=9)
    assert r.role == "evaluator" and r.tool.name == TOOL_NAME and r.temperature == 0.1
    assert "OUT" in r.user and "REF" in r.user and "P" in r.user
    assert "run" not in r.user.lower().split("evaluate")[0]  # nothing about runs/targets


# --- legacy adapters --------------------------------------------------------------------------


def test_legacy_bedrock_judge_errors_carry_the_classified_failure():
    fake = FakeConverse(code_error("ThrottlingException", 429))
    judge = BedrockJudge("m", client=fake)
    with pytest.raises(JudgeError) as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert not isinstance(info.value, JudgeTimeoutError)
    assert (info.value.failure.failure_class, info.value.failure.kind) == (
        FailureClass.INFRA,
        "rate_limited",
    )
    fake.result = converse(judged(a=1, b=1), stop="max_tokens")
    with pytest.raises(JudgeOutputError) as out:
        judge.judge("p", "o", None, None, RUBRIC)
    assert out.value.kind == "truncated" and out.value.failure.kind == "truncated"
    assert out.value.raw["message"]["content"]  # evidence kept


def test_legacy_openai_judge_sees_the_same_semantics_as_the_client():
    fake = FakeOpenAI(chat({"a": 1}, finish="content_filter"))
    judge = BedrockOpenAIJudge("m", client=fake)
    with pytest.raises(JudgeOutputError, match="content filter") as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert info.value.kind == "refused"
    assert (judge.provider, judge.model) == ("bedrock-openai", "m")
