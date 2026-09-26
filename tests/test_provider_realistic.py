"""P1.1 provider behaviour, against stubs that are as realistic as can be without a live provider.

Bedrock: botocore's own `Stubber` on a real `bedrock-runtime` client. It validates every request
against the AWS service model (so a malformed Converse request fails here) and every response
shape, and raises `ClientError`s exactly as boto3 does. OpenAI-compatible: the SDK's real
`APIStatusError` type with the body shape the API documents.

What this does NOT establish: what a live Bedrock account really answers for an over-long input.
The oversize recognition below is a documented heuristic on the message text and error code; it is
unverified against a live provider (see docs/EVALUATION-METHODOLOGY.md)."""

from types import SimpleNamespace

import boto3
import pytest
from botocore.config import Config
from botocore.stub import Stubber
from conftest import judged
from openai import APIStatusError

from evalkit import EvalFailure, Rubric
from evalkit.bedrock import BedrockClient
from evalkit.bedrock_openai import BedrockOpenAIClient
from evalkit.judge_eval import build_request
from evalkit.llm import LLMRequest

RUBRIC = Rubric.from_dict({"quality": "Is it good?"})
_REAL_BOTO_CLIENT = boto3.client  # captured before any test patches it


def stubbed_client():
    real = _REAL_BOTO_CLIENT(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        config=Config(retries={"total_max_attempts": 1}),
    )
    return real, Stubber(real)


def judge_request():
    return build_request("q", "a", "ref", "ctx", RUBRIC, temperature=0.0, timeout_s=10)


def target_request():
    return LLMRequest(system="s", user="u", max_tokens=100, role="target")


TOOL_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "submit_evaluation",
                        "input": judged(quality=5),
                    }
                }
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19},
    "metrics": {"latencyMs": 321},
}


def call_with_error(code, message, status, request):
    real, stub = stubbed_client()
    stub.add_client_error(
        "converse", service_error_code=code, service_message=message, http_status_code=status
    )
    with stub:
        with pytest.raises(EvalFailure) as info:
            BedrockClient("anthropic.claude-x", client=real).call(request)
    return info.value


# --- request and response shapes are the real service model's -----------------------------------


def test_a_judge_request_and_a_real_shaped_response_round_trip():
    real, stub = stubbed_client()
    stub.add_response("converse", TOOL_RESPONSE)
    with stub:  # the Stubber rejects any request that is not valid for the Converse operation
        resp = BedrockClient("anthropic.claude-x", client=real).call(judge_request())
    assert resp.payload == judged(quality=5)
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (12, 7)
    assert resp.provider_latency_ms == 321 and resp.stop_reason == "tool_use"


def test_a_target_request_without_a_tool_is_a_valid_converse_request():
    real, stub = stubbed_client()
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
            "metrics": {"latencyMs": 5},
        },
    )
    with stub:
        resp = BedrockClient("m", client=real).call(target_request())
    assert resp.text == "hello" and resp.stop_reason == "end_turn"


@pytest.mark.parametrize(
    "stop,expected",
    [
        ("max_tokens", "max_tokens"),
        ("guardrail_intervened", "guardrail_intervened"),
        ("content_filtered", "content_filtered"),
        ("malformed_model_output", "malformed"),
        ("malformed_tool_use", "malformed"),
        ("model_context_window_exceeded", "model_context_window_exceeded"),
    ],
)
def test_every_documented_stop_reason_is_understood(stop, expected):
    real, stub = stubbed_client()
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "partial"}]}},
            "stopReason": stop,
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        },
    )
    with stub:
        assert BedrockClient("m", client=real).call(judge_request()).stop_reason == expected


# --- one bad input is not a broken configuration ------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Input is too long for requested model.",
        "The input token count exceeds the maximum context length for this model.",
        "Too many input tokens: 250000 > 200000",
        "prompt is too long: 210000 tokens > 200000 maximum",
    ],
)
@pytest.mark.parametrize("role", ["evaluator", "target"])
def test_an_over_long_input_is_the_cases_problem_not_the_configurations(message, role):
    request = judge_request() if role == "evaluator" else target_request()
    f = call_with_error("ValidationException", message, 400, request)
    assert (f.failure_class.value, f.kind) == ("input", "oversize")
    assert not f.systemic and not f.retryable
    assert f.http_status == 400


@pytest.mark.parametrize(
    "message",
    [
        "The provided model identifier is invalid.",
        "This model doesn't support the toolConfig.toolChoice field.",
        "Malformed input request: extraneous key [foo] is not permitted",
    ],
)
def test_other_rejections_stay_systemic_but_are_stopped_only_on_repetition(message):
    f = call_with_error("ValidationException", message, 400, judge_request())
    assert (f.failure_class.value, f.kind) == ("evaluator", "bad_request") and f.systemic
    t = call_with_error("ValidationException", message, 400, target_request())
    assert (t.failure_class.value, t.kind) == ("input", "bad_config") and t.systemic


@pytest.mark.parametrize(
    "code,status,cls,kind",
    [
        ("ThrottlingException", 429, "infrastructure", "rate_limited"),
        ("ServiceUnavailableException", 503, "infrastructure", "provider_unavailable"),
        ("ModelTimeoutException", 408, "infrastructure", "timeout"),
        ("AccessDeniedException", 403, "infrastructure", "auth"),
        ("ResourceNotFoundException", 404, "evaluator", "bad_request"),
    ],
)
def test_the_documented_bedrock_errors_map_to_the_design_table(code, status, cls, kind):
    f = call_with_error(code, "message", status, judge_request())
    assert (f.failure_class.value, f.kind) == (cls, kind)


# --- the OpenAI-compatible gateway ---------------------------------------------------------------


def status_error(status, message="nope", body=None):
    response = SimpleNamespace(status_code=status, headers={}, request=None)
    return APIStatusError(message, response=response, body=body)


class Recorder:
    """A stand-in for `OpenAI()` that records what `chat.completions.create` was called with."""

    def __init__(self, error=None):
        self.kwargs, self.error = [], error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(content="hi", tool_calls=None, model_dump=lambda mode: {})
        return SimpleNamespace(
            id="req-1",
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )


@pytest.mark.parametrize(
    "error",
    [
        status_error(
            400,
            "This model's maximum context length is 8192 tokens.",
            {"code": "context_length_exceeded", "type": "invalid_request_error"},
        ),
        status_error(400, "Input is too long for requested model."),
    ],
)
def test_an_over_long_prompt_through_the_gateway_is_oversize_not_bad_config(error):
    client = BedrockOpenAIClient("m", client=Recorder(error))
    with pytest.raises(EvalFailure) as info:
        client.call(judge_request())
    assert (info.value.failure_class.value, info.value.kind) == ("input", "oversize")


def test_a_plain_400_through_the_gateway_is_still_a_rejected_request():
    client = BedrockOpenAIClient("m", client=Recorder(status_error(400, "unknown model 'typo'")))
    with pytest.raises(EvalFailure) as info:
        client.call(judge_request())
    assert (info.value.kind, info.value.systemic) == ("bad_request", True)


# --- the run's timeout is the request's timeout --------------------------------------------------


def test_the_openai_client_passes_the_requests_timeout_to_the_sdk():
    recorder = Recorder()
    client = BedrockOpenAIClient("m", client=recorder, timeout=60)
    client.call(judge_request().__class__(**{**judge_request().__dict__, "timeout_s": 7.5}))
    assert recorder.kwargs[0]["timeout"] == 7.5


def test_bedrock_builds_a_client_with_the_requests_timeout_and_reuses_it(monkeypatch):
    built = []

    def fake_boto_client(service, region_name=None, config=None, **kw):
        built.append((service, config.read_timeout, config.connect_timeout))
        real, stub = stubbed_client()
        stub.activate()
        for _ in range(4):
            stub.add_response("converse", TOOL_RESPONSE)
        return real

    monkeypatch.setattr("evalkit.bedrock.boto3.client", fake_boto_client)
    client = BedrockClient("anthropic.claude-x", timeout=60)  # the default client, 60 s
    assert built == [("bedrock-runtime", 60, 60)]
    req = judge_request()
    same = LLMRequest(**{**req.__dict__, "timeout_s": 60})
    short = LLMRequest(**{**req.__dict__, "timeout_s": 7})
    client.call(same)
    assert len(built) == 1  # the default timeout needs no new client
    client.call(short)
    client.call(short)
    assert built[1:] == [("bedrock-runtime", 7, 7)]  # one client for 7 s, built once


def test_an_injected_bedrock_client_is_used_as_is_for_any_timeout():
    real, stub = stubbed_client()
    stub.add_response("converse", TOOL_RESPONSE)
    with stub:
        client = BedrockClient("m", client=real)
        req = LLMRequest(**{**judge_request().__dict__, "timeout_s": 3})
        assert client.call(req).stop_reason == "tool_use"
