import json
from types import SimpleNamespace

import pytest
from conftest import judged
from openai import APIConnectionError, APIError, APITimeoutError

from evalkit import JudgeError, JudgeOutputError, JudgeTimeoutError, Rubric
from evalkit.bedrock_openai import DEFAULT_BASE_URL, MAX_TOKENS, BedrockOpenAIJudge
from evalkit.judge import PROMPT_VERSION, TOOL_NAME, output_schema

RUBRIC = Rubric.from_dict({"a": "A?", "b": "B?"})


def tool_call(arguments, name=TOOL_NAME, call_id="c1"):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


def chat_response(tool_calls, finish_reason="tool_calls"):
    message = SimpleNamespace(role="assistant", content=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeChatCompletions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeOpenAIClient:
    def __init__(self, result):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(result))


def make_judge(result, **kwargs):
    client = FakeOpenAIClient(result)
    judge = BedrockOpenAIJudge("openai.gpt-oss-120b", client=client, **kwargs)
    return judge, client.chat.completions


def call(judge, reference=None):
    return judge.judge("Explain DI.", "DI is ...", reference, RUBRIC)


def test_returns_raw_tool_input_and_sends_forced_tool_request():
    payload = judged(a=4, b=5)
    judge, completions = make_judge(chat_response([tool_call(payload)]), temperature=0.2)
    assert call(judge, reference="gold answer") == payload

    (req,) = completions.calls
    assert req["model"] == "openai.gpt-oss-120b"
    assert req["temperature"] == 0.2 and req["max_tokens"] == MAX_TOKENS
    assert req["tool_choice"] == {"type": "function", "function": {"name": TOOL_NAME}}
    fn = req["tools"][0]["function"]
    assert fn["name"] == TOOL_NAME and fn["parameters"] == output_schema(RUBRIC)
    text = req["messages"][1]["content"]
    assert "Explain DI." in text and "DI is ..." in text and "gold answer" in text


def test_identity_attributes():
    judge, _ = make_judge(chat_response([]))
    assert (judge.provider, judge.model, judge.prompt_version) == (
        "bedrock-openai",
        "openai.gpt-oss-120b",
        PROMPT_VERSION,  # shared with BedrockJudge: same prompt template and tool schema
    )


def test_does_not_validate_payload():
    judge, _ = make_judge(chat_response([tool_call({"garbage": True})]))
    assert call(judge) == {"garbage": True}


@pytest.mark.parametrize(
    "response",
    [
        chat_response([]),  # no tool call at all
        chat_response([tool_call(judged(a=1, b=1), name="other_tool")]),
        chat_response([tool_call(judged(a=1, b=1))], finish_reason="length"),
        SimpleNamespace(choices=[]),  # no choices
    ],
)
def test_missing_or_truncated_payload_is_output_error(response):
    judge, _ = make_judge(response)
    with pytest.raises(JudgeOutputError):
        call(judge)


def test_invalid_json_arguments_is_output_error():
    bad_fn = SimpleNamespace(name=TOOL_NAME, arguments="{not json")
    bad_call = SimpleNamespace(id="c1", function=bad_fn)
    judge, _ = make_judge(chat_response([bad_call]))
    with pytest.raises(JudgeOutputError):
        call(judge)


def test_timeout():
    judge, completions = make_judge(APITimeoutError(request=None))
    with pytest.raises(JudgeTimeoutError):
        call(judge)
    assert len(completions.calls) == 1


def test_connection_error_is_judge_error_not_timeout():
    judge, _ = make_judge(APIConnectionError(request=None))
    with pytest.raises(JudgeError) as info:
        call(judge)
    assert not isinstance(info.value, JudgeTimeoutError)


def test_other_api_error_is_judge_error():
    class FakeAPIError(APIError):
        def __init__(self):
            self.message = "rate limited"
            self.request = None
            self.body = None

    judge, _ = make_judge(FakeAPIError())
    with pytest.raises(JudgeError) as info:
        call(judge)
    assert not isinstance(info.value, (JudgeTimeoutError, JudgeOutputError))


def test_real_client_has_sdk_retries_disabled_and_uses_mantle_base_url():
    # constructing an OpenAI client makes no network call
    judge = BedrockOpenAIJudge("m", api_key="k", timeout=9)
    assert judge._client.max_retries == 0
    assert str(judge._client.base_url).rstrip("/") == DEFAULT_BASE_URL.rstrip("/")
