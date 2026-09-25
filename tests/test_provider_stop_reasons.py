"""Provider stop reasons other than max_tokens used to surface as "no tool call". Each is now a
distinct, recorded kind, with the partial provider output kept as evidence."""

import json

import pytest
from conftest import judged
from test_bedrock import converse_response, make_judge
from test_bedrock_openai import chat_response, tool_call
from test_bedrock_openai import make_judge as make_openai_judge

from evalkit import JudgeOutputError, Rubric
from evalkit.judge import TOOL_NAME

RUBRIC = Rubric.from_dict({"a": "A?", "b": "B?"})


@pytest.mark.parametrize(
    ("stop_reason", "kind"),
    [
        ("max_tokens", "truncated"),
        ("guardrail_intervened", "refused"),
        ("content_filtered", "refused"),
        ("malformed_model_output", "malformed_output"),
        ("malformed_tool_use", "malformed_output"),
        ("model_context_window_exceeded", "context_window"),
    ],
)
def test_bedrock_stop_reasons_are_distinct_kinds_with_evidence(stop_reason, kind):
    # even if a (partial) tool call is present, these stop reasons mean "unusable"
    judge, _ = make_judge(converse_response(judged(a=5, b=5), stop_reason=stop_reason))
    with pytest.raises(JudgeOutputError, match=stop_reason) as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert info.value.kind == kind
    assert info.value.raw["message"]["content"]  # partial provider output kept for the attempt


@pytest.mark.parametrize("stop_reason", ["end_turn", "stop_sequence"])
def test_bedrock_no_tool_call_is_its_own_kind(stop_reason):
    response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "I refuse."}]}},
        "stopReason": stop_reason,
    }
    judge, _ = make_judge(response)
    with pytest.raises(JudgeOutputError, match="no submit_evaluation") as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert info.value.kind == "no_tool_call" and info.value.raw["message"]["content"][0]["text"]


def test_bedrock_normal_tool_use_is_unaffected():
    payload = judged(a=4, b=4)
    judge, _ = make_judge(converse_response(payload, stop_reason="tool_use"))
    assert judge.judge("p", "o", None, None, RUBRIC) == payload


def test_openai_finish_reasons_and_evidence():
    from types import SimpleNamespace

    def response(finish_reason, tool_calls=None, content="partial text"):
        message = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            model_dump=lambda mode="json": {"role": "assistant", "content": content},
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
        )

    for finish, kind in [("length", "truncated"), ("content_filter", "refused")]:
        judge, _ = make_openai_judge(response(finish))
        with pytest.raises(JudgeOutputError) as info:
            judge.judge("p", "o", None, None, RUBRIC)
        assert info.value.kind == kind and info.value.raw == {
            "role": "assistant",
            "content": "partial text",
        }

    judge, _ = make_openai_judge(response("stop"))
    with pytest.raises(JudgeOutputError) as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert info.value.kind == "no_tool_call"


@pytest.mark.parametrize(
    "arguments", ['{"a": NaN}', '{"a": Infinity}', '{"a": 1e999}', '{"a": 1, "a": 2}', "{not json"]
)
def test_openai_tool_arguments_are_parsed_strictly_and_kept_as_evidence(arguments):
    from types import SimpleNamespace

    call = SimpleNamespace(id="c", function=SimpleNamespace(name=TOOL_NAME, arguments=arguments))
    judge, _ = make_openai_judge(chat_response([call]))
    with pytest.raises(JudgeOutputError, match="not valid JSON") as info:
        judge.judge("p", "o", None, None, RUBRIC)
    assert info.value.kind == "invalid_json" and info.value.raw == arguments


def test_openai_valid_arguments_still_parse():
    judge, _ = make_openai_judge(chat_response([tool_call(judged(a=1, b=2))]))
    assert judge.judge("p", "o", None, None, RUBRIC) == judged(a=1, b=2)
    assert json.loads(json.dumps(judged(a=1, b=2)))  # sanity: fixture is plain JSON


def test_kind_and_raw_default_to_none_for_plain_errors():
    from evalkit import JudgeError

    e = JudgeError("x")
    assert e.kind is None and e.raw is None and e.evaluation_id is None
