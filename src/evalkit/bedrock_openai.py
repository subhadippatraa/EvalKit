"""Bedrock via its OpenAI-compatible endpoint (chat completions + function calling).

Added on top of the original v1 architecture, which specified only `bedrock.py`
(boto3 Converse). Kept isolated the same way: this is the only module that imports
`openai`, reuses the shared prompt/schema/version from `evalkit.judge`, and the core
still depends only on the `Judge` Protocol. Requires the `bedrock-openai` extra
(`pip install evalkit[bedrock-openai]`).
"""

import json
from typing import Any

from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import OpenAI

from evalkit.errors import JudgeError, JudgeOutputError, JudgeTimeoutError
from evalkit.judge import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    output_schema,
    render_prompt,
)
from evalkit.models import Rubric

# ponytail: fixed output budget; very large rubrics may truncate -> JudgeOutputError
MAX_TOKENS = 4096
DEFAULT_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/v1"


class BedrockOpenAIJudge:
    provider = "bedrock-openai"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 60.0,
        client: Any = None,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # evalkit owns retries (OD-1); no SDK-level retries here either
            default_headers={"OpenAI-Project": "default"},
        )

    def judge(
        self,
        prompt: str,
        model_output: str,
        reference_output: str | None,
        context: str | None,
        rubric: Rubric,
    ) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": render_prompt(
                            prompt, model_output, reference_output, context, rubric
                        ),
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "description": TOOL_DESCRIPTION,
                            "parameters": output_schema(rubric),
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                temperature=self.temperature,
                max_tokens=MAX_TOKENS,
            )
        except OpenAIAPITimeoutError as e:
            raise JudgeTimeoutError(f"Bedrock (OpenAI-compatible) request timed out: {e}") from e
        except OpenAIAPIConnectionError as e:
            raise JudgeError(f"Bedrock (OpenAI-compatible) connection failed: {e}") from e
        except OpenAIAPIError as e:
            raise JudgeError(f"Bedrock (OpenAI-compatible) API error: {e}") from e

        if not response.choices:
            raise JudgeOutputError("judge response contained no choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise JudgeOutputError("judge response was truncated (finish_reason=length)")

        for call in choice.message.tool_calls or []:
            if call.function.name == TOOL_NAME:
                try:
                    return json.loads(call.function.arguments)
                except json.JSONDecodeError as e:
                    raise JudgeOutputError(f"judge tool arguments were not valid JSON: {e}") from e
        raise JudgeOutputError(f"judge response contained no {TOOL_NAME} tool call")
