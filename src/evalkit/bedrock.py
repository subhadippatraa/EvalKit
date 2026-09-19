"""AWS Bedrock judge (Converse API). The only module that imports boto3/botocore."""

from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ConnectTimeoutError, ReadTimeoutError

from evalkit.errors import ConfigError, JudgeError, JudgeOutputError, JudgeTimeoutError
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


class BedrockJudge:
    provider = "bedrock"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        timeout: float = 60.0,
        region: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.temperature = temperature
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

    def judge(
        self, prompt: str, model_output: str, reference_output: str | None, rubric: Rubric
    ) -> dict[str, Any]:
        try:
            response = self._client.converse(
                modelId=self.model,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": render_prompt(prompt, model_output, reference_output, rubric)}
                        ],
                    }
                ],
                toolConfig={
                    "tools": [
                        {
                            "toolSpec": {
                                "name": TOOL_NAME,
                                "description": TOOL_DESCRIPTION,
                                "inputSchema": {"json": output_schema(rubric)},
                            }
                        }
                    ],
                    "toolChoice": {"tool": {"name": TOOL_NAME}},
                },
                inferenceConfig={"temperature": self.temperature, "maxTokens": MAX_TOKENS},
            )
        except (ReadTimeoutError, ConnectTimeoutError) as e:
            raise JudgeTimeoutError(f"Bedrock request timed out after {self.timeout}s: {e}") from e
        except ClientError as e:
            err = e.response.get("Error", {})
            code, message = err.get("Code", "Unknown"), err.get("Message", str(e))
            if code == "ModelTimeoutException":
                raise JudgeTimeoutError(f"Bedrock {code}: {message}") from e
            raise JudgeError(f"Bedrock {code}: {message}") from e
        except BotoCoreError as e:
            raise JudgeError(f"Bedrock request failed: {e}") from e

        if response.get("stopReason") == "max_tokens":
            raise JudgeOutputError("judge response was truncated (stopReason=max_tokens)")
        for block in response.get("output", {}).get("message", {}).get("content", []):
            tool_use = block.get("toolUse")
            if tool_use and tool_use.get("name") == TOOL_NAME:
                return tool_use.get("input")
        raise JudgeOutputError(f"judge response contained no {TOOL_NAME} tool call")
