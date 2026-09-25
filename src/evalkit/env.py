"""LLM clients from environment variables (the operator's configuration; secrets never persist).

    EVALKIT_JUDGE_PROVIDER   bedrock (default) | bedrock-openai
    EVALKIT_JUDGE_MODEL      required
    EVALKIT_JUDGE_TIMEOUT    seconds (default 60)
    EVALKIT_JUDGE_API_KEY    bedrock-openai only
    EVALKIT_JUDGE_BASE_URL   bedrock-openai only

`prefix="EVALKIT_TARGET"` reads the same names for a model *target*. SDK imports stay lazy so the
core never needs boto3 or openai to be importable for anything else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from evalkit.errors import ConfigError
from evalkit.llm import LLMClient


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not 0 < value < float("inf"):
        raise ConfigError(f"{name} must be positive and finite, got {raw!r}")
    return value


def client_from_env(
    prefix: str = "EVALKIT_JUDGE", env: Mapping[str, str] | None = None
) -> LLMClient:
    env = os.environ if env is None else env
    provider = env.get(f"{prefix}_PROVIDER") or "bedrock"
    model = env.get(f"{prefix}_MODEL")
    if not model:
        raise ConfigError(f"{prefix}_MODEL is required")
    timeout = _float(env, f"{prefix}_TIMEOUT", 60.0)
    if provider == "bedrock":
        from evalkit.bedrock import BedrockClient

        return BedrockClient(model=model, timeout=timeout)
    if provider == "bedrock-openai":
        from evalkit.bedrock_openai import DEFAULT_BASE_URL, BedrockOpenAIClient

        api_key = env.get(f"{prefix}_API_KEY")
        if not api_key:
            raise ConfigError(f"{prefix}_API_KEY is required for provider 'bedrock-openai'")
        return BedrockOpenAIClient(
            model=model,
            api_key=api_key,
            base_url=env.get(f"{prefix}_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout,
        )
    raise ConfigError(
        f"unsupported {prefix}_PROVIDER {provider!r}; supported: 'bedrock', 'bedrock-openai'"
    )
