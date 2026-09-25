"""Clients from environment variables."""

import pytest

from evalkit import ConfigError
from evalkit.env import client_from_env


def test_bedrock_is_the_default_provider_and_the_model_is_required():
    with pytest.raises(ConfigError, match="EVALKIT_JUDGE_MODEL is required"):
        client_from_env(env={})
    c = client_from_env(env={"EVALKIT_JUDGE_MODEL": "m1", "AWS_DEFAULT_REGION": "us-east-1"})
    assert (c.provider, c.model, c.timeout) == ("bedrock", "m1", 60.0)


def test_a_prefix_selects_a_separate_configuration_for_a_target():
    env = {"EVALKIT_JUDGE_MODEL": "judge", "EVALKIT_TARGET_MODEL": "target",
           "EVALKIT_TARGET_TIMEOUT": "12", "AWS_DEFAULT_REGION": "us-east-1"}  # fmt: skip
    assert client_from_env("EVALKIT_TARGET", env).model == "target"
    assert client_from_env("EVALKIT_TARGET", env).timeout == 12.0
    assert client_from_env(env=env).model == "judge"


def test_the_openai_compatible_provider_needs_a_key_and_disables_sdk_retries():
    env = {"EVALKIT_JUDGE_PROVIDER": "bedrock-openai", "EVALKIT_JUDGE_MODEL": "m"}
    with pytest.raises(ConfigError, match="API_KEY is required"):
        client_from_env(env=env)
    c = client_from_env(
        env={**env, "EVALKIT_JUDGE_API_KEY": "k", "EVALKIT_JUDGE_BASE_URL": "https://x.example/v1"}
    )
    assert c.provider == "bedrock-openai" and c._client.max_retries == 0
    assert str(c._client.base_url).startswith("https://x.example")


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "inf", "nan"])
def test_a_bad_timeout_is_refused(bad):
    with pytest.raises(ConfigError, match="TIMEOUT"):
        client_from_env(env={"EVALKIT_JUDGE_MODEL": "m", "EVALKIT_JUDGE_TIMEOUT": bad})


def test_an_unknown_provider_is_refused():
    with pytest.raises(ConfigError, match="unsupported"):
        client_from_env(
            env={"EVALKIT_JUDGE_MODEL": "m", "EVALKIT_JUDGE_PROVIDER": "carrier-pigeon"}
        )
