"""The reproducibility snapshot: what it records, what it never records, and which differences
between two runs are confounders."""

import json
import re

import pytest
from conftest import judged

from evalkit import EvaluatorSpec, Rubric
from evalkit.compare import ComparisonError, compare
from evalkit.envsnapshot import environment_differences, git_commit, runtime_facts
from evalkit.evaluators import llm_judge_spec
from evalkit.llm import LLMResponse, Usage
from evalkit.targets import ModelTarget, PrecomputedTarget

RUBRIC = Rubric.from_dict({"quality": "Good?"})
FAST = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0}}
PRICING = {
    "version": "p-1",
    "models": [
        {"provider": "fake", "model": "judge-1", "input_per_mtok": 1.0, "output_per_mtok": 2.0}
    ],
}


class Judge:
    provider, model = "fake", "judge-1"

    def __init__(self, endpoint="us-east-1"):
        self.endpoint = endpoint

    def call(self, req):
        return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=Usage(1, 1))


def dataset(kit, n=6):
    kit.datasets.import_cases(
        "qa", [{"case_key": f"c{i}", "prompt": f"q{i}", "output": f"o{i}"} for i in range(n)]
    )


def run_with(kit, judge=None, policy=None, execute=True):
    judge = judge or Judge()
    run = kit.controller.create(
        "qa",
        target=PrecomputedTarget(),
        evaluators=[llm_judge_spec("quality", RUBRIC, judge)],
        policy={**FAST, **(policy or {})},
    )
    if execute:
        kit.controller.execute(run.id, clients=[judge])
    return kit.runs.get(run.id)


# ---- what is recorded ---------------------------------------------------------------------------


def test_the_frozen_environment_records_what_defines_the_measurement(kit):
    dataset(kit)
    policy = {
        "pricing": PRICING,
        "cache": {"mode": "readwrite"},
        "budget": {"max_tokens": 5000},
        "retry": {**FAST["retry"], "validation_retries": 2, "max_attempts": 3},
    }
    run = run_with(kit, policy=policy, execute=False)
    env = run.environment
    assert env["schema"] == 2
    assert env["evalkit_version"] and env["python"] and env["python_implementation"]
    assert env["platform"] and env["sqlite"] and isinstance(env["sdk"], dict)
    assert env["evalkit_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", env["evalkit_commit"])
    version = kit.datasets.resolve("qa")
    assert env["dataset"] == {
        "ref": version.ref,
        "version_id": version.id,
        "content_hash": version.content_hash,
        "case_count": 6,
    }
    (ev,) = env["evaluators"]
    spec = run.config.evaluators[0]
    assert ev["key"] == spec.key and (ev["provider"], ev["model"]) == ("fake", "judge-1")
    assert ev["temperature"] == 0.0 and ev["max_tokens"] == spec.params["max_tokens"]
    assert ev["prompt_version"] == spec.params["prompt_version"] and ev["version"] == 1
    assert ev["scoring_version"] == spec.params["scoring_version"]
    assert ev["rubric_content_hash"] == spec.params["rubric_content_hash"]
    assert env["scoring_version"] == run.config.scoring_version
    assert env["retry"]["validation_retries"] == 2 and env["retry"]["max_attempts"] == 3
    assert env["request_timeout_s"] == 2.0 and env["unit_deadline_s"] == 6.0
    assert env["pricing"]["version"] == "p-1" and env["pricing"]["models"] == 1
    assert len(env["pricing"]["sha256"]) == 64
    assert env["cache"]["mode"] == "readwrite" and env["budget"] == {"max_tokens": 5000}
    assert env["target"]["kind"] == "precomputed" and "preflight" in env


def test_a_model_target_records_its_provider_model_and_generation_settings(kit):
    dataset(kit, 2)

    class Gen:
        provider, model = "fake", "gen-1"

        def call(self, req):
            return LLMResponse(text="x", stop_reason="end_turn")

    target = ModelTarget(Gen(), "{prompt}", temperature=0.3, max_tokens=77)
    run = kit.controller.create(
        "qa",
        target=target,
        evaluators=[EvaluatorSpec(kind="regex", name="r", params={"pattern": "x"})],
        policy=FAST,
    )
    t = run.environment["target"]
    assert (t["provider"], t["model"], t["temperature"], t["max_tokens"]) == (
        "fake", "gen-1", 0.3, 77,
    )  # fmt: skip
    assert len(t["identity_sha256"]) == 64


def test_the_environment_is_frozen_with_the_run(kit, store):
    dataset(kit)
    run = run_with(kit, execute=False)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="frozen|illegal run status"):
        store._conn.execute("UPDATE runs SET environment_json = '{}' WHERE id = ?", (run.id,))
    store._conn.rollback()


def test_each_execution_records_the_runtime_and_endpoints_it_used(kit):
    dataset(kit)
    run = run_with(kit, Judge("eu-west-1"))
    (execution,) = kit.store.list_executions(run.id)
    env = execution["environment"]
    assert env["endpoints"] == {"fake/judge-1": "eu-west-1"}
    assert env["python"] == runtime_facts()["python"] and env["cache_mode"] == "off"


def test_the_git_commit_is_a_sha_or_absent():
    c = git_commit()
    assert c is None or re.fullmatch(r"[0-9a-f]{40}", c)


def test_a_copy_outside_an_evalkit_checkout_reports_no_commit(tmp_path, monkeypatch):
    """A package inside some other project's repository must not report that repository's commit."""
    import shutil
    import subprocess
    import sys

    repo = tmp_path / "project"
    pkg = repo / ".venv" / "lib" / "evalkit"
    pkg.mkdir(parents=True)
    from evalkit import envsnapshot

    shutil.copy(envsnapshot.__file__, pkg / "envsnapshot.py")
    for cmd in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit",
                                 "-q", "--allow-empty", "-m", "x"]):  # fmt: skip
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    code = (
        "import sys, importlib.util;"
        "sys.modules.pop('evalkit', None);"
        f"spec = importlib.util.spec_from_file_location('es', {str(pkg / 'envsnapshot.py')!r});"
        "m = importlib.util.module_from_spec(spec);"
        "sys.modules['es'] = m; spec.loader.exec_module(m); print(m.git_commit())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "None"


# ---- what is never recorded ---------------------------------------------------------------------


def test_no_secret_reaches_the_environment(kit, monkeypatch):
    secrets = {
        "AWS_SECRET_ACCESS_KEY": "SECRETKEY-aws-9f8e7d",
        "AWS_SESSION_TOKEN": "SECRETTOKEN-session-1a2b",
        "EVALKIT_JUDGE_API_KEY": "SECRETAPIKEY-judge-3c4d",
        "EVALKIT_TARGET_API_KEY": "SECRETAPIKEY-target-5e6f",
        "OPENAI_API_KEY": "SECRETOPENAI-7a8b",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    dataset(kit)
    run = run_with(kit, policy={"pricing": PRICING})
    stored = json.dumps(run.environment) + json.dumps(kit.store.list_executions(run.id))
    stored += json.dumps(run.config.model_dump(mode="json"))
    for value in secrets.values():
        assert value not in stored


def test_a_client_reports_only_the_host_of_its_base_url():
    pytest.importorskip("openai")
    from evalkit.bedrock_openai import BedrockOpenAIClient

    client = BedrockOpenAIClient(
        "m", api_key="SECRET", base_url="https://user:hunter2@host.example.com:8443/v1?key=SECRET",
        client=object(),
    )  # fmt: skip
    assert client.endpoint == "host.example.com"


# ---- comparing environments ---------------------------------------------------------------------


def test_different_endpoints_are_a_confounder(kit):
    dataset(kit)
    a = run_with(kit, Judge("us-east-1"))
    b = run_with(kit, Judge("eu-west-1"))
    with pytest.raises(ComparisonError, match="endpoint"):
        compare(kit, a.id, b.id)
    cmp = compare(kit, a.id, b.id, allow_confounders=True)
    assert [c.kind for c in cmp.confounders] == ["environment.endpoint"] and cmp.confounded
    assert "eu-west-1" in cmp.confounders[0].detail


def test_the_same_endpoint_is_not_a_confounder(kit):
    dataset(kit)
    a, b = run_with(kit), run_with(kit)
    cmp = compare(kit, a.id, b.id)
    assert not cmp.confounders


def test_a_different_validation_retry_setting_is_a_confounder(kit):
    dataset(kit)
    a = run_with(kit, policy={"retry": {**FAST["retry"], "validation_retries": 0}})
    b = run_with(kit, policy={"retry": {**FAST["retry"], "validation_retries": 2}})
    with pytest.raises(ComparisonError, match="retried 0 vs 2"):
        compare(kit, a.id, b.id)


def test_timeouts_versions_and_prices_are_information_not_confounders(kit, store):
    dataset(kit)
    a = run_with(kit, policy={"pricing": PRICING})
    other = {**PRICING, "version": "p-2"}
    b = run_with(kit, policy={"pricing": other, "retry": {**FAST["retry"], "timeout_s": 5.0}})
    store._conn.execute("DROP TRIGGER runs_config_immutable")
    store._conn.execute("DROP TRIGGER runs_legal_transition")
    env = dict(b.environment)
    env["python"], env["sdk"] = "3.99.0", {"boto3": "9.9.9"}
    store._conn.execute(
        "UPDATE runs SET environment_json = ? WHERE id = ?", (json.dumps(env), b.id)
    )
    store._conn.commit()
    cmp = compare(kit, a.id, b.id)
    assert not cmp.confounders
    text = " | ".join(cmp.informational)
    assert "request timeout differs" in text and "Python differs" in text
    assert "price tables differ" in text


def test_a_run_without_a_snapshot_is_unknown_not_different(kit, store):
    dataset(kit)
    a = run_with(kit)
    b = run_with(kit)
    store._conn.execute("DROP TRIGGER runs_config_immutable")
    store._conn.execute("DROP TRIGGER runs_legal_transition")
    store._conn.execute("UPDATE runs SET environment_json = '{}' WHERE id = ?", (a.id,))
    store._conn.commit()
    cmp = compare(kit, a.id, b.id)
    assert not cmp.confounders
    assert any("snapshot is missing" in note for note in cmp.informational)


def test_environment_differences_directly():
    base = {"schema": 2, "retry": {"validation_retries": 1}}
    conf, notes = environment_differences(
        base,
        {"schema": 2, "retry": {"validation_retries": 1}},
        [{"endpoints": {"p/m": "a"}}],
        [{"endpoints": {"p/m": "a"}}],
    )
    assert conf == [] and notes == []
    conf, _ = environment_differences(
        base, base, [{"endpoints": {"p/m": "a"}}], [{"endpoints": {"p/m": "b"}}]
    )
    assert [k for k, _ in conf] == ["environment.endpoint"]
    _, notes = environment_differences(base, base, [{"endpoints": {"p/m": "a"}}], [{}])
    assert notes == ["the endpoint is recorded for only one of the runs"]
