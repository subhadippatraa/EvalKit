# ruff: noqa: F811  (the `cli` fixture is imported from test_cli_platform)
"""P2.1 through the CLI: pricing, cache, budgets, retry-failed, status, logging."""

import json
import os
import threading

from conftest import judged
from test_cli_platform import EM, JUDGE_SPEC, N, cli, create, write  # noqa: F401

from evalkit import EvalFailure, FailureClass, cli_platform
from evalkit.cli import main
from evalkit.llm import LLMResponse, Usage

PRICES = (
    'version = "cli-1"\n[[models]]\nprovider = "fake"\nmodel = "j"\n'
    "input_per_mtok = 1000.0\noutput_per_mtok = 2000.0\n"
)
FAST_POLICY = {"retry": {"base_s": 0.001, "cap_s": 0.002, "timeout_s": 2.0, "max_attempts": 1}}


class J:
    provider, model, endpoint = "fake", "j", "cli-region"

    def __init__(self, fail_at=()):
        self.calls = 0
        self.fail_at = set(fail_at)  # the sequence numbers of the calls that fail
        self.lock = threading.Lock()

    def call(self, req):
        with self.lock:
            self.calls += 1
            fail = self.calls in self.fail_at
        if fail:
            raise EvalFailure(
                FailureClass.INFRA, "provider_unavailable", "503 SECRET-PROVIDER-TEXT"
            )
        return LLMResponse(payload=judged(quality=5), stop_reason="tool_use", usage=Usage(50, 5))


def use(monkeypatch, judge):
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: judge)
    return judge


def judge_spec(policy=None):
    return {"evaluators": [JUDGE_SPEC], "policy": {**FAST_POLICY, **(policy or {})}}


def test_create_freezes_pricing_cache_and_budget_flags_into_the_policy(cli, monkeypatch):
    use(monkeypatch, J())
    price_file = write(cli, "prices.toml", PRICES)
    run = create(
        cli, judge_spec(), "--pricing", price_file, "--cache", "readwrite",
        "--max-tokens", "500000",
        "--max-cost-usd", "2.5",
    )  # fmt: skip
    _, shown, _ = cli("runs", "show", run)
    policy = shown["run"]["config"]["policy"]
    assert policy["pricing"]["version"] == "cli-1" and policy["cache"] == {"mode": "readwrite"}
    assert policy["budget"] == {"max_tokens": 500000, "max_cost_usd": 2.5}
    env = shown["run"]["environment"]
    assert env["pricing"]["version"] == "cli-1" and env["cache"]["mode"] == "readwrite"
    assert env["budget"]["max_cost_usd"] == 2.5 and env["evaluators"][0]["model"] == "j"


def test_a_usd_budget_without_prices_is_refused_before_anything_is_spent(cli, monkeypatch):
    judge = use(monkeypatch, J())
    code, out, err = cli(
        "runs", "create", "qa", "--config", write(cli, "s.json", judge_spec()),
        "--max-cost-usd", "1",
    )  # fmt: skip
    assert code == 1 and "price" in err and judge.calls == 0


def test_execute_reports_operations_on_stdout_and_a_summary_on_stderr(cli, monkeypatch, capsys):
    use(monkeypatch, J())
    run = create(cli, judge_spec(), "--pricing", write(cli, "p.toml", PRICES))
    code, out, err = cli("runs", "execute", run)
    assert code == 0
    ops = out["operations"]
    assert ops["cases"]["completed"] == N and ops["calls"]["provider_calls"] == N
    assert ops["tokens"]["total"] == N * 55 and ops["cost"]["usd_estimate"] > 0
    assert out["run_spend"]["calls"] == N and out["spend"]["cost_usd"] > 0
    assert f"{N} completed" in err and "cost" in err and "estimate" in err
    capsys.readouterr()
    assert main(["runs", "status", run, "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert f"{N} completed" in text and "coverage 100.0%" in text and "(estimate)" in text
    _, status, _ = cli("runs", "status", run)
    assert status["budget"]["state"] == "none" and status["run"]["status"] == "succeeded"


def test_quiet_silences_the_summary(cli, monkeypatch):
    use(monkeypatch, J())
    run = create(cli, judge_spec())
    _, _, err = cli("runs", "execute", run, "--quiet")
    assert err.strip() == ""


def test_the_cache_serves_a_second_run_and_can_be_inspected_and_cleared(cli, monkeypatch):
    judge = use(monkeypatch, J())
    spec = judge_spec()
    first = create(cli, spec, "--cache", "readwrite")
    cli("runs", "execute", first)
    assert judge.calls == N
    second = create(cli, spec, "--cache", "readwrite")
    _, out, _ = cli("runs", "execute", second)
    assert judge.calls == N and out["spend"]["cache_hits"] == N  # no provider call at all
    assert out["operations"]["cache"]["hit_rate"] == 1.0
    _, stats, _ = cli("cache", "stats")
    assert stats["entries"] == N and stats["by_model"][0]["model"] == "j"
    # replay from the cache, refusing to call the provider
    third = create(cli, spec, "--cache", "replay")
    code, out, _ = cli("runs", "execute", third)
    assert code == 0 and judge.calls == N
    _, cleared, _ = cli("cache", "clear")
    assert cleared["deleted"] == N and cli("cache", "stats")[1]["entries"] == 0
    fourth = create(cli, spec, "--cache", "replay")
    code, out, _ = cli("runs", "execute", fourth)
    assert code == 1 and out["stop_reason"] == "cache_miss" and judge.calls == N


def test_clearing_only_old_entries(cli, monkeypatch):
    use(monkeypatch, J())
    run = create(cli, judge_spec(), "--cache", "readwrite")
    cli("runs", "execute", run)
    assert cli("cache", "clear", "--older-than-days", "1")[1]["deleted"] == 0
    assert cli("cache", "stats")[1]["entries"] == N
    assert cli("cache", "clear", "--older-than-days", "0")[1]["deleted"] == N


def test_a_token_budget_stops_the_run_and_a_raised_one_finishes_it(cli, monkeypatch):
    judge = use(monkeypatch, J())
    run = create(cli, judge_spec(), "--max-tokens", "9000")
    code, out, _ = cli("runs", "execute", run)
    assert code == 1 and out["run"]["status"] == "partial" and out["stop_reason"] == "budget"
    assert out["operations"]["budget"]["state"] == "exhausted"
    made = judge.calls
    assert 0 < made < N
    code, out, _ = cli("runs", "resume", run, "--max-tokens", "100000000")
    assert code == 0 and out["run"]["status"] == "succeeded" and judge.calls == N
    assert out["run_spend"]["input_tokens"] == N * 50  # per-run totals across executions


def test_retry_failed_through_the_cli_with_history(cli, monkeypatch):
    judge = use(monkeypatch, J(fail_at=(3, 7, 9)))
    run = create(cli, judge_spec())
    code, out, _ = cli("runs", "execute", run)
    assert code == 0 and out["counts"]["evaluator_results"]
    failed_before = sum(v.get("failed", 0) for v in out["counts"]["evaluator_results"].values())
    assert failed_before >= 1
    calls_before = judge.calls
    _, none, _ = cli("runs", "failures", run, "--history")
    assert none == []
    code, out, err = cli("runs", "execute", run, "--retry-failed")
    assert code == 0 and out["run"]["status"] == "succeeded"
    assert out["retry"]["reopened_evaluators"] == failed_before
    assert judge.calls - calls_before == failed_before  # only the failures were called again
    assert "retried" in err
    _, hist, _ = cli("runs", "failures", run, "--history")
    assert len(hist) == failed_before and hist[0]["failure_kind"] == "provider_unavailable"
    assert "SECRET-PROVIDER-TEXT" in json.dumps(hist)  # stored evidence, never logged (below)
    _, shown, _ = cli("runs", "show", run)
    assert shown["retry"]["units_retried"] == failed_before
    assert [e["retry_failed"] for e in shown["executions"]] == [0, 1]
    _, current, _ = cli("runs", "failures", run)
    assert current == []
    assert cli("runs", "verify", run)[0] == 0


def test_retry_failed_on_a_clean_run_changes_nothing(cli, monkeypatch):
    judge = use(monkeypatch, J())
    run = create(cli, judge_spec())
    cli("runs", "execute", run)
    calls = judge.calls
    code, out, _ = cli("runs", "execute", run, "--retry-failed")
    assert code == 0 and out["units"] == 0 and judge.calls == calls
    code, out, err = cli("runs", "execute", run)
    assert code == 1 and "already succeeded" in err


def test_logging_flags_emit_structured_events_without_content(cli, monkeypatch):
    use(monkeypatch, J(fail_at=(3,)))
    run = create(cli, judge_spec())
    code, out, err = cli("--log-level", "debug", "runs", "execute", run, "--quiet")
    lines = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    names = {r["event"] for r in lines}
    assert {"run.started", "run.finished", "provider.call", "evaluator.failed"} <= names
    assert all(r.get("run_id") in (None, run) for r in lines)
    assert "SECRET-PROVIDER-TEXT" not in err and "Good?" not in err and '"q3"' not in err


def test_logging_can_go_to_a_file_and_in_text_form(cli, monkeypatch, tmp_path):
    use(monkeypatch, J())
    run = create(cli, judge_spec())
    log = tmp_path / "events.log"
    code, _, err = cli(
        "--log-level", "info", "--log-format", "text", "--log-file", str(log),
        "runs", "execute", run, "--quiet",
    )  # fmt: skip
    text = log.read_text()
    assert "run.started" in text and f"run_id={run}" in text and "run.finished" in text
    assert (log.stat().st_mode & 0o777) == 0o600 and "run.started" not in err


def test_logging_from_the_environment_and_bad_settings(cli, monkeypatch):
    use(monkeypatch, J())
    run = create(cli, judge_spec())
    monkeypatch.setenv("EVALKIT_LOG_LEVEL", "info")
    _, _, err = cli("runs", "execute", run, "--quiet")
    assert '"event": "run.started"' in err
    monkeypatch.setenv("EVALKIT_LOG_LEVEL", "LOUD")
    code, _, err = cli("runs", "list")
    assert code == 2 and "log" in err


def test_the_ambient_environment_is_not_stored_in_the_run(cli, monkeypatch):
    use(monkeypatch, J())
    monkeypatch.setenv("EVALKIT_JUDGE_API_KEY", "SECRETAPIKEY-cli-1234")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRETAWS-cli-5678")
    run = create(cli, judge_spec())
    cli("runs", "execute", run, "--quiet")
    _, shown, _ = cli("runs", "show", run, "--summary")
    blob = json.dumps(shown)
    assert "SECRETAPIKEY" not in blob and "SECRETAWS" not in blob
    assert os.environ["EVALKIT_DB_PATH"]
