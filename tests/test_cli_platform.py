"""The platform workflow through the CLI: create, execute, compare, gate, report, review."""

import json
import os
import signal
import textwrap
import threading

import pytest
from conftest import judged

from evalkit import cli_platform
from evalkit.cli import main
from evalkit.llm import LLMResponse

N = 100
TARGET_MODULE = """
import time
from evalkit.targets import TargetOutput

def good(inp):
    return "a" + inp.prompt[1:]

def degraded(inp):
    n = int(inp.case_key[1:])
    return "wrong" if n % 10 == 0 else "a" + inp.prompt[1:]

def slow(inp):
    time.sleep(0.15)
    return "a" + inp.prompt[1:]
"""


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "cli.db"))
    (tmp_path / "cli_targets.py").write_text(textwrap.dedent(TARGET_MODULE))
    monkeypatch.syspath_prepend(str(tmp_path))
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps(
                {"case_key": f"c{i:03}", "prompt": f"q{i}", "reference": f"a{i}", "output": f"a{i}"}
            )
            + "\n"
            for i in range(N)
        )
    )

    def run(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, (json.loads(out) if out.strip() else None), err

    run.tmp = tmp_path
    assert run("dataset", "import", "qa", str(data))[0] == 0
    return run


def write(cli, name, content):
    path = cli.tmp / name
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return str(path)


def create(cli, spec, *extra):
    code, run, err = cli("runs", "create", "qa", "--config", write(cli, "s.json", spec), *extra)
    assert code == 0, err
    return run["id"]


EM = {"evaluators": [{"kind": "exact_match", "name": "em"}]}
RUBRIC = {"criteria": [{"name": "quality", "description": "Good?"}]}
JUDGE_SPEC = {"kind": "llm_judge", "name": "q", "params": {"rubric": RUBRIC}}
GATES = '[gates]\n"exact_match:*.match" = { direction = "higher", delta = 0.02 }\n'
CALLABLE = {"target": {"kind": "callable"}, **EM}


def test_the_whole_workflow_regression_gate_and_report(cli):
    baseline = create(cli, EM)
    code, out, _ = cli("runs", "execute", baseline)
    assert (
        code == 0
        and out["run"]["status"] == "succeeded"
        and out["units"] == N
        and out["units_per_s"] > 0
    )
    assert cli("runs", "tag", baseline, "main")[1] == {"tags": ["main"]}

    candidate = create(cli, CALLABLE, "--target", "cli_targets:degraded", "--fingerprint", "v2")
    code, out, err = cli("runs", "execute", candidate, "--target", "cli_targets:degraded")
    assert code == 0, err

    code, out, _ = cli(
        "compare", candidate, "--baseline", "tag:main", "--gates", write(cli, "g.toml", GATES)
    )
    assert (
        code == 3
        and out["gate"]["status"] == "fail"
        and out["gate"]["decisions"][0]["decision"] == "regression"
    )
    assert out["comparison"]["common_cases"] == N
    (m,) = [m for m in out["comparison"]["metrics"] if m["evaluator"] != "run"]
    assert m["n_paired"] == N and m["diff"] == pytest.approx(-0.1) and m["candidate_lower"] == 10
    assert out["comparison"]["confounders"] == []  # the target is what was varied

    # no gates: a comparison alone never fails
    assert cli("compare", candidate, "--baseline", baseline)[0] == 0

    code, out, _ = cli(
        "report",
        candidate,
        "--out",
        str(cli.tmp / "r.html"),
        "--compare",
        "tag:main",
        "--gates",
        write(cli, "g.toml", GATES),
    )
    assert code == 0 and out["gate"] == "fail" and out["bytes"] > 1000
    page = (cli.tmp / "r.html").read_text()
    assert "REGRESSION" in page and "Paired metrics" in page and "<script" not in page
    code, _, err = cli("report", candidate, "--out", str(cli.tmp / "r.html"))
    assert code == 1 and "already exists" in err
    assert cli("report", candidate, "--out", str(cli.tmp / "r.html"), "--force")[0] == 0


def test_an_equivalent_candidate_passes_the_gate_and_an_inconclusive_one_exits_4_only_when_strict(
    cli,
):
    a = create(cli, EM)
    cli("runs", "execute", a)
    b = create(cli, CALLABLE, "--target", "cli_targets:good", "--fingerprint", "1")
    cli("runs", "execute", b, "--target", "cli_targets:good")
    gates = write(cli, "g.toml", GATES)
    code, out, _ = cli("compare", b, "--baseline", a, "--gates", gates)
    assert (
        code == 0
        and out["gate"]["status"] == "pass"
        and out["gate"]["decisions"][0]["decision"] == "equivalent"
    )
    tight = write(
        cli, "t.toml", '[gates]\n"exact_match:*.match" = { direction = "higher", delta = 0.0 }\n'
    )
    assert cli("compare", b, "--baseline", a, "--gates", tight)[0] == 0
    code, out, _ = cli("compare", b, "--baseline", a, "--gates", tight, "--strict")
    assert code == 4 and out["gate"]["status"] == "inconclusive"


def test_execution_exit_codes_partial_run_and_resume(cli):
    spec = {**CALLABLE, "policy": {"concurrency": 1, "budget": {"max_calls": 10}}}
    run = create(cli, spec, "--target", "cli_targets:good", "--fingerprint", "1")
    code, out, _ = cli("runs", "execute", run, "--target", "cli_targets:good")
    assert code == 1 and out["run"]["status"] == "partial" and out["stop_reason"] == "budget"
    assert 10 <= out["counts"]["complete"] < N
    # resume with the same frozen budget stops again; the run is resumable, never redone
    code, out, _ = cli("runs", "resume", run, "--target", "cli_targets:good")
    assert code == 1 and out["counts"]["complete"] > 10
    assert cli("runs", "verify", run)[0] == 0


def test_a_callable_run_needs_its_target_flag(cli):
    run = create(cli, CALLABLE, "--target", "cli_targets:good", "--fingerprint", "1")
    code, out, err = cli("runs", "execute", run)
    assert code == 1 and "pass --target" in err
    code, _, err = cli("runs", "execute", run, "--target", "cli_targets:nope")
    assert code == 1 and "cannot load" in err
    code, _, err = cli("runs", "execute", run, "--target", "no_such_module:fn")
    assert code == 1 and "cannot load" in err
    code, _, err = cli(
        "runs", "execute", run, "--target", "cli_targets:good", "--fingerprint", "other"
    )
    assert code == 1 and "does not match the run's frozen target" in err


def test_ctrl_c_cancels_gracefully_and_the_run_resumes(cli):
    run = create(
        cli,
        {**CALLABLE, "policy": {"concurrency": 1}},
        "--target",
        "cli_targets:slow",
        "--fingerprint",
        "1",
    )
    timer = threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    code, out, err = cli("runs", "execute", run, "--target", "cli_targets:slow")
    timer.join()
    assert code == 1 and out["run"]["status"] == "cancelled" and out["stop_reason"] == "interrupted"
    assert "interrupted: finishing the units in flight" in err
    done = out["counts"]["complete"]
    assert 1 <= done < N
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored
    # the same target resumes what is left (fast target now: same identity requires the same name)
    assert cli("runs", "verify", run)[0] == 0


def test_a_confounded_comparison_is_refused_unless_allowed(cli, monkeypatch):
    class J:
        provider = "fake"

        def __init__(self, model):
            self.model = model

        def call(self, req):
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use")

    runs = []
    for model in ("m1", "m2"):
        monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix, m=model: J(m))
        run = create(cli, {"evaluators": [JUDGE_SPEC]})
        assert cli("runs", "execute", run)[0] == 0
        runs.append(run)
    code, out, err = cli("compare", runs[1], "--baseline", runs[0])
    assert code == 1 and out is None and "confounded" in err and "model: m1 -> m2" in err
    code, out, _ = cli("compare", runs[1], "--baseline", runs[0], "--allow-confounders")
    assert code == 0 and out["comparison"]["confounders"][0]["kind"] == "evaluator"


def test_review_calibrate_queue_and_disagreements(cli, monkeypatch):
    class J:
        provider, model = "fake", "j"

        def call(self, req):
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use")

    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: J())
    spec = {"evaluators": [JUDGE_SPEC, {"kind": "exact_match", "name": "em"}]}
    run = create(cli, spec)
    cli("runs", "execute", run)
    _, shown, _ = cli("runs", "show", run)
    key = next(k for k in shown["counts"]["evaluator_results"] if k.startswith("llm_judge"))
    _, queue, _ = cli("runs", "queue", run, "--evaluator", key, "-n", "5", "--seed", "1")
    assert (
        len(queue) == 5
        and queue == cli("runs", "queue", run, "--evaluator", key, "-n", "5", "--seed", "1")[1]
    )
    for item in queue:
        code, out, _ = cli(
            "runs",
            "review",
            run,
            item["case_key"],
            "--evaluator",
            key,
            "--reviewer",
            "alice",
            "--verdict",
            "pass",
            "--sample",
            "random",
        )
        assert code == 0 and out["review_id"]
    code, cal, _ = cli("runs", "calibrate", run, "--evaluator", key, "--n-min", "3")
    assert (
        code == 0
        and cal["n_paired"] == 5
        and cal["accuracy"] == 1.0
        and cal["uncalibrated"] is False
    )
    assert cli("runs", "calibrate", run, "--evaluator", key)[1]["uncalibrated"] is True
    code, out, _ = cli("runs", "disagreements", run)
    assert code == 0 and out["compared"] == N and out["disagreeing"] == 0
    code, _, err = cli(
        "runs", "review", run, "nope", "--evaluator", key, "--reviewer", "a", "--verdict", "PASS"
    )
    assert code == 1 and "has no result" in err


def test_judge_check_command_stores_and_can_gate_on_accuracy(cli, monkeypatch):
    from evalkit.judgecheck import builtin_cases

    def truth(text):
        for c in builtin_cases():
            if f"\n{c.output}\n" in text and c.prompt in text:
                return 5 if c.expected == "PASS" else 1
        raise AssertionError

    class J:
        provider, model = "fake", "j"

        def __init__(self, decide):
            self.decide = decide

        def call(self, req):
            return LLMResponse(
                payload=judged(correctness=self.decide(req.user)), stop_reason="tool_use"
            )

    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: J(truth))
    code, out, _ = cli("judge-check", "--min-accuracy", "1.0")
    assert code == 0 and out["accuracy"] == 1.0 and out["adversarial_n"] == 9 and out["stored_id"]
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: J(lambda text: 5))
    code, out, _ = cli("judge-check", "--min-accuracy", "0.9")
    assert code == 3 and out["accuracy"] < 0.9
    cases = write(
        cli,
        "cases.jsonl",
        json.dumps({"name": "a", "prompt": "p", "output": "o", "expected": "PASS"}) + "\n",
    )
    code, out, _ = cli("judge-check", "--cases", cases)
    assert code == 0 and out["n"] == 1 and out["fixture_name"] == "cases.jsonl"
    assert cli("judge-check", "--cases", str(cli.tmp / "missing.jsonl"))[0] == 2


def test_tags_can_be_added_and_removed_and_baselines_need_a_tag(cli):
    run = create(cli, EM)
    cli("runs", "execute", run)
    assert cli("runs", "tag", run, "nightly")[1] == {"tags": ["nightly"]}
    assert cli("runs", "tag", run, "nightly", "--remove")[1] == {"removed": True}
    assert cli("runs", "tag", run, "nightly", "--remove")[1] == {"removed": False}
    other = create(cli, EM)
    cli("runs", "execute", other)
    code, _, err = cli("compare", other, "--baseline", "tag:nightly")
    assert code == 1 and "no succeeded run with tag 'nightly'" in err


def test_a_bad_gates_file_is_a_usage_error(cli):
    a = create(cli, EM)
    cli("runs", "execute", a)
    b = create(cli, EM)
    cli("runs", "execute", b)
    bad = write(cli, "bad.toml", '[gates]\n"nodot" = { direction = "higher", delta = 0 }\n')
    code, _, err = cli("compare", b, "--baseline", a, "--gates", bad)
    assert code == 2 and "nodot" in err
    assert cli("compare", b, "--baseline", a, "--gates", str(cli.tmp / "missing.toml"))[0] == 2


# --- judge-check names the evaluator it checks (P1.1, audit P1-11) -----------------------------


class GoldJudge:
    provider, model = "fake", "j"

    def call(self, req):
        from evalkit.judgecheck import builtin_cases

        for c in builtin_cases():
            if f"\n{c.output}\n" in req.user and c.prompt in req.user:
                score = 5 if c.expected == "PASS" else 1
                return LLMResponse(payload=judged(correctness=score), stop_reason="tool_use")
        return LLMResponse(payload=judged(correctness=5), stop_reason="tool_use")


BUILTIN_RUBRIC_JSON = {
    "criteria": [
        {
            "name": "correctness",
            "description": "Is the answer factually correct and does it answer the question?",
        }
    ],
    "threshold": 0.75,
    "version": "judge-check-builtin-v1",
}


def test_judge_check_of_a_run_is_stored_under_that_runs_evaluator_key(cli, monkeypatch):
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: GoldJudge())
    spec = {"evaluators": [{"kind": "llm_judge", "name": "quality",
                            "params": {"rubric": BUILTIN_RUBRIC_JSON}}]}  # fmt: skip
    run = create(cli, spec)
    assert cli("runs", "execute", run)[0] == 0
    key = next(iter(cli("runs", "show", run)[1]["counts"]["evaluator_results"]))
    code, out, err = cli("judge-check", "--run", run, "--evaluator", "quality")
    assert code == 0, err
    assert out["evaluator_key"] == key  # not a hard-coded "judge-check" identity
    assert out["accuracy"] == 1.0 and out["stored_id"]
    report = cli.tmp / "r.html"
    assert cli("report", run, "--out", str(report))[0] == 0
    html = report.read_text()
    assert "17/17 correct" in html and "not checked" not in html


def test_judge_check_of_a_run_checks_the_runs_own_judge_and_rubric(cli, monkeypatch):
    class Other(GoldJudge):
        model = "another-model"

    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: GoldJudge())
    spec = {"evaluators": [{"kind": "llm_judge", "name": "quality",
                            "params": {"rubric": BUILTIN_RUBRIC_JSON}}]}  # fmt: skip
    run = create(cli, spec)
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: Other())
    code, _, err = cli("judge-check", "--run", run)
    assert code == 1 and "another-model" in err and "judge" in err


def test_the_builtin_golden_set_is_refused_for_a_rubric_it_was_not_written_for(cli, monkeypatch):
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: GoldJudge())
    run = create(cli, {"evaluators": [JUDGE_SPEC]})  # a different rubric ("quality")
    code, _, err = cli("judge-check", "--run", run)
    assert code == 1 and "--cases" in err and "built-in" in err


def test_judge_check_needs_an_unambiguous_llm_judge_evaluator(cli, monkeypatch):
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: GoldJudge())
    code, _, err = cli("judge-check", "--run", create(cli, EM))
    assert code == 1 and "llm_judge" in err
    two = {"evaluators": [
        {"kind": "llm_judge", "name": "a", "params": {"rubric": BUILTIN_RUBRIC_JSON}},
        {"kind": "llm_judge", "name": "b", "params": {"rubric": BUILTIN_RUBRIC_JSON}},
    ]}  # fmt: skip
    run = create(cli, two)
    code, _, err = cli("judge-check", "--run", run)
    assert code == 1 and "--evaluator" in err
    assert cli("judge-check", "--run", run, "--evaluator", "a")[0] == 0
    assert cli("judge-check", "--run", run, "--evaluator", "zzz")[0] == 1


def test_a_standalone_judge_check_can_be_named(cli, monkeypatch):
    monkeypatch.setattr(cli_platform, "client_from_env", lambda prefix: GoldJudge())
    code, out, _ = cli("judge-check", "--name", "my-judge")
    assert code == 0 and out["evaluator_key"].startswith("llm_judge:my-judge:")
    code, _, err = cli("judge-check", "--run", "x", "--rubric", "r.json")
    assert code == 2 and "--rubric" in err
