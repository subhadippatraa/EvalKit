"""Spec-file targets, callable loading, and CLI edge cases."""

import json
import sys
import types

import pytest

from evalkit import ConfigError, EvalKit
from evalkit import cli_platform as cp
from evalkit.cli import main
from evalkit.llm import LLMResponse
from evalkit.targets import CallableTarget, ModelTarget, PrecomputedTarget


def test_build_target_covers_every_kind(monkeypatch):
    assert isinstance(cp.build_target({}), PrecomputedTarget)
    assert isinstance(cp.build_target({"kind": "precomputed"}), PrecomputedTarget)
    assert cp.build_target({"kind": "reuse", "source_run_id": "r1"}).source_run_id == "r1"
    mod = types.ModuleType("fake_targets_mod")
    mod.fn = lambda inp: "x"
    mod.notcallable = 5
    monkeypatch.setitem(sys.modules, "fake_targets_mod", mod)
    t = cp.build_target(
        {
            "kind": "callable",
            "callable": "fake_targets_mod:fn",
            "fingerprint": "f1",
            "on_empty": "fail",
        }
    )
    assert isinstance(t, CallableTarget) and t.fingerprint == "f1" and t.on_empty == "fail"
    flagged = cp.build_target({}, callable_ref="fake_targets_mod:fn", fingerprint="cli")
    assert (
        isinstance(flagged, CallableTarget) and flagged.fingerprint == "cli"
    )  # a flag implies callable

    class Client:
        provider, model = "fake", "m"

        def call(self, req):
            return LLMResponse(text="x")

    monkeypatch.setattr(cp, "client_from_env", lambda prefix: Client())
    m = cp.build_target(
        {"kind": "model", "template": "Q: {prompt}", "temperature": 0.3, "max_tokens": 9}
    )
    assert (
        isinstance(m, ModelTarget)
        and m.identity["temperature"] == 0.3
        and m.template == "Q: {prompt}"
    )


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"kind": "telepathy"}, "unknown target kind"),
        ({"kind": "precomputed", "extra": 1}, "unknown key"),
        ({"kind": "reuse"}, "needs source_run_id"),
        ({"kind": "callable"}, "needs `callable"),
    ],
)
def test_build_target_refuses_bad_specs(spec, message):
    with pytest.raises(ConfigError, match=message):
        cp.build_target(spec)


def test_load_callable_errors(monkeypatch):
    mod = types.ModuleType("fake_mod2")
    mod.value = 5
    monkeypatch.setitem(sys.modules, "fake_mod2", mod)
    with pytest.raises(ConfigError, match="not callable"):
        cp.load_callable("fake_mod2:value")
    for ref in ("nomodule", ":fn", "mod:", "fake_mod2:missing", "no_such_pkg_xyz:fn"):
        with pytest.raises(Exception, match="package.module:function|cannot load"):
            cp.load_callable(ref)


def test_a_huge_spec_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "MAX_INPUT_FILE_BYTES", 10)
    big = tmp_path / "s.json"
    big.write_text(json.dumps({"evaluators": []}))
    with pytest.raises(cp.UsageError, match="exceeds"):
        cp._read_spec(str(big))


def test_judge_specs_fill_provider_and_model_from_the_environment(monkeypatch):
    class Client:
        provider, model = "bedrock", "env-model"

    monkeypatch.setattr(cp, "client_from_env", lambda prefix: Client())
    rubric = {"criteria": [{"name": "q", "description": "Good?"}]}
    (spec,) = cp._judge_specs([{"kind": "llm_judge", "name": "q", "params": {"rubric": rubric}}])
    assert spec.params["provider"] == "bedrock" and spec.params["model"] == "env-model"
    (explicit,) = cp._judge_specs(
        [
            {
                "kind": "llm_judge",
                "name": "q",
                "params": {"rubric": rubric, "provider": "p", "model": "m"},
            }
        ]
    )
    assert explicit.params["model"] == "m"
    with pytest.raises(cp.UsageError):
        cp._judge_specs(["not a dict"])
    with pytest.raises(cp.UsageError):
        cp._judge_specs([{"kind": "x", "name": "n", "surprise": 1}])


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "c.db"))
    data = tmp_path / "d.jsonl"
    data.write_text(
        "".join(
            json.dumps({"case_key": f"c{i}", "prompt": f"q{i}", "output": "o", "reference": "o"})
            + "\n"
            for i in range(4)
        )
    )

    def run(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, (json.loads(out) if out.strip() else None), err

    run.tmp = tmp_path
    run("dataset", "import", "qa", str(data))
    return run


def test_a_reuse_spec_creates_and_executes_a_rescoring_run(cli):
    (cli.tmp / "a.json").write_text(
        json.dumps({"evaluators": [{"kind": "exact_match", "name": "em"}]})
    )
    _, first, _ = cli("runs", "create", "qa", "--config", str(cli.tmp / "a.json"))
    assert cli("runs", "execute", first["id"])[0] == 0
    (cli.tmp / "b.json").write_text(
        json.dumps(
            {
                "target": {"kind": "reuse", "source_run_id": first["id"]},
                "evaluators": [{"kind": "regex", "name": "r", "params": {"pattern": "o"}}],
            }
        )
    )
    code, second, err = cli("runs", "create", "qa", "--config", str(cli.tmp / "b.json"))
    assert code == 0, err
    code, out, _ = cli("runs", "execute", second["id"])
    assert code == 0 and out["counts"]["complete"] == 4 and out["spend"]["calls"] == 0


def test_a_model_target_run_needs_only_the_environment_to_execute(cli, monkeypatch):
    class Client:
        provider, model = "fake", "m"

        def call(self, req):
            return LLMResponse(text="o", stop_reason="end_turn")

    monkeypatch.setattr(cp, "client_from_env", lambda prefix: Client())
    (cli.tmp / "m.json").write_text(
        json.dumps(
            {
                "target": {"kind": "model", "template": "{prompt}"},
                "evaluators": [{"kind": "exact_match", "name": "em"}],
            }
        )
    )
    _, run, _ = cli("runs", "create", "qa", "--config", str(cli.tmp / "m.json"))
    code, out, err = cli("runs", "execute", run["id"])
    assert code == 0, err
    assert out["spend"]["calls"] == 4 and out["counts"]["complete"] == 4


def test_a_second_ctrl_c_stops_waiting(cli, monkeypatch):
    import signal

    (cli.tmp / "a.json").write_text(
        json.dumps({"evaluators": [{"kind": "exact_match", "name": "em"}]})
    )
    _, run, _ = cli("runs", "create", "qa", "--config", str(cli.tmp / "a.json"))
    seen = {}

    def fake_execute(self, run_id, **kw):
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)  # first: cancel
        seen["cancelled"] = kw["token"].cancelled
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)  # second: give up
        raise KeyboardInterrupt

    monkeypatch.setattr(type(EvalKit.open(":memory:").controller), "execute", fake_execute)
    with pytest.raises(KeyboardInterrupt):
        main(["runs", "execute", run["id"]])
    assert seen["cancelled"] and signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_judge_check_with_a_rubric_file(cli, monkeypatch):
    class J:
        provider, model = "fake", "j"

        def call(self, req):
            return LLMResponse(
                payload={"quality": {"reasoning": "r", "score": 5}}, stop_reason="tool_use"
            )

    monkeypatch.setattr(cp, "client_from_env", lambda prefix: J())
    rubric = cli.tmp / "rubric.json"
    rubric.write_text(json.dumps({"criteria": [{"name": "quality", "description": "Good?"}]}))
    cases = cli.tmp / "cases.jsonl"
    cases.write_text(
        json.dumps({"name": "a", "prompt": "p", "output": "o", "expected": "PASS"}) + "\n"
    )
    code, out, _ = cli("judge-check", "--cases", str(cases), "--rubric", str(rubric))
    assert code == 0 and out["accuracy"] == 1.0
    (cli.tmp / "bad.json").write_text("nope")
    assert cli("judge-check", "--cases", str(cases), "--rubric", str(cli.tmp / "bad.json"))[0] == 1


def test_compare_and_report_reject_bad_gates_files(cli):
    (cli.tmp / "a.json").write_text(
        json.dumps({"evaluators": [{"kind": "exact_match", "name": "em"}]})
    )
    ids = []
    for _ in range(2):
        _, run, _ = cli("runs", "create", "qa", "--config", str(cli.tmp / "a.json"))
        cli("runs", "execute", run["id"])
        ids.append(run["id"])
    bad = cli.tmp / "g.toml"
    bad.write_text('[gates]\n"nodot" = { direction = "higher", delta = 0 }\n')
    code, _, err = cli(
        "report", ids[1], "--out", str(cli.tmp / "r.html"), "--compare", ids[0], "--gates", str(bad)
    )
    assert code == 2 and "nodot" in err
