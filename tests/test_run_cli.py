"""`evalkit runs ...`: create / inspect / verify through spec files."""

import json

import pytest

from evalkit import CaseOutcome, EvalKit, Failure
from evalkit.cli import main

SPEC = {
    "evaluators": [{"kind": "exact_match", "name": "em"}],
    "policy": {"concurrency": 2},
}


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    db = tmp_path / "cli.db"
    monkeypatch.setenv("EVALKIT_DB_PATH", str(db))
    data = tmp_path / "qa.jsonl"
    data.write_text(
        "".join(
            json.dumps({"case_key": k, "prompt": "p", "output": "o", "reference": "o"}) + "\n"
            for k in "abc"
        )
    )
    cfg = tmp_path / "spec.json"
    cfg.write_text(json.dumps(SPEC))

    def run(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, (json.loads(out) if out.strip() else None), err

    run.db, run.cfg, run.tmp = db, cfg, tmp_path
    assert run("dataset", "import", "qa", str(data))[0] == 0
    return run


def test_create_list_show(cli):
    code, run, _ = cli("runs", "create", "qa", "--config", str(cli.cfg), "--name", "base")
    assert code == 0 and run["status"] == "created" and run["dataset_ref"] == "qa@1"
    assert run["name"] == "base" and run["config"]["policy"] == {"concurrency": 2}
    assert run["environment"]["preflight"]["ok"] is True
    code, listing, _ = cli("runs", "list", "--dataset", "qa@1", "--status", "created")
    assert code == 0 and [r["id"] for r in listing] == [run["id"]]
    code, shown, _ = cli("runs", "show", run["id"])
    assert code == 0 and shown["run"]["id"] == run["id"] and shown["tags"] == []
    assert shown["counts"]["total_cases"] == 3 and shown["counts"]["pending"] == 3  # planned
    assert shown["failure_counts"] == []
    code, with_summary, _ = cli("runs", "show", run["id"], "--summary")
    assert with_summary["summary"]["cases"] == 3


def test_create_is_idempotent_with_a_key_and_refuses_a_conflicting_config(cli, tmp_path):
    args = ("runs", "create", "qa", "--config", str(cli.cfg), "--idempotency-key", "k1")
    _, a, _ = cli(*args)
    _, b, _ = cli(*args)
    assert a["id"] == b["id"]
    other = tmp_path / "other.json"
    other.write_text(
        json.dumps({"evaluators": [{"kind": "regex", "name": "r", "params": {"pattern": "o"}}]})
    )
    code, out, err = cli("runs", "create", "qa", "--config", str(other), "--idempotency-key", "k1")
    assert code == 1 and out is None and "different dataset version or configuration" in err


def test_a_preflight_failure_refuses_the_run_with_the_reasons(cli, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"evaluators": [{"kind": "regex", "name": "r", "params": {"pattern": "("}}]})
    )
    code, out, err = cli("runs", "create", "qa", "--config", str(bad))
    assert code == 1 and out is None and "preflight failed" in err and "bad_config" in err
    assert cli("runs", "list")[1] == []


def test_failures_and_verify_after_results_were_recorded(cli):
    _, run, _ = cli("runs", "create", "qa", "--config", str(cli.cfg))
    kit = EvalKit.open(cli.db)
    kit.runs.start(run["id"])
    kit.runs.record_case_result(
        run["id"],
        "a",
        CaseOutcome.fail(Failure(failure_class="target", kind="timeout", message="t")),
    )
    kit.runs.record_case_result(run["id"], "b", CaseOutcome.complete("o"))
    kit.close()
    code, out, _ = cli("runs", "failures", run["id"])
    assert code == 0 and [(f["scope"], f["case_key"], f["failure"]["kind"]) for f in out] == [
        ("case", "a", "timeout")
    ]
    assert cli("runs", "failures", run["id"], "--class", "evaluator")[1] == []
    _, shown, _ = cli("runs", "show", run["id"])
    assert shown["failure_counts"] == [
        {"scope": "case", "failure_class": "target", "kind": "timeout", "count": 1}
    ]
    assert cli("runs", "verify", run["id"])[1] == {"ok": True, "problems": []}


def test_verify_exits_1_on_a_mismatch(cli):
    _, run, _ = cli("runs", "create", "qa", "--config", str(cli.cfg))
    kit = EvalKit.open(cli.db)
    conn = kit.store._conn
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("UPDATE runs SET identity_hash = ?", ("0" * 64,))
    conn.commit()
    kit.close()
    code, report, _ = cli("runs", "verify", run["id"])
    assert code == 1 and report["ok"] is False and "identity_hash" in report["problems"][0]


@pytest.mark.parametrize(
    "argv",
    [
        ("runs", "show", "nope"),
        ("runs", "failures", "nope"),
        ("runs", "verify", "nope"),
        ("runs", "execute", "nope"),
    ],
)
def test_unknown_runs_exit_1_with_a_message(cli, argv):
    code, out, err = cli(*argv)
    assert code == 1 and out is None and "not found" in err


def test_creating_against_an_unknown_dataset_exits_1(cli):
    code, _, err = cli("runs", "create", "nodata", "--config", str(cli.cfg))
    assert code == 1 and "not found" in err


def test_usage_errors_exit_2(cli, tmp_path):
    assert cli("runs", "create", "qa", "--config", str(tmp_path / "missing.json"))[0] == 2
    for content in (
        "[1, 2]",
        "not json",
        json.dumps({"bogus": 1}),
        json.dumps({"evaluators": [{"kind": "x"}]}),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(content)
        code, _, err = cli("runs", "create", "qa", "--config", str(bad))
        assert code == 2 and err.startswith("error:"), content
    with pytest.raises(SystemExit) as e:
        main(["runs", "failures", "x", "--class", "model"])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["runs", "list", "--status", "paused"])
    assert cli("runs", "create", "qa", "--config", str(cli.cfg), "--target", "nocolon")[0] == 2


def test_toml_specs_work(cli, tmp_path):
    spec = tmp_path / "spec.toml"
    spec.write_text(
        '[target]\nkind = "precomputed"\n[[evaluators]]\nkind = "exact_match"\nname = "em"\n'
        "[policy]\nconcurrency = 3\n"
    )
    code, run, _ = cli("runs", "create", "qa", "--config", str(spec))
    assert code == 0 and run["config"]["policy"] == {"concurrency": 3}


def test_existing_commands_are_unaffected(cli):
    code, out, _ = cli("dataset", "list")
    assert code == 0 and out[0]["name"] == "qa"
