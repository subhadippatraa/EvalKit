"""The quickstart example must keep working exactly as documented."""

import json
import shutil
from pathlib import Path

import pytest

from evalkit.cli import main

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "quickstart"


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    for f in EXAMPLE.iterdir():
        if f.suffix in (".jsonl", ".toml", ".py"):
            shutil.copy(f, tmp_path / f.name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "quickstart.db"))

    def run(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, json.loads(out) if out.strip() else None, err

    run.tmp = tmp_path
    return run


def test_the_documented_workflow_end_to_end(cli):
    assert (
        cli("dataset", "import", "arithmetic", "cases.jsonl")[1]["version"]["ref"] == "arithmetic@1"
    )
    code, base, _ = cli(
        "runs", "create", "arithmetic", "--config", "baseline.toml", "--name", "baseline"
    )
    assert code == 0 and base["environment"]["preflight"]["ok"]
    code, out, _ = cli("runs", "execute", base["id"])
    assert code == 0 and out["counts"]["complete"] == 60
    assert cli("runs", "tag", base["id"], "main")[1] == {"tags": ["main"]}

    code, cand, _ = cli(
        "runs", "create", "arithmetic", "--config", "candidate.toml", "--name", "candidate"
    )
    assert code == 0
    code, out, _ = cli("runs", "execute", cand["id"], "--target", "targets:candidate")
    assert code == 0 and out["run"]["status"] == "succeeded"

    code, shown, _ = cli("runs", "show", cand["id"], "--summary")
    (answer,) = [
        v for k, v in shown["summary"]["evaluators"].items() if k.startswith("exact_match")
    ]
    assert answer["metrics"]["match"]["n"] == 60 and answer["coverage"] == 1.0
    assert answer["metrics"]["match"]["observed_mean"] == pytest.approx(44 / 60)
    assert cli("runs", "failures", cand["id"])[1] == []

    code, cmp, _ = cli("compare", cand["id"], "--baseline", "tag:main", "--gates", "gates.toml")
    assert code == 3 and cmp["gate"]["status"] == "fail"  # the regression is caught
    decisions = {d["ref"]: d["decision"] for d in cmp["gate"]["decisions"]}
    assert decisions == {
        "exact_match:answer.match": "regression",
        "run.target_failure_rate": "equivalent",
    }
    assert {f["code"] for f in cmp["gate"]["failures"]} == {"regression", "floor"}
    (m,) = [m for m in cmp["comparison"]["metrics"] if m["evaluator"].startswith("exact_match")]
    assert (m["n_paired"], m["candidate_lower"], m["candidate_higher"]) == (60, 16, 3)
    assert m["mean_baseline"] == pytest.approx(0.95)
    assert m["mean_candidate"] == pytest.approx(44 / 60)
    assert cmp["comparison"]["common_cases"] == 60 and cmp["comparison"]["confounders"] == []

    code, out, _ = cli(
        "report",
        cand["id"],
        "--compare",
        "tag:main",
        "--gates",
        "gates.toml",
        "--out",
        "report.html",
    )
    assert code == 0 and (cli.tmp / "report.html").read_text().startswith("<!DOCTYPE html>")
    assert cli("runs", "verify", cand["id"])[0] == 0


def test_the_shell_script_names_only_existing_commands():
    text = (EXAMPLE / "run.sh").read_text()
    assert "evalkit compare" in text and "evalkit report" in text and "tag:main" in text
