"""`evalkit dataset ...`: thin JSON-in/JSON-out wrapper over the dataset service."""

import json
import sqlite3

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import Evaluator
from evalkit.cli import main

LINES = [
    {"case_key": "q1", "prompt": "What is 2+2?", "output": "4", "reference": "4", "tags": ["math"]},
    {
        "case_key": "q2",
        "prompt": "Capital of <France> & co?",
        "output": "Paris",
        "metadata": {"s": "faq"},
    },
]


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "cli.db"))
    data = tmp_path / "qa.jsonl"
    data.write_text("".join(json.dumps(x) + "\n" for x in LINES))

    def run(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, (json.loads(out) if out.strip() else None), err

    run.data = data
    run.tmp = tmp_path
    return run


def test_import_list_show_export_lint_verify(cli):
    code, out, _ = cli("dataset", "import", "qa", str(cli.data), "--description", "smoke")
    assert code == 0 and out["created"] is True
    v = out["version"]
    assert (v["ref"], v["case_count"], v["dataset_name"], v["source"]) == (
        "qa@1",
        2,
        "qa",
        "file:qa.jsonl",
    )

    code, again, _ = cli("dataset", "import", "qa", str(cli.data))
    assert code == 0 and again["created"] is False and again["version"]["id"] == v["id"]

    code, listing, _ = cli("dataset", "list")
    assert code == 0 and [
        (d["name"], d["version_count"], d["latest_version"]) for d in listing
    ] == [("qa", 1, 1)]

    code, shown, _ = cli("dataset", "show", "qa@latest", "--cases", "1")
    assert code == 0 and shown["content_hash"] == v["content_hash"]
    assert shown["cases"] == [LINES[0]]  # first by case_key, in import form

    target = cli.tmp / "out.jsonl"
    code, exported, _ = cli("dataset", "export", "qa", str(target))
    assert code == 0 and exported["cases"] == 2 and exported["content_hash"] == v["content_hash"]
    assert [json.loads(line) for line in target.read_text().splitlines()] == LINES  # lossless

    code, _, err = cli("dataset", "export", "qa", str(target))
    assert code == 1 and "already exists" in err and "--force" in err
    assert cli("dataset", "export", "qa", str(target), "--force")[0] == 0

    assert cli("dataset", "lint", "qa")[:2] == (0, [])
    code, report, _ = cli("dataset", "verify", "qa@1")
    assert code == 0 and report == {"ok": True, "case_count": 2, "problems": []}


def test_show_resolves_selectors_and_reports_unknown_refs(cli):
    cli("dataset", "import", "qa", str(cli.data))
    assert cli("dataset", "show", "qa@1")[0] == 0
    for ref in ("nope", "qa@9", "qa@zzz", "a b"):
        code, out, err = cli("dataset", "show", ref)
        assert code == 1 and out is None and err.startswith("error:")


def test_a_new_version_appears_when_the_file_changes(cli):
    cli("dataset", "import", "qa", str(cli.data))
    cli.data.write_text(
        cli.data.read_text() + json.dumps({"case_key": "q3", "prompt": "p3"}) + "\n"
    )
    code, out, _ = cli("dataset", "import", "qa", str(cli.data))
    assert code == 0 and out["created"] and out["version"]["ref"] == "qa@2"


def test_invalid_datasets_exit_1_with_line_numbers_and_store_nothing(cli):
    bad = cli.tmp / "bad.jsonl"
    bad.write_text(
        '{"case_key": "a", "prompt": "p"}\n{oops}\n{"case_key": "a", "prompt": "dup"}\n[1]\n'
    )
    code, out, err = cli("dataset", "import", "qa", str(bad))
    assert code == 1 and out is None
    assert "line 2" in err and "line 4" in err and "duplicate case_key" in err
    assert "nothing was imported" in err
    assert cli("dataset", "list")[1] == []


def test_a_missing_file_is_a_usage_error(cli):
    code, out, err = cli("dataset", "import", "qa", str(cli.tmp / "nope.jsonl"))
    assert code == 2 and "cannot read dataset file" in err


def test_invalid_names_and_arguments(cli):
    assert cli("dataset", "import", "bad name", str(cli.data))[0] == 1
    for argv in [("dataset",), ("dataset", "import"), ("dataset", "bogus")]:
        with pytest.raises(SystemExit) as info:
            main(list(argv))
        assert info.value.code == 2


def test_lint_reports_findings_and_verify_fails_on_tampering(cli):
    dup = cli.tmp / "dup.jsonl"
    dup.write_text('{"case_key": "a", "prompt": "same"}\n{"case_key": "b", "prompt": "same"}\n')
    cli("dataset", "import", "d", str(dup))
    code, findings, _ = cli("dataset", "lint", "d")
    assert code == 0 and [f["code"] for f in findings] == ["duplicate_content"]
    assert findings[0]["examples"] == ["a", "b"]

    conn = sqlite3.connect(cli.tmp / "cli.db")
    conn.execute("DROP TRIGGER cases_no_update")
    conn.execute("UPDATE cases SET prompt = 'tampered' WHERE case_key = 'a'")
    conn.commit()
    conn.close()
    code, report, _ = cli("dataset", "verify", "d")
    assert code == 1 and report["ok"] is False and "a" in report["problems"][0]


def test_existing_commands_still_work_against_the_same_database(cli, tmp_path, monkeypatch):
    cli("dataset", "import", "qa", str(cli.data))
    monkeypatch.setattr(
        Evaluator,
        "from_env",
        classmethod(
            lambda cls, with_judge=True: cls(
                FakeJudge(judged(correctness=5, clarity=5)) if with_judge else None,
                __import__("evalkit").store.SQLiteStore(tmp_path / "cli.db"),
            )
        ),
    )
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"prompt": "p", "model_output": "o", "criteria": CRITERIA}))
    code, result, _ = cli("run", "--input", str(inp))
    assert code == 0 and result["verdict"] == "PASS"
    assert cli("list")[0] == 0 and cli("get", result["id"])[0] == 0
    assert cli("dataset", "verify", "qa")[0] == 0
