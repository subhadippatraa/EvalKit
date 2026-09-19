import json

import pytest
from conftest import CRITERIA, FakeJudge, judged

from evalkit import Evaluator, JudgeTimeoutError
from evalkit.cli import main


@pytest.fixture
def cli(monkeypatch, store, capsys):
    """Run the CLI against a FakeJudge + temp store; returns (exit_code, stdout, stderr)."""
    judge = FakeJudge()
    monkeypatch.setattr(
        Evaluator,
        "from_env",
        classmethod(lambda cls, with_judge=True: cls(judge if with_judge else None, store)),
    )

    def invoke(*argv):
        code = main(list(argv))
        out, err = capsys.readouterr()
        return code, out, err

    invoke.judge = judge
    return invoke


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(
        json.dumps(
            {
                "prompt": "Explain DI.",
                "model_output": "DI is ...",
                "criteria": CRITERIA,
                "metadata": {"app": "ragforge"},
                "tags": ["regression-suite"],
            }
        )
    )
    return path


def test_run_get_list_review(cli, input_file):
    cli.judge.responses.append(judged(correctness=4, clarity=2))
    code, out, _ = cli("run", "--input", str(input_file))
    assert code == 0
    result = json.loads(out)
    assert result["verdict"] == "FAIL" and result["status"] == "ok"  # FAIL verdict still exits 0

    code, out, _ = cli("list", "--tag", "regression-suite", "--limit", "5")
    assert code == 0 and [r["id"] for r in json.loads(out)] == [result["id"]]
    assert json.loads(cli("list", "--tag", "other")[1]) == []

    code, out, _ = cli(
        "review", result["id"], "--reviewer", "alice", "--verdict", "pass", "--score", "0.8"
    )
    assert code == 0 and json.loads(out)["verdict"] == "PASS"
    cli("review", result["id"], "--reviewer", "bob", "--verdict", "FAIL", "--comment", "wrong")

    code, out, _ = cli("get", result["id"])
    got = json.loads(out)
    assert code == 0 and [r["reviewer"] for r in got["reviews"]] == ["alice", "bob"]
    assert got["reviews"][0]["score"] == 0.8 and got["reviews"][1]["comment"] == "wrong"


def test_run_judge_error_exits_1_with_evaluation_id(cli, input_file, store):
    cli.judge.responses.append(JudgeTimeoutError("timed out"))
    code, out, err = cli("run", "--input", str(input_file))
    assert code == 1 and out == ""
    (row,) = store.list()
    assert f"evaluation_id: {row.id}" in err and "timed out" in err


def test_run_invalid_rubric_exits_1(cli, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"prompt": "p", "model_output": "o"}))
    code, _, err = cli("run", "--input", str(path))
    assert code == 1 and "criteria" in err


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"prompt": "p", "unknown": 1}'])
def test_run_bad_input_file_exits_2(cli, tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert cli("run", "--input", str(path))[0] == 2


def test_run_missing_file_exits_2(cli, tmp_path):
    assert cli("run", "--input", str(tmp_path / "nope.json"))[0] == 2


def test_get_and_review_missing_exit_1(cli):
    assert cli("get", "missing")[0] == 1
    assert cli("review", "missing", "--reviewer", "a", "--verdict", "PASS")[0] == 1


def test_review_out_of_range_score_exits_1(cli, input_file):
    cli.judge.responses.append(judged(correctness=4, clarity=4))
    result = json.loads(cli("run", "--input", str(input_file))[1])
    code, _, err = cli(
        "review", result["id"], "--reviewer", "a", "--verdict", "PASS", "--score", "4"
    )
    assert code == 1 and "score" in err


def test_usage_errors_exit_2(cli):
    for argv in [(), ("review", "x", "--verdict", "MAYBE", "--reviewer", "a"), ("run",)]:
        with pytest.raises(SystemExit) as info:
            cli(*argv)
        assert info.value.code == 2


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_list_rejects_non_positive_limit(cli, limit):
    with pytest.raises(SystemExit) as info:
        cli("list", "--limit", limit)
    assert info.value.code == 2


def test_get_list_review_work_without_judge_env(monkeypatch, tmp_path, capsys):
    """get/list/review use Evaluator.from_env(with_judge=False), so they must not need
    EVALKIT_JUDGE_MODEL or any AWS configuration (this bypasses the `cli` fixture, which
    fakes Evaluator.from_env, to exercise the real one)."""
    monkeypatch.delenv("EVALKIT_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EVALKIT_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("EVALKIT_DB_PATH", str(tmp_path / "real.db"))

    assert main(["list"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert main(["get", "missing"]) == 1
    assert "not found" in capsys.readouterr().err

    assert main(["review", "missing", "--reviewer", "a", "--verdict", "PASS"]) == 1
    assert "not found" in capsys.readouterr().err
