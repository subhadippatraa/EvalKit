"""Kill -9 a real process mid-run, then resume: no duplicates, no lost cases, no half results."""

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from evalkit import EvalKit, EvaluatorSpec
from evalkit.targets import CallableTarget

N = 150
WORKER = """
import sys, time
from evalkit import EvalKit, EvaluatorSpec
from evalkit.targets import CallableTarget

def slow(inp):
    time.sleep(0.02)
    return "a" + inp.prompt[1:]

kit = EvalKit.open(sys.argv[1])
target = CallableTarget(slow, name="slow", fingerprint="1")
kit.controller.execute(sys.argv[2], target=target)
"""
SPECS = [
    EvaluatorSpec(kind="exact_match", name="em"),
    EvaluatorSpec(kind="regex", name="digits", params={"pattern": r"\d"}),
]


def fast(inp):
    return "a" + inp.prompt[1:]


@pytest.mark.skipif(os.name != "posix", reason="needs SIGKILL")
@pytest.mark.parametrize("delay_rows", [20, 120])
def test_a_killed_run_resumes_exactly_once(tmp_path, delay_rows):
    db = tmp_path / "crash.db"
    kit = EvalKit.open(db)
    kit.datasets.import_cases(
        "qa",
        [{"case_key": f"c{i:03}", "prompt": f"q{i}", "reference": f"a{i}"} for i in range(N)],
    )
    target = CallableTarget(lambda i: "x", name="slow", fingerprint="1")
    run = kit.controller.create(
        "qa",
        target=target,
        evaluators=SPECS,
        policy={"concurrency": 2, "batch_size": 10, "batch_ms": 30},
    )
    kit.close()

    script = tmp_path / "worker.py"
    script.write_text(textwrap.dedent(WORKER))
    proc = subprocess.Popen([sys.executable, str(script), str(db), run.id])
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(0.05)
            try:
                conn = sqlite3.connect(db, timeout=5)
                (done,) = conn.execute(
                    "SELECT COUNT(*) FROM evaluator_results WHERE run_id = ?", (run.id,)
                ).fetchone()
                conn.close()
            except sqlite3.OperationalError:
                continue
            if done >= delay_rows:
                break
        else:
            pytest.fail("the worker never made progress")
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()
    assert proc.returncode == -signal.SIGKILL

    kit = EvalKit.open(db)
    interrupted = kit.runs.counts(run.id)
    assert interrupted.complete + interrupted.pending == N and interrupted.pending > 0
    assert (
        kit.runs.get(run.id).status == "running"
    )  # the crash left it mid-run: not falsely finished

    target = CallableTarget(fast, name="slow", fingerprint="1")  # same identity, quicker function
    report = kit.controller.execute(run.id, target=target)
    assert report.status == "succeeded"
    conn = kit.store._conn
    assert tuple(
        conn.execute("SELECT COUNT(*), COUNT(DISTINCT case_id) FROM case_results").fetchone()
    ) == (N, N)
    assert tuple(
        conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT case_result_id || evaluator_key) "
            "FROM evaluator_results"
        ).fetchone()
    ) == (2 * N, 2 * N)
    assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 2 * N
    assert (
        kit.runs.verify(run.id).ok and conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    )
    counts = kit.runs.counts(run.id)
    assert (counts.complete, counts.pending, counts.missing) == (N, 0, 0)
    assert all(sum(v.values()) == N for v in counts.evaluator_results.values())
    kit.close()
