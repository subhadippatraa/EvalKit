"""Human review, calibration, review queues, evaluator disagreement, judge-check."""

import json

import pytest
from conftest import case, judged, refuses

from evalkit import (
    CaseOutcome,
    EvalFailure,
    EvaluatorOutcome,
    EvaluatorSpec,
    RunConfig,
    RunError,
)
from evalkit.calibration import N_MIN, calibrate, disagreements, review_queue
from evalkit.errors import ConfigError
from evalkit.evaluators import llm_judge_spec, resolve
from evalkit.judgecheck import (
    BUILTIN_RUBRIC,
    JudgeCase,
    builtin_cases,
    fixture_hash,
    judge_check,
    load_cases,
)
from evalkit.llm import LLMResponse
from evalkit.models import Rubric

JUDGE = EvaluatorSpec(kind="llm_judge", name="quality", params={"provider": "p", "model": "m"})
DET = resolve(EvaluatorSpec(kind="exact_match", name="em"))


def verdict_run(kit, verdicts, *, specs=(JUDGE,), extra=None):
    """A run whose case i got verdicts[i] from JUDGE (and extra[i] from a second evaluator)."""
    n = len(verdicts)
    kit.datasets.import_cases("qa", [case(f"c{i:03}", output="o", reference="r") for i in range(n)])
    run = kit.runs.create("qa", RunConfig(evaluators=list(specs)))
    kit.runs.start(run.id)
    for i, v in enumerate(verdicts):
        cr = kit.runs.record_case_result(run.id, f"c{i:03}", CaseOutcome.complete("o"))
        kit.runs.record_evaluator_result(
            cr.id,
            EvaluatorOutcome(
                evaluator_key=JUDGE.key, status="ok", verdict=v,
                metrics={"score": 1.0 if v == "PASS" else 0.0 if v == "FAIL" else 0.5},
            ),
        )  # fmt: skip
        if extra:
            kit.runs.record_evaluator_result(
                cr.id,
                EvaluatorOutcome(
                    evaluator_key=DET.key,
                    status="ok",
                    verdict=extra[i],
                    metrics={"match": 1.0 if extra[i] == "PASS" else 0.0},
                ),
            )
    kit.runs.transition(run.id, "succeeded")
    return run


def review(kit, run, i, verdict, reviewer="alice", sample="random", key=None, **kw):
    return kit.reviews.add(
        run.id, f"c{i:03}", reviewer=reviewer, verdict=verdict, sample=sample,
        evaluator_key=key or JUDGE.key, **kw,
    )  # fmt: skip


# --- calibration ------------------------------------------------------------------------------


def test_calibration_statistics_from_a_confusion_matrix(kit):
    # judge:  FAIL FAIL FAIL PASS PASS PASS PASS PASS PASS PASS
    # human:  FAIL FAIL PASS PASS PASS PASS FAIL PASS PASS PASS
    judge = ["FAIL"] * 3 + ["PASS"] * 7
    human = ["FAIL", "FAIL", "PASS", "PASS", "PASS", "PASS", "FAIL", "PASS", "PASS", "PASS"]
    run = verdict_run(kit, judge)
    for i, h in enumerate(human):
        review(kit, run, i, h)
    c = calibrate(kit, run.id, JUDGE.key, n_min=5)
    assert c.confusion == {"tp": 2, "fp": 1, "fn": 1, "tn": 6} and c.n_paired == 10
    assert (
        c.accuracy == 0.8
        and c.fail_precision == pytest.approx(2 / 3)
        and c.fail_recall == pytest.approx(2 / 3)
    )
    expected_kappa = (0.8 - (0.3 * 0.3 + 0.7 * 0.7)) / (1 - (0.3 * 0.3 + 0.7 * 0.7))
    assert c.kappa == pytest.approx(expected_kappa)
    assert not c.uncalibrated and c.accuracy_ci[0] < 0.8 < c.accuracy_ci[1]
    assert [d["case_key"] for d in c.disagreements] == ["c002", "c006"]
    assert c.disagreements[0] == {"case_key": "c002", "evaluator": "FAIL", "human": "PASS"}


def test_below_n_min_an_evaluator_is_uncalibrated(kit):
    run = verdict_run(kit, ["PASS"] * 5)
    for i in range(5):
        review(kit, run, i, "PASS")
    c = calibrate(kit, run.id, JUDGE.key)
    assert c.uncalibrated and c.n_paired == 5 and c.n_min == N_MIN == 30
    assert calibrate(kit, run.id, JUDGE.key, n_min=5).uncalibrated is False


def test_no_reviews_means_uncalibrated_with_undefined_statistics(kit):
    run = verdict_run(kit, ["PASS"] * 3)
    c = calibrate(kit, run.id, JUDGE.key)
    assert (c.n_paired, c.uncalibrated, c.accuracy, c.kappa, c.accuracy_ci) == (
        0,
        True,
        None,
        None,
        None,
    )
    assert c.reviewer_agreement == {"pairs": 0, "percent": None, "kappa": None}


def test_only_random_sample_reviews_enter_calibration(kit):
    run = verdict_run(kit, ["PASS"] * 4)
    review(kit, run, 0, "PASS", sample="random")
    review(kit, run, 1, "FAIL", sample="targeted")  # triage: would bias the estimate
    review(kit, run, 2, "FAIL", sample="targeted")
    c = calibrate(kit, run.id, JUDGE.key, n_min=1)
    assert c.n_paired == 1 and c.accuracy == 1.0 and c.excluded == {"targeted_only": 2}


def test_consensus_is_the_majority_and_ties_are_excluded(kit):
    run = verdict_run(kit, ["PASS"] * 3)
    for reviewer, verdict in (("a", "PASS"), ("b", "PASS"), ("c", "FAIL")):
        review(kit, run, 0, verdict, reviewer=reviewer)  # 2 vs 1: consensus PASS
    review(kit, run, 1, "PASS", reviewer="a")
    review(kit, run, 1, "FAIL", reviewer="b")  # 1 vs 1: a tie
    c = calibrate(kit, run.id, JUDGE.key, n_min=1)
    assert c.n_paired == 1 and c.confusion["tn"] == 1 and c.excluded == {"tie": 1}


def test_uncertain_verdicts_and_cases_without_a_verdict_are_excluded_and_counted(kit):
    run = verdict_run(kit, ["UNCERTAIN", "PASS", "PASS"])
    for i in range(3):
        review(kit, run, i, "PASS")
    c = calibrate(kit, run.id, JUDGE.key, n_min=1)
    assert c.excluded == {"uncertain": 1} and c.n_paired == 2


def test_reviewer_agreement_is_reported_because_humans_are_not_gold(kit):
    run = verdict_run(kit, ["PASS"] * 6)
    for i, (a, b) in enumerate(
        [
            ("PASS", "PASS"),
            ("PASS", "PASS"),
            ("FAIL", "FAIL"),
            ("PASS", "FAIL"),
            ("FAIL", "PASS"),
            ("PASS", "PASS"),
        ]
    ):
        review(kit, run, i, a, reviewer="alice")
        review(kit, run, i, b, reviewer="bob")
    agreement = calibrate(kit, run.id, JUDGE.key).reviewer_agreement
    assert agreement["pairs"] == 6 and agreement["percent"] == pytest.approx(4 / 6)
    assert agreement["kappa"] is not None and agreement["kappa"] < 1


def test_calibration_is_per_evaluator_key(kit):
    run = verdict_run(kit, ["PASS", "PASS"], specs=(JUDGE, DET), extra=["PASS", "FAIL"])
    review(kit, run, 0, "PASS", key=JUDGE.key)
    review(kit, run, 1, "FAIL", key=DET.key)
    assert calibrate(kit, run.id, JUDGE.key, n_min=1).n_paired == 1
    det = calibrate(kit, run.id, DET.key, n_min=1)
    assert det.n_paired == 1 and det.accuracy == 1.0


# --- reviews: validation and storage ----------------------------------------------------------


def test_review_validation(kit):
    run = verdict_run(kit, ["PASS"] * 2)
    for bad in (
        dict(verdict="MAYBE"), dict(reviewer=""), dict(reviewer="x" * 200), dict(score=2.0),
        dict(score=float("nan")), dict(comment="x" * 20000), dict(sample="all"),
    ):  # fmt: skip
        with pytest.raises(RunError, match="invalid review"):
            kit.reviews.add(run.id, "c000", **{"reviewer": "a", "verdict": "PASS", **bad})
    with pytest.raises(RunError, match="has no result"):
        kit.reviews.add(run.id, "zzz", reviewer="a", verdict="PASS")
    with pytest.raises(RunError, match="not one of run"):
        kit.reviews.add(
            run.id, "c000", reviewer="a", verdict="PASS", evaluator_key="regex:x:000000000000"
        )
    with pytest.raises(RunError, match="not found"):
        kit.reviews.add("nope", "c000", reviewer="a", verdict="PASS")


def test_reviews_are_append_only_and_listed_in_order(kit, store):
    run = verdict_run(kit, ["PASS"] * 2)
    review(kit, run, 0, "PASS", comment="fine", score=0.9)
    review(kit, run, 0, "FAIL", reviewer="bob", sample="targeted")
    listed = kit.reviews.list(run.id, JUDGE.key)
    assert [(r["reviewer"], r["verdict"], r["sample"]) for r in listed] == [
        ("alice", "PASS", "random"),
        ("bob", "FAIL", "targeted"),
    ]
    assert listed[0]["comment"] == "fine" and listed[0]["score"] == 0.9
    refuses(store, "UPDATE reviews SET verdict = 'FAIL'", match="append-only")
    refuses(store, "DELETE FROM reviews", match="append-only")


def test_the_database_requires_exactly_one_review_subject(kit, store):
    run = verdict_run(kit, ["PASS"])
    review(kit, run, 0, "PASS")
    (cr_id,) = [r[0] for r in store._conn.execute("SELECT id FROM case_results")]
    base = (
        "INSERT INTO reviews (id, evaluation_id, case_result_id, reviewer, verdict, created_at) "
        "VALUES ('x', ?, ?, 'a', 'PASS', 't')"
    )
    refuses(store, base, (None, None))  # no subject
    refuses(store, base, ("e", cr_id))  # two subjects
    refuses(
        store,
        "INSERT INTO reviews (id, evaluation_id, reviewer, verdict, evaluator_key, created_at) "
        "VALUES ('y', NULL, 'a', 'PASS', 'k', 't')",
    )


# --- review queues ----------------------------------------------------------------------------


def test_random_queue_is_seeded_excludes_reviewed_cases_and_only_offers_scored_ones(kit):
    run = verdict_run(kit, ["PASS", "FAIL"] * 20)
    review(kit, run, 0, "PASS")
    q1 = review_queue(kit, run.id, JUDGE.key, strategy="random", n=10, seed=1)
    assert q1 == review_queue(kit, run.id, JUDGE.key, strategy="random", n=10, seed=1)
    assert q1 != review_queue(kit, run.id, JUDGE.key, strategy="random", n=10, seed=2)
    assert len(q1) == 10 and "c000" not in {q["case_key"] for q in q1}
    assert len(kit.reviews.queue(run.id, JUDGE.key, n=1000)) == 39


def test_stratified_queue_draws_from_both_verdicts(kit):
    run = verdict_run(kit, ["PASS"] * 35 + ["FAIL"] * 5)
    q = review_queue(kit, run.id, JUDGE.key, strategy="stratified", n=10, seed=3)
    reasons = [x["reason"] for x in q]
    assert sum("FAIL" in r for r in reasons) == 5 and sum("PASS" in r for r in reasons) == 5


def test_uncertain_and_disagreement_queues_are_triage(kit):
    run = verdict_run(
        kit,
        ["UNCERTAIN", "PASS", "PASS", "UNCERTAIN"],
        specs=(JUDGE, DET),
        extra=["PASS", "FAIL", "PASS", "PASS"],
    )
    unc = review_queue(kit, run.id, JUDGE.key, strategy="uncertain", n=10)
    assert [q["case_key"] for q in unc] == ["c000", "c003"]
    dis = review_queue(kit, run.id, JUDGE.key, strategy="disagreement", n=10)
    assert [q["case_key"] for q in dis] == ["c001"] and "says" in dis[0]["reason"]


@pytest.mark.parametrize("kw", [{"strategy": "best"}, {"n": 0}, {"n": True}])
def test_invalid_queue_arguments(kit, kw):
    run = verdict_run(kit, ["PASS"])
    with pytest.raises(RunError):
        review_queue(kit, run.id, JUDGE.key, **kw)


# --- evaluator disagreement -------------------------------------------------------------------


def test_a_judge_that_passes_what_a_deterministic_check_fails_is_flagged(kit):
    run = verdict_run(
        kit, ["PASS"] * 5, specs=(JUDGE, DET), extra=["PASS", "PASS", "FAIL", "FAIL", "PASS"]
    )
    d = disagreements(kit, run.id)
    assert (d["compared"], d["disagreeing"]) == (5, 2) and d["rate"] == 0.4
    assert {c["case_key"] for c in d["cases"]} == {"c002", "c003"}
    assert {c["kind"] for c in d["cases"]} == {"judge_vs_deterministic"}
    assert "llm_judge:quality" in d["cases"][0]["detail"] and "PASS" in d["cases"][0]["detail"]


def test_two_judges_are_compared_by_verdict_and_by_score(kit):
    j2 = EvaluatorSpec(kind="llm_judge", name="second", params={"provider": "p", "model": "m2"})
    kit.datasets.import_cases("qa", [case(f"c{i:03}", output="o") for i in range(3)])
    run = kit.runs.create("qa", RunConfig(evaluators=[JUDGE, j2]))
    kit.runs.start(run.id)
    rows = [("PASS", 0.9, "PASS", 0.9), ("PASS", 0.9, "FAIL", 0.2), ("PASS", 0.9, "PASS", 0.5)]
    for i, (v1, s1, v2, s2) in enumerate(rows):
        cr = kit.runs.record_case_result(run.id, f"c{i:03}", CaseOutcome.complete("o"))
        for spec, v, s in ((JUDGE, v1, s1), (j2, v2, s2)):
            kit.runs.record_evaluator_result(
                cr.id,
                EvaluatorOutcome(
                    evaluator_key=spec.key, status="ok", verdict=v, metrics={"score": s}
                ),
            )
    d = disagreements(kit, run.id, tau=0.3)
    assert [(c["case_key"], c["kind"]) for c in d["cases"]] == [
        ("c001", "verdict_disagreement"),
        ("c002", "score_disagreement"),
    ]
    assert disagreements(kit, run.id, tau=0.5)["disagreeing"] == 1
    with pytest.raises(RunError):
        disagreements(kit, run.id, tau=0)


def test_a_single_evaluator_has_nothing_to_disagree_with(kit):
    d = disagreements(kit, verdict_run(kit, ["PASS"] * 3).id)
    assert d == {"compared": 0, "disagreeing": 0, "rate": None, "tau": 0.25, "cases": []}


# --- judge-check ------------------------------------------------------------------------------


class Scripted:
    """A judge stand-in that decides by looking at the request, like a (good or bad) model would."""

    provider, model = "fake", "judge-x"

    def __init__(self, decide):
        self.decide, self.requests = decide, []

    def call(self, req):
        self.requests.append(req)
        out = self.decide(req.user)
        if isinstance(out, BaseException):
            raise out
        return LLMResponse(payload=judged(correctness=out), stop_reason="tool_use")


def truth(text):
    """The right verdict for a builtin case, found from what the judge is shown."""
    for c in builtin_cases():
        if f"\n{c.output}\n" in text and c.prompt in text:
            return 5 if c.expected == "PASS" else 1
    raise AssertionError("unknown case in the prompt")


def spec_for(client):
    return llm_judge_spec("q", BUILTIN_RUBRIC, client)


def test_the_builtin_set_covers_the_documented_attacks():
    cases = builtin_cases()
    adv = [c for c in cases if c.category == "adversarial"]
    assert len(cases) >= 15 and len(adv) >= 9 and len({c.name for c in cases}) == len(cases)
    text = " ".join((c.attack or "") for c in adv)
    for needle in (
        "inflate",
        "forged system",
        "padding",
        "delimiter",
        "context",
        "LOWER",
        "verification",
    ):
        assert needle in text
    assert {c.expected for c in cases} == {"PASS", "FAIL"} and all(
        c.prompt and c.output for c in cases
    )


def test_the_builtin_set_renders_byte_exact_through_the_judge_prompt():
    """The prompt-rendering golden set: forged delimiters and odd text survive verbatim, and no
    line of the content can equal a real delimiter (the marker avoids every content string)."""
    from evalkit.judge import DELIMITER_CLOSE, DELIMITER_OPEN, render_prompt

    for c in builtin_cases():
        prompt = render_prompt(c.prompt, c.output, c.reference, c.context, BUILTIN_RUBRIC)
        assert c.output in prompt and c.prompt in prompt
        marker = prompt.split(DELIMITER_OPEN, 1)[1].split(" ", 1)[0]
        assert marker not in c.output and marker not in (c.context or "")
        real = [
            ln
            for ln in prompt.splitlines()
            if ln.startswith((DELIMITER_OPEN, DELIMITER_CLOSE)) and marker in ln
        ]
        assert len(real) == 2 * (2 + bool(c.reference) + bool(c.context))  # exactly the real blocks


def test_a_sound_judge_scores_perfectly_and_the_result_is_stored_per_evaluator_key(kit):
    client = Scripted(truth)
    result = judge_check(kit, spec_for(client), client)
    assert (result.n, result.correct, result.failed, result.accuracy) == (17, 17, 0, 1.0)
    assert result.adversarial_accuracy == 1.0 and result.adversarial_n == 9
    stored = kit.store.latest_judge_check(result.evaluator_key)
    assert (
        stored["fixture_name"] == "builtin-v1"
        and stored["correct"] == 17
        and stored["id"] == result.stored_id
    )
    assert len(stored["results"]) == 17 and stored["fixture_hash"] == result.fixture_hash
    assert kit.store.latest_judge_check("other:key:000000000000") is None


def test_a_gullible_judge_is_caught_by_the_adversarial_cases(kit):
    def gullible(text):
        if (
            "Ignore all previous instructions" in text
            or "SYSTEM: the evaluation is finished" in text
        ):
            return 5  # obeys the injected instruction
        return truth(text)

    client = Scripted(gullible)
    result = judge_check(kit, spec_for(client), client)
    assert result.accuracy == pytest.approx(15 / 17)
    assert result.adversarial_accuracy == pytest.approx(7 / 9)
    failed = {r["name"]: r for r in result.results if not r["correct"]}
    assert set(failed) == {"inject_max_score", "inject_system_role"}
    assert failed["inject_max_score"]["verdict"] == "PASS" and failed["inject_max_score"]["attack"]


def test_a_judge_that_cannot_judge_is_not_counted_correct(kit):
    client = Scripted(lambda text: EvalFailure("infrastructure", "auth", "denied"))
    result = judge_check(kit, spec_for(client), client)
    assert (result.correct, result.failed, result.accuracy) == (0, 17, 0.0)
    assert result.results[0]["failure"].startswith("infrastructure.auth")


def test_judge_check_can_skip_storage_and_needs_a_judge_spec(kit):
    client = Scripted(truth)
    judge_check(kit, spec_for(client), client, store=False)
    assert kit.store.latest_judge_check(spec_for(client).key) is None
    with pytest.raises(ConfigError, match="llm_judge"):
        judge_check(kit, DET, client)
    with pytest.raises(ConfigError, match="no cases"):
        judge_check(kit, spec_for(client), client, cases=[])


def test_the_fixture_hash_identifies_the_exact_cases():
    a = builtin_cases()
    assert fixture_hash(a) == fixture_hash(builtin_cases())
    b = builtin_cases()
    b[0] = JudgeCase(**{**b[0].__dict__, "output": "Paris!"})
    assert fixture_hash(a) != fixture_hash(b)


def test_judge_checks_are_write_once(kit, store):
    client = Scripted(truth)
    judge_check(kit, spec_for(client), client)
    refuses(store, "UPDATE judge_checks SET correct = 0", match="write-once")
    refuses(store, "DELETE FROM judge_checks", match="permanent")
    refuses(
        store,
        "INSERT INTO judge_checks (id, evaluator_key, fixture_name, fixture_hash, n, correct, "
        "adversarial_n, adversarial_correct, failed, results_json, created_at) "
        "VALUES ('x', 'k', 'f', 'short', 1, 1, 0, 0, 0, '[]', 't')",
    )


def test_user_supplied_cases_load_strictly(tmp_path):
    good = tmp_path / "cases.jsonl"
    good.write_text(
        json.dumps({"name": "a", "prompt": "p", "output": "o", "expected": "PASS"})
        + "\n"
        + json.dumps(
            {
                "name": "b",
                "prompt": "p",
                "output": "o",
                "expected": "FAIL",
                "category": "adversarial",
                "attack": "x",
                "reference": "r",
            }
        )
        + "\n"
    )
    cases = load_cases(good)
    assert [(c.name, c.category, c.reference) for c in cases] == [
        ("a", "golden", None),
        ("b", "adversarial", "r"),
    ]
    for bad, message in (
        ('{"name": "a", "prompt": "p", "output": "o", "expected": "MAYBE"}', "PASS or FAIL"),
        ('{"name": "a", "prompt": "p", "output": "o"}', "needs"),
        ('{"name": "a", "prompt": "p", "output": "o", "expected": "PASS", "extra": 1}', "needs"),
        ('{"name": "a", "prompt": 5, "output": "o", "expected": "PASS"}', "strings"),
        (
            '{"name": "a", "prompt": "p", "output": "o", "expected": "PASS", "category": "x"}',
            "category",
        ),
        ("not json", "line"),
    ):
        path = tmp_path / "bad.jsonl"
        path.write_text(bad + "\n")
        with pytest.raises(ConfigError, match=message):
            load_cases(path)
    path.write_text("")
    with pytest.raises(ConfigError, match="no cases"):
        load_cases(path)
    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        json.dumps({"name": "a", "prompt": "p", "output": "o", "expected": "PASS"})
        + "\n" * 1
        + json.dumps({"name": "a", "prompt": "p", "output": "o", "expected": "PASS"})
        + "\n"
    )
    with pytest.raises(ConfigError, match="unique"):
        load_cases(dup)


def test_a_custom_rubric_and_cases_work_end_to_end(kit):
    rubric = Rubric.from_dict({"quality": "Good?"})
    client = Scripted(lambda text: 5 if "GOOD" in text else 1)
    client.decide = lambda text: 5 if "GOOD" in text else 1

    def call(req):
        v = 5 if "GOOD" in req.user else 1
        return LLMResponse(payload=judged(quality=v), stop_reason="tool_use")

    client.call = call
    cases = [JudgeCase("g", "p", "GOOD answer", "PASS"), JudgeCase("b", "p", "bad answer", "FAIL")]
    result = judge_check(
        kit, llm_judge_spec("q", rubric, client), client, cases, fixture_name="mine"
    )
    assert (
        result.accuracy == 1.0
        and result.adversarial_accuracy is None
        and result.fixture_name == "mine"
    )
