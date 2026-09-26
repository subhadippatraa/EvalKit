"""Remaining edge and defence-in-depth paths of the platform modules."""

import importlib.metadata
import os
from datetime import UTC, datetime

import pytest
from conftest import case, judged

from evalkit import (
    CaseOutcome,
    EvalFailure,
    EvaluatorOutcome,
    EvaluatorSpec,
    Failure,
    RunAttempt,
    RunConfig,
    RunError,
)
from evalkit.calls import CallRunner, RetryPolicy
from evalkit.errors import ConfigError, ScoringError
from evalkit.evaluators import EvalContext, EvalInput, build, llm_judge_spec, resolve
from evalkit.llm import LLMResponse
from evalkit.models import Rubric
from evalkit.report import clip, render_report, write_report


def unit():
    return CallRunner(RetryPolicy(base_s=0.001, cap_s=0.002)).unit()


def make(kind, **params):
    return build(resolve(EvaluatorSpec(kind=kind, name="e", params=params)))


# --- evaluators -------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["strip", ["nope"], 5])
def test_invalid_normalize_is_a_config_error(bad):
    with pytest.raises(ConfigError, match="normalize"):
        resolve(EvaluatorSpec(kind="exact_match", name="e", params={"normalize": bad}))


def test_json_schema_config_limits():
    with pytest.raises(ConfigError, match="larger than"):
        resolve(
            EvaluatorSpec(
                kind="json_schema", name="j", params={"schema": {"description": "x" * 200_000}}
            )
        )
    with pytest.raises(ConfigError, match="strip_code_fence"):
        resolve(
            EvaluatorSpec(
                kind="json_schema", name="j", params={"schema": {}, "strip_code_fence": "yes"}
            )
        )


def test_the_validator_itself_never_fetches_a_remote_reference():
    """Config-time checks refuse remote refs; this is the second line: even if one got through,
    resolving it raises instead of touching the network."""
    from evalkit.evaluators import _validator

    v = _validator({"$ref": "http://example.invalid/schema.json"})
    with pytest.raises(Exception, match="(?i)unresolvable|no such|resource"):
        list(v.iter_errors({}))


def test_a_recursion_error_inside_validation_is_a_failed_validation(monkeypatch):
    ev = make("json_schema", schema={"type": "object"})

    class Exploding:
        def iter_errors(self, instance):
            raise RecursionError

    monkeypatch.setattr(ev, "_validator", Exploding())
    out = ev.evaluate(EvalInput("k", "p", "{}"), unit())
    assert out.metrics == {"valid": 0.0} and "too deeply nested" in out.detail["errors"][0]


def test_citation_check_needs_retrieved_ids():
    with pytest.raises(EvalFailure) as info:
        make("citation_check").evaluate(EvalInput("k", "p", "see [1]"), unit())
    assert info.value.kind == "missing_field"


def test_precision_and_recall_are_omitted_when_undefined_for_citations():
    ev = make("citation_check")
    out = ev.evaluate(EvalInput("k", "p", "see [9]", retrieved=["d1"], relevance={"d1": 1}), unit())
    # nothing resolved: precision is undefined (no cited document); recall is defined (0 of 1)
    assert out.metrics == {"citation_validity": 0.0, "citation_recall": 0.0}
    out = ev.evaluate(EvalInput("k", "p", "see [1]", retrieved=["d1"], relevance={"d1": 0}), unit())
    assert out.metrics == {
        "citation_validity": 1.0,
        "citation_precision": 0.0,
    }  # no relevant docs: no recall


def test_a_scoring_bug_in_the_judge_path_is_an_evaluator_internal_error(monkeypatch):
    class Client:
        provider, model = "fake", "j"

        def call(self, req):
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use")

    client = Client()
    ev = build(
        llm_judge_spec("q", Rubric.from_dict({"quality": "Good?"}), client), EvalContext((client,))
    )

    def broken(self, raw):
        raise ScoringError("inconsistent")

    monkeypatch.setattr(Rubric, "score", broken)
    with pytest.raises(EvalFailure) as info:
        ev.evaluate(EvalInput("k", "p", "o"), unit())
    assert (info.value.failure_class.value, info.value.kind) == ("evaluator", "internal_error")


# --- report -----------------------------------------------------------------------------------


def run_with_data(kit):
    kit.datasets.import_cases("qa", [case("a", output="o", reference="r")])
    spec = resolve(EvaluatorSpec(kind="exact_match", name="em"))
    run = kit.runs.create("qa", RunConfig(evaluators=[spec]))
    kit.runs.start(run.id)
    cr = kit.runs.record_case_result(
        run.id, "a", CaseOutcome.complete("o", retrieved=["d1", "d2"]),
        attempts=[
            RunAttempt.succeeded(
                1, datetime.now(UTC), 5, provider="p", model="m1",
                input_tokens=1000, output_tokens=500,
            )
        ],
    )  # fmt: skip
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=spec.key,
            status="failed",
            failure=Failure(failure_class="evaluator", kind="refused", message="judge said no"),
        ),
    )
    kit.runs.transition(run.id, "succeeded")
    return run


def test_report_shows_retrieved_docs_evaluator_failures_and_estimated_cost(kit):
    run = run_with_data(kit)
    page = render_report(kit, run.id, prices={"m1": (3.0, 15.0)}, min_coverage=0.0)
    assert "retrieved: d1, d2" in page and "judge said no" in page and "evaluator.refused" in page
    assert "estimated cost $0.0105" in page and "an ESTIMATE" in page


def test_clip_of_nothing_is_a_dash():
    assert clip(None) == "—"


def test_a_calibrated_judge_shows_its_agreement_statistics(kit):
    kit.datasets.import_cases("qa", [case(f"c{i:02}", output="o") for i in range(40)])
    spec = EvaluatorSpec(kind="llm_judge", name="q", params={"provider": "p", "model": "m"})
    run = kit.runs.create("qa", RunConfig(evaluators=[spec]))
    kit.runs.start(run.id)
    for i in range(40):
        cr = kit.runs.record_case_result(run.id, f"c{i:02}", CaseOutcome.complete("o"))
        v = "PASS" if i % 4 else "FAIL"
        kit.runs.record_evaluator_result(
            cr.id,
            EvaluatorOutcome(
                evaluator_key=spec.key, status="ok", verdict=v, metrics={"score": 1.0}
            ),
        )
        kit.reviews.add(
            run.id, f"c{i:02}", reviewer="a", verdict=v, evaluator_key=spec.key, sample="random"
        )
    kit.runs.transition(run.id, "succeeded")
    page = render_report(kit, run.id)
    assert "accuracy 100.0% (n=40)" in page and "UNCALIBRATED" not in page


def test_a_failed_write_leaves_neither_the_target_nor_a_temp_file(kit, tmp_path, monkeypatch):
    run = run_with_data(kit)

    def boom(*a):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_report(kit, run.id, tmp_path / "r.html")
    assert not (tmp_path / "r.html").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".r.html")]


def test_a_case_missing_from_the_evidence_selection_is_skipped(kit):
    from evalkit.report import _case_block

    run = run_with_data(kit)
    assert _case_block(kit, run.id, kit.runs.get(run.id).dataset_version_id, 0, "zzz") == ""


# --- analysis / store / misc ------------------------------------------------------------------


def test_the_version_falls_back_when_the_package_is_not_installed(monkeypatch):
    from evalkit.analysis import evalkit_version

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert evalkit_version() == "unknown"


def test_non_finite_prices_are_refused(kit):
    from evalkit.analysis import summarize

    run = run_with_data(kit)
    with pytest.raises(ValueError, match="finite"):
        summarize(kit, run.id, prices={"m1": (float("inf"), 1.0)})


@pytest.mark.parametrize("tag", ["", "x" * 200, "bad\ntag", 5])
def test_tags_are_validated(kit, tag):
    run = run_with_data(kit)
    with pytest.raises(RunError, match="printable"):
        kit.store.add_run_tag(run.id, tag)


def test_no_summary_yet_is_none_and_a_run_without_a_tag_has_no_baseline(kit):
    run = run_with_data(kit)
    assert kit.store.latest_summary(run.id) is None
    assert kit.store.latest_run_with_tag("main") is None
    assert kit.store.latest_run_with_tag("main", kit.runs.get(run.id).dataset_version_id) is None


def test_calibration_excludes_cases_whose_evaluator_had_no_verdict(kit):
    run = run_with_data(kit)  # the only evaluator result failed: no verdict to compare
    key = kit.runs.get(run.id).config.evaluators[0].key
    kit.reviews.add(run.id, "a", reviewer="a", verdict="PASS", evaluator_key=key, sample="random")
    c = kit.reviews.calibrate(run.id, key, n_min=1)
    assert c.n_paired == 0 and c.excluded == {"no_verdict": 1}
