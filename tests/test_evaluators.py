"""Case evaluators: metric semantics, identity, purity, numeric safety, failure attribution."""

import math
import sys

import pytest
from conftest import judged
from hypothesis import given, settings
from hypothesis import strategies as st

from evalkit import ConfigError, EvalFailure, EvaluatorSpec, FailureClass, Rubric
from evalkit.calls import CallRunner, RetryPolicy
from evalkit.evaluators import (
    EVALUATOR_KINDS,
    EvalContext,
    EvalInput,
    NotApplicable,
    build,
    llm_judge_spec,
    resolve,
    retrieval_metrics,
)
from evalkit.llm import STOP_END, LLMResponse, Usage
from evalkit.runs import EvaluatorOutcome


def make(kind, name="e", **params):
    return build(resolve(EvaluatorSpec(kind=kind, name=name, params=params)))


def unit():
    return CallRunner(RetryPolicy(base_s=0.001, cap_s=0.002)).unit()


def inp(output="out", **kw):
    return EvalInput(case_key="k", prompt="p", output=output, **kw)


def run(ev, i, u=None):
    return ev.evaluate(i, u or unit())


def as_outcome(ev, out):
    """The engine's step: an EvalOutcome becomes a validated EvaluatorOutcome (finite metrics)."""
    return EvaluatorOutcome(
        evaluator_key=ev.key,
        status="ok",
        metrics=out.metrics,
        verdict=out.verdict,
        detail=out.detail,
    )


# --- registry and identity --------------------------------------------------------------------


def test_the_kinds_are_the_designs_set_and_nothing_more():
    assert set(EVALUATOR_KINDS) == {
        "exact_match", "regex", "json_schema", "retrieval", "citation_check", "llm_judge",
    }  # fmt: skip


def test_no_generic_similarity_metrics_exist():
    """Deliberately absent (design 6.2): BLEU/ROUGE/BERTScore/embedding similarity."""
    for name in ("bleu", "rouge", "bertscore", "embedding_similarity", "semantic_similarity"):
        with pytest.raises(ConfigError, match="unknown evaluator kind"):
            resolve(EvaluatorSpec(kind=name, name="x"))


def test_resolve_fills_defaults_and_is_idempotent():
    s = resolve(EvaluatorSpec(kind="exact_match", name="em"))
    assert s.params == {"normalize": []} and s.version == 1
    assert resolve(s) == s and resolve(s).key == s.key


def test_a_wrong_version_and_unknown_params_are_refused():
    with pytest.raises(ConfigError, match="is at version 1"):
        resolve(EvaluatorSpec(kind="regex", name="r", version=2, params={"pattern": "x"}))
    with pytest.raises(ConfigError, match="unknown parameter"):
        resolve(EvaluatorSpec(kind="exact_match", name="em", params={"ignore_case": True}))


def test_build_refuses_a_spec_that_is_not_in_canonical_form():
    with pytest.raises(ConfigError, match="not in canonical form"):
        build(EvaluatorSpec(kind="exact_match", name="em"))  # defaults not filled: key differs


def test_normalization_steps_are_order_independent_in_the_identity():
    a = resolve(
        EvaluatorSpec(kind="exact_match", name="a", params={"normalize": ["strip", "casefold"]})
    )
    b = resolve(
        EvaluatorSpec(
            kind="exact_match", name="a", params={"normalize": ["casefold", "strip", "strip"]}
        )
    )
    assert a.key == b.key and a.params["normalize"] == ["casefold", "strip"]


def test_every_configuration_change_changes_the_key():
    keys = {
        resolve(EvaluatorSpec(kind="exact_match", name="a")).key,
        resolve(EvaluatorSpec(kind="exact_match", name="a", params={"normalize": ["strip"]})).key,
        resolve(EvaluatorSpec(kind="regex", name="a", params={"pattern": "x"})).key,
        resolve(EvaluatorSpec(kind="regex", name="a", params={"pattern": "y"})).key,
        resolve(EvaluatorSpec(kind="regex", name="a", params={"pattern": "y", "flags": ["i"]})).key,
        resolve(EvaluatorSpec(kind="retrieval", name="a", params={"k": [5]})).key,
        resolve(EvaluatorSpec(kind="retrieval", name="a", params={"k": [5, 10]})).key,
    }
    assert len(keys) == 7


# --- exact_match ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "steps,output,reference,match",
    [
        ([], "Paris", "Paris", True),
        ([], "Paris", "paris", False),
        (["casefold"], "Paris", "PARIS", True),
        ([], " Paris ", "Paris", False),
        (["strip"], " Paris\n", "Paris", True),
        (["collapse_whitespace"], "a   b\n c", "a b c", True),
        (["nfkc"], "ﬁ", "fi", True),  # ligature
        ([], "ﬁ", "fi", False),
        (["nfkc", "casefold", "collapse_whitespace"], "  Ａ B ", "a b", True),
        ([], "", "", True),
        ([], "", "x", False),
    ],
)
def test_exact_match_semantics(steps, output, reference, match):
    ev = make("exact_match", normalize=steps)
    out = run(ev, inp(output, reference=reference))
    assert out.metrics == {"match": 1.0 if match else 0.0}
    assert out.verdict == ("PASS" if match else "FAIL")


def test_exact_match_reports_where_the_difference_starts_and_needs_a_reference():
    out = run(make("exact_match"), inp("abcXef", reference="abcYef"))
    assert out.detail["first_difference"] == 3
    with pytest.raises(EvalFailure) as info:
        run(make("exact_match"), inp("x"))
    assert (info.value.failure_class, info.value.kind) == (FailureClass.INPUT, "missing_field")
    assert make("exact_match").requires == {"reference"}


# --- regex ------------------------------------------------------------------------------------


def test_regex_search_fullmatch_and_flags():
    assert run(make("regex", pattern=r"\d{3}"), inp("call 555 now")).metrics == {"match": 1.0}
    assert run(make("regex", pattern=r"\d{3}", mode="fullmatch"), inp("call 555 now")).metrics == {
        "match": 0.0
    }
    assert run(make("regex", pattern="hello", flags=["i"]), inp("HeLLo")).metrics == {"match": 1.0}
    out = run(make("regex", pattern=r"(\d+)-(\d+)"), inp("id 12-34 ok"))
    assert out.detail["span"] == [3, 8] and out.detail["text"] == "12-34"


def test_regex_evidence_is_bounded():
    out = run(make("regex", pattern="a+"), inp("a" * 5000))
    assert len(out.detail["text"]) == 200


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"pattern": ""},
        {"pattern": "("},
        {"pattern": "x" * 2001},
        {"pattern": "x", "flags": ["z"]},
        {"pattern": "x", "mode": "prefix"},
        {"pattern": 5},
    ],
)
def test_invalid_regex_configuration_is_refused_before_any_run(params):
    with pytest.raises(ConfigError):
        resolve(EvaluatorSpec(kind="regex", name="r", params=params))


# --- json_schema (optional dependency) --------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name"],
    "additionalProperties": False,
}


def test_json_schema_valid_and_invalid_outputs_are_quality_signals_not_failures():
    ev = make("json_schema", schema=SCHEMA)
    good = run(ev, inp('{"name": "a", "age": 3}'))
    assert (good.metrics, good.verdict) == ({"valid": 1.0}, "PASS")
    bad = run(ev, inp('{"age": -1, "extra": 1}'))
    assert (bad.metrics, bad.verdict) == ({"valid": 0.0}, "FAIL")
    assert bad.detail["parsed"] and bad.detail["error_count"] == 3
    assert {e["path"] for e in bad.detail["errors"]} == {"/", "/age"}
    broken = run(ev, inp("not json at all"))
    assert broken.metrics == {"valid": 0.0} and broken.detail["parsed"] is False


@pytest.mark.parametrize("text", ['{"name": NaN}', '{"name": "a", "name": "b"}', "", "{"])
def test_strict_json_parsing_rejects_nan_duplicate_keys_and_garbage(text):
    out = run(make("json_schema", schema={"type": "object"}), inp(text))
    assert out.metrics == {"valid": 0.0}


def test_code_fences_are_only_stripped_when_declared():
    fenced = '```json\n{"name": "a"}\n```'
    assert run(make("json_schema", schema=SCHEMA), inp(fenced)).metrics == {"valid": 0.0}
    assert run(make("json_schema", schema=SCHEMA, strip_code_fence=True), inp(fenced)).metrics == {
        "valid": 1.0
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "http://example.com/schema.json"},
        {"properties": {"a": {"$ref": "https://evil.example/x"}}},
        {"items": [{"$ref": "file:///etc/passwd"}]},
        {"$ref": "other.json"},
    ],
)
def test_remote_references_are_refused_at_configuration_time(schema):
    with pytest.raises(ConfigError, match="remote reference"):
        resolve(EvaluatorSpec(kind="json_schema", name="j", params={"schema": schema}))


def test_local_references_work_and_nothing_is_fetched():
    schema = {
        "$defs": {"n": {"type": "integer"}},
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/n"}},
    }
    ev = make("json_schema", schema=schema)
    assert run(ev, inp('{"a": 1}')).metrics == {"valid": 1.0}
    assert run(ev, inp('{"a": "x"}')).metrics == {"valid": 0.0}


@pytest.mark.parametrize("schema", [{"type": "nonsense"}, "str", {"required": "x"}])
def test_an_invalid_schema_is_a_config_error(schema):
    with pytest.raises(ConfigError):
        resolve(EvaluatorSpec(kind="json_schema", name="j", params={"schema": schema}))


def test_a_pathologically_nested_output_is_a_failed_validation_not_a_crash():
    out = run(make("json_schema", schema={}), inp("[" * 200_000 + "]" * 200_000))
    assert out.metrics == {"valid": 0.0} and out.detail["parsed"] is False


def test_without_the_optional_dependency_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)  # makes `import jsonschema` fail
    with pytest.raises(ConfigError, match=r"evalkit\[jsonschema\]"):
        resolve(EvaluatorSpec(kind="json_schema", name="j", params={"schema": {"type": "object"}}))


def test_the_core_does_not_import_jsonschema_unless_used():
    import subprocess

    code = "import sys, evalkit, evalkit.evaluators; assert 'jsonschema' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# --- retrieval: hand-computed fixtures --------------------------------------------------------

L2, L3, L4, L5 = (math.log2(n) for n in (2, 3, 4, 5))


def test_retrieval_worked_example():
    relevance = {"a": 3, "b": 2, "c": 0, "d": 1}
    m = retrieval_metrics(["c", "a", "x", "b"], relevance, [1, 3, 10])
    assert m["mrr"] == 0.5  # first relevant (a) is at rank 2
    assert (m["recall@1"], m["precision@1"], m["hit@1"], m["ndcg@1"]) == (0, 0, 0, 0)
    assert m["recall@3"] == pytest.approx(1 / 3) and m["precision@3"] == pytest.approx(1 / 3)
    assert m["hit@3"] == 1
    idcg3 = 7 / L2 + 3 / L3 + 1 / L4  # ideal order: grades 3, 2, 1
    assert m["ndcg@3"] == pytest.approx((7 / L3) / idcg3)
    assert m["recall@10"] == pytest.approx(2 / 3)
    assert m["precision@10"] == pytest.approx(0.2)  # k is the denominator even for short lists
    assert m["ndcg@10"] == pytest.approx((7 / L3 + 3 / L5) / idcg3)


def test_a_perfect_ranking_scores_one_and_unretrieved_relevant_docs_lower_ndcg():
    relevance = {"a": 2, "b": 1}
    perfect = retrieval_metrics(["a", "b"], relevance, [2])
    assert perfect["ndcg@2"] == 1.0 and perfect["recall@2"] == 1.0 and perfect["mrr"] == 1.0
    missing = retrieval_metrics(["a", "z"], relevance, [2])  # b was never retrieved
    assert (
        missing["ndcg@2"] == pytest.approx((3 / L2) / (3 / L2 + 1 / L3)) and missing["ndcg@2"] < 1
    )
    assert missing["recall@2"] == 0.5


def test_duplicates_are_collapsed_and_unlabelled_documents_are_not_relevant():
    m = retrieval_metrics(["a", "a", "u", "a"], {"a": 1}, [3])
    assert m["precision@3"] == pytest.approx(1 / 3)  # ranking is [a, u]; hits = 1 of k = 3
    assert m["mrr"] == 1.0


def test_no_retrieved_documents_scores_zero_but_no_relevant_documents_is_undefined():
    m = retrieval_metrics([], {"a": 1}, [5])
    assert all(v == 0 for v in m.values())
    for relevance in ({}, {"a": 0}, {"a": 0, "b": 0}):
        with pytest.raises(NotApplicable, match="no relevant documents"):
            retrieval_metrics(["a"], relevance, [5])


def test_the_retrieval_evaluator_reads_the_targets_documents_and_the_labels():
    ev = make("retrieval", k=[1, 5])
    out = run(ev, inp(retrieved=["a", "b"], relevance={"b": 1}))
    assert out.metrics["mrr"] == 0.5 and out.metrics["recall@5"] == 1.0
    assert out.verdict is None  # a ranking metric has no PASS/FAIL of its own
    assert set(out.metrics) == {
        "mrr",
        *(f"{m}@{k}" for m in ("recall", "precision", "hit", "ndcg") for k in (1, 5)),
    }
    assert ev.requires == {"relevance", "retrieved"}
    with pytest.raises(NotApplicable):
        run(ev, inp(retrieved=["a"], relevance={"a": 0}))
    with pytest.raises(EvalFailure):
        run(ev, inp(relevance={"a": 1}))


@pytest.mark.parametrize("k", [0, -1, 101, "5", [], [1.5], [True], None])
def test_invalid_k_is_refused(k):
    with pytest.raises(ConfigError):
        resolve(EvaluatorSpec(kind="retrieval", name="r", params={"k": k}))


ids = st.text("abcdefgh", min_size=1, max_size=1)


@settings(max_examples=200, deadline=None)
@given(
    retrieved=st.lists(ids, max_size=12),
    relevance=st.dictionaries(ids, st.integers(0, 100), min_size=1, max_size=8),
    ks=st.lists(st.integers(1, 15), min_size=1, max_size=4),
)
def test_property_retrieval_metrics_are_finite_bounded_and_monotone(retrieved, relevance, ks):
    try:
        m = retrieval_metrics(retrieved, relevance, ks)
    except NotApplicable:
        assert not any(g > 0 for g in relevance.values())
        return
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 + 1e-12 for v in m.values())
    ordered = sorted(set(ks))
    for a, b in zip(ordered, ordered[1:], strict=False):
        assert m[f"recall@{a}"] <= m[f"recall@{b}"] + 1e-12
        assert m[f"hit@{a}"] <= m[f"hit@{b}"]
    assert m == retrieval_metrics(list(retrieved), dict(relevance), list(ks))  # deterministic


@settings(max_examples=100, deadline=None)
@given(relevance=st.dictionaries(ids, st.integers(1, 100), min_size=1, max_size=8))
def test_property_the_ideal_ranking_has_ndcg_one(relevance):
    ideal = sorted(relevance, key=lambda d: -relevance[d])
    m = retrieval_metrics(ideal, relevance, [len(ideal)])
    assert m[f"ndcg@{len(ideal)}"] == pytest.approx(1.0) and m["mrr"] == 1.0


# --- citation_check ---------------------------------------------------------------------------


def test_citation_check_index_style():
    ev = make("citation_check")
    out = run(
        ev,
        inp(
            "Paris [1] is big [2]; see [7] and [1].",
            retrieved=["d1", "d2", "d3"],
            relevance={"d1": 2, "d3": 1},
        ),
    )
    assert out.detail["cited"] == ["1", "2", "7"] and out.detail["unresolved"] == ["7"]
    assert out.metrics["citation_validity"] == pytest.approx(2 / 3)
    assert out.metrics["citation_precision"] == pytest.approx(1 / 2)  # cited {d1, d2}; relevant d1
    assert out.metrics["citation_recall"] == pytest.approx(1 / 2)  # relevant {d1, d3}; cited d1
    assert out.verdict == "FAIL" and out.detail["documents"] == ["d1", "d2"]


def test_citation_check_id_style_all_valid_passes_and_no_labels_means_no_precision():
    ev = make("citation_check", style="id", marker=r"\((doc-\d+)\)")
    out = run(ev, inp("x (doc-1) y (doc-2)", retrieved=["doc-1", "doc-2"]))
    assert out.metrics == {"citation_validity": 1.0} and out.verdict == "PASS"


def test_no_citations_is_not_applicable_never_zero_or_one():
    with pytest.raises(NotApplicable, match="no citations"):
        run(make("citation_check"), inp("plain prose", retrieved=["d"]))


def test_index_style_rejects_non_ascii_digits_and_out_of_range_markers():
    out = run(make("citation_check"), inp("[١] [0] [4]", retrieved=["a", "b", "c"]))
    assert out.metrics["citation_validity"] == 0.0


@pytest.mark.parametrize(
    "params",
    [
        {"style": "name"},
        {"marker": "("},
        {"marker": r"\[\d+\]"},
        {"marker": r"(a)(b)"},
        {"marker": ""},
    ],
)
def test_invalid_citation_configuration_is_refused(params):
    with pytest.raises(ConfigError):
        resolve(EvaluatorSpec(kind="citation_check", name="c", params=params))


def test_citation_check_requires_retrieved_and_does_not_claim_to_judge_support():
    ev = make("citation_check")
    assert ev.requires == {"retrieved"}
    assert "does NOT judge" in type(ev).__doc__


# --- LLM judge --------------------------------------------------------------------------------

RUBRIC = Rubric.from_dict({"correctness": "Right?", "clarity": "Clear?"})


class FakeJudgeClient:
    provider = "fake"
    model = "judge-1"

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def call(self, req):
        self.requests.append(req)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def tool_response(payload, **kw):
    return LLMResponse(
        payload=payload, stop_reason="tool_use", usage=Usage(100, 20), request_id="rq", **kw
    )


def judge(client, rubric=RUBRIC, **kw):
    spec = llm_judge_spec("quality", rubric, client, **kw)
    return build(spec, EvalContext(clients=[client]))


def test_the_judge_emits_score_criteria_verdict_and_evidence():
    client = FakeJudgeClient(tool_response(judged(correctness=5, clarity=3)))
    ev = judge(client)
    u = unit()
    out = run(ev, inp("answer", reference="gold", context="ctx"), u)
    assert out.metrics == {"score": 0.75, "criterion.correctness": 1.0, "criterion.clarity": 0.5}
    assert out.verdict == "PASS" and out.detail["values"] == {"correctness": 5, "clarity": 3}
    assert set(out.detail["reasoning"]) == {"correctness", "clarity"}
    assert out.detail["threshold"] == 0.75 and out.detail["failed_gates"] == []
    (a,) = u.attempts
    assert (a.outcome, a.input_tokens, a.output_tokens, a.request_id, a.provider, a.model) == (
        "ok", 100, 20, "rq", "fake", "judge-1",
    )  # fmt: skip
    assert a.raw is None and a.raw_sha256  # payload hashed, content lives in the evaluator result
    as_outcome(ev, out)  # and it is a valid, finite EvaluatorOutcome


def test_the_judge_sees_prompt_output_reference_and_context_only():
    client = FakeJudgeClient(tool_response(judged(correctness=4, clarity=4)))
    ev = judge(client)
    run(ev, inp("THE-OUTPUT", reference="THE-REF", context=None))
    req = client.requests[0]
    assert req.role == "evaluator" and req.tool.name == "submit_evaluation"
    assert (
        "THE-OUTPUT" in req.user and "THE-REF" in req.user and "context: (not provided)" in req.user
    )
    assert not hasattr(EvalInput("k", "p", "o"), "metadata") and not hasattr(
        EvalInput("k", "p", "o"), "tags"
    )


def test_labels_and_weights_flow_through_the_shared_scoring_code():
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {"name": "tone", "description": "Tone?", "labels": ["BAD", "OK", "GOOD"]},
                {"name": "n", "description": "N?", "scale": [0, 10], "weight": 3},
            ]
        }
    )
    client = FakeJudgeClient(
        tool_response(
            {"tone": {"reasoning": "r", "label": "GOOD"}, "n": {"reasoning": "r", "score": 5}}
        )
    )
    out = run(judge(client, rubric), inp())
    assert out.metrics["criterion.tone"] == 1.0 and out.metrics["criterion.n"] == 0.5
    assert out.metrics["score"] == pytest.approx((1.0 * 1 + 0.5 * 3) / 4)


def test_a_judge_key_covers_rubric_model_temperature_and_prompt_fingerprint():
    a = FakeJudgeClient()
    base = llm_judge_spec("q", RUBRIC, a)
    assert (
        base.params["prompt_version"] and base.params["rubric_content_hash"] == RUBRIC.content_hash
    )
    other_model = FakeJudgeClient()
    other_model.model = "judge-2"
    assert llm_judge_spec("q", RUBRIC, other_model).key != base.key
    assert llm_judge_spec("q", RUBRIC, a, temperature=0.5).key != base.key
    assert llm_judge_spec("q", Rubric.from_dict({"correctness": "Right?"}), a).key != base.key
    assert llm_judge_spec("q", RUBRIC, a, max_tokens=99).key != base.key
    assert llm_judge_spec("q", RUBRIC, a).key == base.key  # deterministic


def test_the_judge_needs_a_matching_client_and_a_valid_config():
    spec = llm_judge_spec("q", RUBRIC, FakeJudgeClient())
    with pytest.raises(ConfigError, match="no LLM client"):
        build(spec, EvalContext())
    stranger = FakeJudgeClient()
    stranger.model = "other"
    with pytest.raises(ConfigError, match="no LLM client"):
        build(spec, EvalContext(clients=[stranger]))
    good = {"rubric": RUBRIC.model_dump(mode="json"), "provider": "p", "model": "m"}
    bad_params = [
        {},
        {**good, "rubric": {"criteria": []}},
        {k: v for k, v in good.items() if k != "model"},
        {**good, "temperature": -1},
        {**good, "max_tokens": 0},
    ]
    for params in bad_params:
        with pytest.raises(ConfigError):
            resolve(EvaluatorSpec(kind="llm_judge", name="q", params=params))


def test_malformed_judge_output_is_retried_once_with_feedback_and_can_succeed():
    client = FakeJudgeClient(
        tool_response(judged(correctness=9, clarity=3)),  # 9 is outside the 1-5 scale
        tool_response(judged(correctness=5, clarity=5)),
    )
    u = unit()
    out = run(judge(client), inp(), u)
    assert out.metrics["score"] == 1.0
    assert [a.outcome for a in u.attempts] == ["failed", "ok"]
    assert (
        u.attempts[0].error_class is FailureClass.EVALUATOR
        and u.attempts[0].error_kind == "invalid_output"
    )
    assert "outside scale" in u.attempts[0].error and u.attempts[0].raw
    retry = client.requests[1].user
    assert (
        "rejected" in retry
        and "outside scale 1-5" in retry
        and retry.startswith(client.requests[0].user)
    )


def test_malformed_output_twice_is_an_evaluator_failure_with_both_rejections_kept():
    client = FakeJudgeClient(
        tool_response(judged(correctness=9, clarity=3)), tool_response({"garbage": 1})
    )
    u = unit()
    with pytest.raises(EvalFailure) as info:
        run(judge(client), inp(), u)
    assert (info.value.failure_class, info.value.kind) == (FailureClass.EVALUATOR, "invalid_output")
    assert len(u.attempts) == 2 and all(a.raw for a in u.attempts)


@pytest.mark.parametrize(
    "resp,cls,kind",
    [
        (
            LLMResponse(payload=judged(correctness=5, clarity=5), stop_reason="max_tokens"),
            "evaluator",
            "truncated",
        ),
        (LLMResponse(stop_reason="content_filtered"), "evaluator", "refused"),
        (LLMResponse(stop_reason=STOP_END), "evaluator", "invalid_output"),  # no tool call
        (LLMResponse(stop_reason="model_context_window_exceeded"), "input", "oversize"),
    ],
)
def test_unusable_judge_responses_are_evaluator_or_input_failures_never_target_failures(
    resp, cls, kind
):
    u = unit()
    with pytest.raises(EvalFailure) as info:
        run(judge(FakeJudgeClient(resp, resp)), inp(), u)
    assert (info.value.failure_class.value, info.value.kind) == (cls, kind)
    assert info.value.failure_class is not FailureClass.TARGET


def test_provider_failures_are_infrastructure_and_retry_then_succeed():
    client = FakeJudgeClient(
        EvalFailure("infrastructure", "rate_limited", "429", retry_after_s=0.001),
        tool_response(judged(correctness=4, clarity=4)),
    )
    u = unit()
    assert run(judge(client), inp(), u).verdict == "PASS"
    assert [a.error_kind for a in u.attempts] == ["rate_limited", None]


def test_a_rejected_judge_model_is_evaluator_bad_request_and_systemic():
    client = FakeJudgeClient(EvalFailure("evaluator", "bad_request", "no such model"))
    with pytest.raises(EvalFailure) as info:
        run(judge(client), inp())
    assert info.value.systemic and len(client.requests) == 1


# --- must-pass criteria (design 6.3) ---------------------------------------------------------

GATED = Rubric.model_validate(
    {
        "criteria": [
            {"name": "safety", "description": "Safe?", "must_pass": True},
            {"name": "helpful", "description": "Helpful?", "weight": 5},
        ],
        "threshold": 0.5,
    }
)


def test_a_failed_must_pass_criterion_fails_the_verdict_even_with_a_high_mean():
    client = FakeJudgeClient(
        tool_response(
            {
                "safety": {"reasoning": "unsafe", "score": 1},
                "helpful": {"reasoning": "r", "score": 5},
            }
        )
    )
    out = run(judge(client, GATED), inp())
    assert (
        out.metrics["score"] == pytest.approx(5 / 6) and out.verdict == "FAIL"
    )  # 5/6 is well above 0.5
    assert out.detail["failed_gates"] == ["safety"]


def test_must_pass_floor_is_configurable_and_a_met_floor_passes():
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {"name": "safety", "description": "S?", "must_pass": True, "min_normalized": 0.75}
            ],
            "threshold": 0.1,
        }
    )
    for score, verdict in ((3, "FAIL"), (4, "PASS"), (5, "PASS")):
        out = run(
            judge(
                FakeJudgeClient(tool_response({"safety": {"reasoning": "r", "score": score}})),
                rubric,
            ),
            inp(),
        )
        assert out.verdict == verdict
    assert rubric.criteria[0].floor == 0.75 and GATED.criteria[0].floor == 0.5


def test_must_pass_fields_are_validated_and_do_not_change_existing_rubric_hashes():
    with pytest.raises(ValueError, match="only applies to a `must_pass`"):
        Rubric.model_validate(
            {"criteria": [{"name": "a", "description": "A?", "min_normalized": 0.5}]}
        )
    with pytest.raises(ValueError):
        Rubric.model_validate(
            {
                "criteria": [
                    {"name": "a", "description": "A?", "must_pass": True, "min_normalized": 1.5}
                ]
            }
        )
    assert (
        Rubric.from_dict({"a": "A?"}).version == "89e795732810"
    )  # pre-must-pass rubrics: same hash
    plain = Rubric.model_validate({"criteria": [{"name": "a", "description": "A?"}]})
    gated = Rubric.model_validate(
        {"criteria": [{"name": "a", "description": "A?", "must_pass": True}]}
    )
    assert plain.content_hash != gated.content_hash


def test_the_legacy_result_model_accepts_a_gate_failure_verdict_and_rejects_a_wrong_one():
    from evalkit import EvaluationResult
    from evalkit.models import CriterionScore

    scores = {
        "safety": CriterionScore(reasoning="r", score=1),
        "helpful": CriterionScore(reasoning="r", score=5),
    }
    common = dict(
        status="ok", prompt="p", model_output="o", rubric=GATED, rubric_version=GATED.version,
        judge_provider="x", judge_model="m", judge_temperature=0.0, judge_prompt_version="v",
        scores=scores, overall_score=5 / 6,
    )  # fmt: skip
    EvaluationResult(verdict="FAIL", **common)
    with pytest.raises(ValueError, match="contradicts"):
        EvaluationResult(verdict="PASS", **common)


# --- purity / numeric safety ------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(output=st.text(max_size=200), reference=st.text(max_size=200))
def test_property_deterministic_evaluators_are_pure_and_emit_finite_metrics(output, reference):
    for ev in (
        make("exact_match", normalize=["nfkc", "casefold", "strip"]),
        make("regex", pattern=r"\w+"),
    ):
        first = run(ev, inp(output, reference=reference))
        assert run(ev, inp(output, reference=reference)) == first
        assert all(math.isfinite(v) for v in first.metrics.values())
        as_outcome(ev, first)


def test_deterministic_evaluators_make_no_calls():
    u = unit()
    run(make("exact_match"), inp("a", reference="a"), u)
    run(make("regex", pattern="a"), inp("a"), u)
    assert u.attempts == []
