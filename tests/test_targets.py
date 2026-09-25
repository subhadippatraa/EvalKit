"""Targets: normalized output, leak-free input, classified failures, an Attempt per call."""

import dataclasses
import time

import pytest
from conftest import case

from evalkit import ConfigError, EvalFailure, EvaluationCase, FailureClass, RunAttempt
from evalkit.calls import CallRunner, RetryPolicy
from evalkit.failures import Failure
from evalkit.llm import STOP_END, LLMResponse, Usage
from evalkit.targets import (
    CallableTarget,
    ModelTarget,
    PrecomputedTarget,
    ReuseTarget,
    TargetInput,
    TargetOutput,
    coerce_output,
    spec_of,
    view,
)


def unit(policy=None):
    return CallRunner(policy or RetryPolicy(base_s=0.001, cap_s=0.002, timeout_s=2.0)).unit()


def rich_case():
    return EvaluationCase.model_validate(
        case(
            "k",
            output="the system's own answer",
            reference="SECRET GOLD ANSWER",
            context="ctx",
            retrieved=["d1"],
            relevance={"d1": 3},
            metadata={"owner": "team-a"},
            tags=["hard"],
        )
    )


# --- leakage: structural -----------------------------------------------------------------------


def test_a_target_input_has_no_field_that_could_carry_evaluator_information():
    names = {f.name for f in dataclasses.fields(TargetInput)}
    assert names == {"case_key", "prompt", "context", "provided", "inherited_failure"}
    for forbidden in ("reference", "relevance", "tags", "metadata", "expected", "labels"):
        assert forbidden not in names


def test_view_never_copies_reference_relevance_tags_or_metadata():
    c = rich_case()
    v = view(c)
    assert (v.case_key, v.prompt, v.context, v.provided) == ("k", c.prompt, "ctx", None)
    flat = repr(v)
    for secret in ("SECRET GOLD ANSWER", "team-a", "hard"):
        assert secret not in flat
    assert "relevance" not in flat and "d1" not in flat


def test_only_the_precomputed_view_carries_the_datasets_own_output():
    c = rich_case()
    assert view(c).provided is None
    v = view(c, provide_output=True)
    assert v.provided == TargetOutput("the system's own answer", retrieved=["d1"])
    assert "SECRET GOLD ANSWER" not in repr(v)  # the reference still never appears
    assert view(EvaluationCase.model_validate(case("n")), provide_output=True).provided is None


def test_a_callable_target_only_ever_sees_the_view():
    seen = []
    t = CallableTarget(lambda inp: seen.append(inp) or "ok")
    t.generate(view(rich_case()), unit())
    (inp,) = seen
    assert isinstance(inp, TargetInput) and inp.provided is None
    assert not hasattr(inp, "reference") and "SECRET" not in repr(inp)


# --- precomputed / reuse ----------------------------------------------------------------------


def test_precomputed_returns_the_provided_output_without_any_call():
    u = unit()
    out = PrecomputedTarget().generate(view(rich_case(), provide_output=True), u)
    assert out.output == "the system's own answer" and out.retrieved == ["d1"]
    assert u.attempts == []  # no external call: nothing to record


def test_precomputed_without_an_output_is_an_input_failure_not_a_score():
    with pytest.raises(EvalFailure) as info:
        PrecomputedTarget().generate(
            view(EvaluationCase.model_validate(case("n")), provide_output=True), unit()
        )
    assert (info.value.failure_class, info.value.kind) == (FailureClass.INPUT, "missing_field")


def test_an_empty_precomputed_output_is_a_real_output():
    c = EvaluationCase.model_validate(case("e", output=""))
    assert PrecomputedTarget().generate(view(c, provide_output=True), unit()).output == ""


def test_reuse_copies_the_source_output_and_identifies_the_source_run():
    t = ReuseTarget("run-1")
    assert t.identity == {"source_run_id": "run-1"} and spec_of(t).kind == "reuse"
    inp = TargetInput("k", "p", provided=TargetOutput("src", retrieved=["d"]))
    assert t.generate(inp, unit()) == inp.provided


def test_reuse_inherits_the_source_failure_class_and_kind():
    src = Failure(failure_class="target", kind="timeout", message="source timed out")
    with pytest.raises(EvalFailure) as info:
        ReuseTarget("run-1").generate(TargetInput("k", "p", inherited_failure=src), unit())
    f = info.value
    assert (f.failure_class, f.kind) == (FailureClass.TARGET, "timeout")
    assert "source run run-1" in str(f) and "source timed out" in str(f)
    with pytest.raises(EvalFailure) as missing:
        ReuseTarget("run-1").generate(TargetInput("k", "p"), unit())
    assert (missing.value.failure_class, missing.value.kind) == (
        FailureClass.INPUT,
        "missing_field",
    )


# --- callable ---------------------------------------------------------------------------------


def test_a_callable_may_return_a_string_or_a_full_target_output():
    assert (
        CallableTarget(lambda i: "plain").generate(TargetInput("k", "p"), unit()).output == "plain"
    )
    out = CallableTarget(
        lambda i: TargetOutput("ans", retrieved=["a", "b"], usage=Usage(3, 4), meta={"v": 1})
    ).generate(TargetInput("k", "p"), unit())
    assert (out.retrieved, out.usage, out.meta) == (["a", "b"], Usage(3, 4), {"v": 1})


def test_every_callable_call_leaves_an_attempt_with_latency():
    u = unit()

    def slow(inp):
        time.sleep(0.03)
        return "ok"

    CallableTarget(slow, name="slowpoke").generate(TargetInput("k", "p"), u)
    (a,) = u.attempts
    assert (a.n, a.outcome, a.provider, a.model) == (1, "ok", "callable", "slowpoke")
    assert a.duration_ms >= 25


def test_an_exception_is_a_target_exception_with_its_type_and_the_failed_attempt_kept():
    u = unit()

    def boom(inp):
        raise ZeroDivisionError("division by zero in my pipeline")

    with pytest.raises(EvalFailure) as info:
        CallableTarget(boom).generate(TargetInput("k", "p"), u)
    f = info.value
    assert (f.failure_class, f.kind, f.exc_type) == (
        FailureClass.TARGET,
        "exception",
        "ZeroDivisionError",
    )
    assert "division by zero" in str(f) and isinstance(f.__cause__, ZeroDivisionError)
    (a,) = u.attempts
    assert (a.outcome, a.error_class, a.error_kind, a.error_type) == (
        "failed", FailureClass.TARGET, "exception", "ZeroDivisionError",
    )  # fmt: skip
    assert len(u.attempts) == 1  # an exception is not retried


def test_secrets_in_a_target_exception_are_scrubbed():
    def boom(inp):
        raise RuntimeError("auth failed with key sk-abcdefghijklmnopqrstuvwx")

    with pytest.raises(EvalFailure) as info:
        CallableTarget(boom).generate(TargetInput("k", "p"), unit())
    assert "sk-abcdef" not in str(info.value)


def test_a_slow_target_times_out_is_retried_once_and_every_try_is_an_attempt():
    calls = []

    def slow(inp):
        calls.append(1)
        time.sleep(0.3)
        return "late"

    u = unit(RetryPolicy(base_s=0.001, cap_s=0.002, timeout_s=0.05))
    with pytest.raises(EvalFailure) as info:
        CallableTarget(slow).generate(TargetInput("k", "p"), u)
    assert (info.value.failure_class, info.value.kind) == (FailureClass.TARGET, "timeout")
    assert len(calls) == 2 and [a.outcome for a in u.attempts] == ["failed", "failed"]
    assert all(
        a.error_class is FailureClass.TARGET and a.error_kind == "timeout" for a in u.attempts
    )
    assert all(
        50 <= a.duration_ms < 250 for a in u.attempts
    )  # bounded by the timeout, not the sleep


def test_a_target_that_times_out_once_then_answers_is_a_success_with_two_attempts():
    state = {"n": 0}

    def sometimes(inp):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(0.3)
        return "answer"

    u = unit(RetryPolicy(base_s=0.001, cap_s=0.002, timeout_s=0.05))
    assert CallableTarget(sometimes).generate(TargetInput("k", "p"), u).output == "answer"
    assert [a.outcome for a in u.attempts] == ["failed", "ok"]


@pytest.mark.parametrize(
    "value",
    [
        None, 42, ["a"], {"output": "x"}, b"bytes",
        TargetOutput(output=3), TargetOutput("x", retrieved="d1"), TargetOutput("x", retrieved=[1]),
        TargetOutput("x", retrieved=[""]), TargetOutput("x", retrieved=["d"] * 1001),
        TargetOutput("x", meta={"n": float("nan")}), TargetOutput("x", meta={"blob": "y" * 5000}),
        TargetOutput("x", meta={"o": object()}), TargetOutput("x", usage=(1, 2)),
        TargetOutput("bad \ud800"),
    ],
)  # fmt: skip
def test_a_wrong_return_shape_is_a_contract_violation(value):
    with pytest.raises(EvalFailure) as info:
        CallableTarget(lambda i: value).generate(TargetInput("k", "p"), unit())
    assert (info.value.failure_class, info.value.kind) == (
        FailureClass.TARGET,
        "contract_violation",
    )


def test_coerce_output_accepts_valid_values_only():
    assert coerce_output("s").output == "s"
    assert coerce_output(TargetOutput("s", retrieved=[])).retrieved == []


def test_empty_output_policy_is_explicit():
    allow = CallableTarget(lambda i: "").generate(TargetInput("k", "p"), unit())
    assert allow.output == ""  # recorded as the truth; evaluators will score it
    with pytest.raises(EvalFailure) as info:
        CallableTarget(lambda i: "", on_empty="fail").generate(TargetInput("k", "p"), unit())
    assert (info.value.failure_class, info.value.kind) == (FailureClass.TARGET, "empty_output")
    with pytest.raises(ConfigError):
        CallableTarget(lambda i: "", on_empty="maybe")


def test_callable_identity_is_what_is_declared_and_unpinned_without_a_fingerprint():
    def fn(i):
        return "x"

    pinned = CallableTarget(fn, name="rag@1", fingerprint="git:abc123")
    unpinned = CallableTarget(fn)
    assert pinned.identity == {"name": "rag@1", "fingerprint": "git:abc123", "on_empty": "allow"}
    assert pinned.pinned and not unpinned.pinned and unpinned.name.endswith("fn")
    assert spec_of(pinned).identity == pinned.identity
    assert spec_of(pinned) != spec_of(CallableTarget(fn, name="rag@1", fingerprint="git:def456"))
    with pytest.raises(ConfigError):
        CallableTarget("not callable")


# --- model ------------------------------------------------------------------------------------


class FakeClient:
    provider = "fake"
    model = "fake-1"

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def call(self, req):
        self.requests.append(req)
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def text(t="an answer", **kw):
    return LLMResponse(
        text=t,
        stop_reason=kw.pop("stop_reason", STOP_END),
        usage=Usage(11, 5),
        request_id="r1",
        **kw,
    )


def test_a_model_target_renders_the_template_and_returns_text_usage_and_meta():
    client = FakeClient(text("Paris"))
    t = ModelTarget(client, "Q: {prompt}\nC: {context}", temperature=0.2, max_tokens=50)
    u = unit()
    out = t.generate(TargetInput("k", "Capital of France?", "geography"), u)
    assert out.output == "Paris" and out.usage == Usage(11, 5)
    assert out.meta == {"stop_reason": "end_turn", "request_id": "r1"}
    (req,) = client.requests
    assert req.user == "Q: Capital of France?\nC: geography" and req.tool is None
    assert (req.temperature, req.max_tokens, req.role) == (0.2, 50, "target")
    (a,) = u.attempts
    assert (a.outcome, a.provider, a.model, a.input_tokens, a.output_tokens, a.request_id) == (
        "ok", "fake", "fake-1", 11, 5, "r1",
    )  # fmt: skip


def test_a_missing_context_renders_as_empty_and_braces_in_content_are_left_alone():
    client = FakeClient(text())
    t = ModelTarget(client, "{prompt}|{context}")
    t.generate(TargetInput("k", "use {curly} and {{x}}", None), unit())
    assert client.requests[0].user == "use {curly} and {{x}}|"


@pytest.mark.parametrize(
    "template",
    [
        "{reference}",
        "{prompt} {relevance}",
        "{0}",
        "{prompt.__class__}",
        "{prompt!r}",
        "{prompt:>10}",
        "{bad",
        "}",
    ],
)
def test_template_fields_are_limited_so_a_reference_cannot_reach_the_prompt(template):
    with pytest.raises(ConfigError):
        ModelTarget(FakeClient(), template)


def test_template_size_and_literal_braces():
    with pytest.raises(ConfigError, match="larger than"):
        ModelTarget(FakeClient(), "x" * (16 * 1024 + 1))
    assert (
        ModelTarget(FakeClient(), "literal {{braces}} {prompt}").render(TargetInput("k", "p"))
        == "literal {braces} p"
    )


def test_model_target_identity_pins_model_template_and_parameters():
    a = ModelTarget(FakeClient(), "{prompt}", temperature=0.0)
    b = ModelTarget(FakeClient(), "{prompt}", temperature=0.5)
    c = ModelTarget(FakeClient(), "Answer: {prompt}")
    assert a.identity["provider"] == "fake" and a.identity["model"] == "fake-1"
    assert len({str(spec_of(x).identity) for x in (a, b, c)}) == 3
    assert a.identity["template_sha256"] and "sk-" not in str(a.identity)


@pytest.mark.parametrize(
    "resp,kind_class,kind",
    [
        (
            text(stop_reason="content_filtered", provider_stop="finish_reason=content_filter"),
            "target",
            "blocked",
        ),
        (text(stop_reason="guardrail_intervened"), "target", "blocked"),
        (text(stop_reason="malformed"), "target", "contract_violation"),
        (text(stop_reason="model_context_window_exceeded"), "input", "oversize"),
    ],
)
def test_unusable_model_responses_are_classified_by_the_targets_role(resp, kind_class, kind):
    with pytest.raises(EvalFailure) as info:
        ModelTarget(FakeClient(resp), "{prompt}").generate(TargetInput("k", "p"), unit())
    assert (info.value.failure_class.value, info.value.kind) == (kind_class, kind)


def test_a_truncated_model_answer_is_still_an_answer_and_is_flagged():
    out = ModelTarget(
        FakeClient(text("cut off mid", stop_reason="max_tokens")), "{prompt}"
    ).generate(TargetInput("k", "p"), unit())
    assert out.output == "cut off mid" and out.meta["stop_reason"] == "max_tokens"


def test_a_model_target_retries_throttling_and_records_every_call():
    client = FakeClient(
        EvalFailure("infrastructure", "rate_limited", "429", retry_after_s=0.001), text("ok now")
    )
    u = unit()
    assert ModelTarget(client, "{prompt}").generate(TargetInput("k", "p"), u).output == "ok now"
    assert [(a.n, a.outcome, a.error_kind) for a in u.attempts] == [
        (1, "failed", "rate_limited"), (2, "ok", None),
    ]  # fmt: skip


def test_a_bad_model_id_is_input_bad_config_and_systemic_for_a_target():
    from evalkit.calls import RunGuard

    guard = RunGuard()
    runner = CallRunner(RetryPolicy(base_s=0.001, cap_s=0.002), guard=guard)
    bad = EvalFailure("input", "bad_config", "Bedrock ResourceNotFoundException: no such model")
    with pytest.raises(EvalFailure) as info:
        ModelTarget(FakeClient(bad), "{prompt}").generate(TargetInput("k", "p"), runner.unit())
    assert info.value.systemic and guard.abort == ("failed", "input.bad_config")


def test_empty_model_text_follows_the_empty_policy():
    assert (
        ModelTarget(FakeClient(text("")), "{prompt}").generate(TargetInput("k", "p"), unit()).output
        == ""
    )
    with pytest.raises(EvalFailure) as info:
        ModelTarget(
            FakeClient(LLMResponse(text=None, stop_reason=STOP_END)), "{prompt}", on_empty="fail"
        ).generate(TargetInput("k", "p"), unit())
    assert info.value.kind == "empty_output"


def test_attempts_from_targets_are_valid_run_attempts():
    u = unit()
    CallableTarget(lambda i: "x").generate(TargetInput("k", "p"), u)
    assert all(isinstance(a, RunAttempt) for a in u.attempts)
