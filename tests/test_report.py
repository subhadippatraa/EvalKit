"""The static HTML report: complete, self-contained, and safe against hostile content."""

import os
import re
import stat
from datetime import UTC, datetime
from html.parser import HTMLParser

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
from evalkit.compare import Gates, compare, evaluate_gates
from evalkit.evaluators import llm_judge_spec, resolve
from evalkit.llm import LLMResponse
from evalkit.models import Rubric
from evalkit.report import CSP, MAX_TEXT, render_report, write_report
from evalkit.targets import CallableTarget, PrecomputedTarget

EM = EvaluatorSpec(kind="exact_match", name="em")
ALLOWED_TAGS = {
    "html", "head", "meta", "title", "style", "body", "h1", "h2", "h3", "p", "div", "span",
    "table", "thead", "tbody", "tr", "th", "td", "pre", "code", "details", "summary", "br", "b",
}  # fmt: skip

XSS = [
    "<script>alert('xss')</script>",
    "<img src=x onerror=alert(1)>",
    '"><svg/onload=alert(1)>',
    "</pre><script>steal()</script>",
    "<iframe src='https://evil.example/'></iframe>",
    "<a href='javascript:alert(1)'>x</a>",
    "<style>body{background:url(https://evil.example/x)}</style>",
    "<link rel=stylesheet href=https://evil.example/x.css>",
    "&lt;already-escaped&gt; & ampersand",
]


class Audit(HTMLParser):
    """Records every tag and attribute so the test can assert what the page can DO."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags, self.attrs, self.text = [], [], []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs += [(tag, k, v) for k, v in attrs]

    def handle_data(self, data):
        self.text.append(data)


def audit(page):
    a = Audit()
    a.feed(page)
    return a


def make(kit, n=6, *, prompt="q", output="o", name=None, tags=("t",), target=None):
    kit.datasets.import_cases(
        "qa",
        [
            case(f"c{i:02}", prompt=f"{prompt} {i}", output=output, reference="r", tags=list(tags))
            for i in range(n)
        ],
    )
    run = kit.controller.create(
        "qa", target=target or PrecomputedTarget(), evaluators=[EM], name=name,
        policy={"retry": {"base_s": 0.001, "cap_s": 0.002}},
    )  # fmt: skip
    kit.controller.execute(run.id, target=target)
    return run


# --- content ----------------------------------------------------------------------------------


def test_the_report_has_every_required_section_and_the_runs_facts(kit):
    run = make(kit, name="baseline-run")
    page = render_report(kit, run.id)
    for heading in ("Run overview", "Summary", "Failure breakdown", "Case-level evidence"):
        assert f"<h2>{heading}</h2>" in page
    for fact in (run.id, "baseline-run", "qa@1", run.identity_hash, run.exec_hash, "succeeded",
                 resolve(EM).key, "precomputed", "Frozen configuration", "Preflight"):  # fmt: skip
        assert fact in page
    assert "Cases</span>" in page and "Case coverage" in page and "Target failure rate" in page
    for cls in ("input", "target", "evaluator", "infrastructure"):
        assert f"<span>{cls}</span>" in page
    assert "match" in page and "wilson" in page and "100.0%" in page


def test_case_level_evidence_shows_input_output_evaluators_evidence_and_attempts(kit):
    def target(inp):
        if inp.case_key == "c01":
            raise RuntimeError("pipeline exploded")
        return "wrong answer"

    t = CallableTarget(target, name="t", fingerprint="1")
    run = make(kit, target=t)
    page = render_report(kit, run.id)
    assert "wrong answer" in page and "pipeline exploded" in page
    assert "target.exception" in page and "has failures" in page and "skipped" in page
    assert "<h3>Input</h3>" in page and "<h3>Reference</h3>" in page and "tags: t" in page
    assert page.index("c01") < page.index("c00")  # failures come first
    assert "callable" in page  # the attempt's provider


def test_failed_attempts_show_their_kept_evidence_and_the_failure_class(kit):
    kit.datasets.import_cases("qa", [case("a", output="o", reference="r")])
    run = kit.runs.create("qa", RunConfig(evaluators=[resolve(EM)]))
    kit.runs.start(run.id)
    cr = kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("o"))
    attempts = [
        RunAttempt.failed(
            1,
            datetime.now(UTC),
            900,
            EvalFailure("infrastructure", "rate_limited", "429 slow down", http_status=429),
            evidence={"partial": "provider body"},
            provider="bedrock",
            model="m",
        ),  # fmt: skip
        RunAttempt.succeeded(
            2,
            datetime.now(UTC),
            100,
            provider="bedrock",
            model="m",
            input_tokens=10,
            output_tokens=2,
        ),
    ]
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=resolve(EM).key,
            status="ok",
            verdict="PASS",
            metrics={"match": 1.0},
            detail={"why": "equal"},
        ),
        attempts=attempts,
    )
    page = render_report(kit, run.id)
    assert (
        "infrastructure.rate_limited" in page
        and "provider body" in page
        and "429 slow down" in page
    )
    assert "evidence of failed attempt 1" in page and "10/2" in page and "&quot;why&quot;" in page


def test_a_failure_is_reported_separately_by_class(kit):
    kit.datasets.import_cases("qa", [case(f"c{i}", output="o", reference="r") for i in range(4)])
    run = kit.runs.create("qa", RunConfig(evaluators=[resolve(EM)]))
    kit.runs.start(run.id)
    for i, (cls, kind) in enumerate(
        [("target", "timeout"), ("input", "oversize"), ("infrastructure", "cancelled")]
    ):
        kit.runs.record_case_result(
            run.id, f"c{i}", CaseOutcome.fail(Failure(failure_class=cls, kind=kind, message="m"))
        )
    page = render_report(kit, run.id)
    for expected in (
        "target</td><td>timeout",
        "input</td><td>oversize",
        "infrastructure</td><td>cancelled",
    ):
        assert expected in page
    assert "<b>1</b><span>target</span>" in page and "<b>1</b><span>input</span>" in page


def test_insufficient_coverage_withholds_the_headline_in_the_report(kit):
    def flaky(inp):
        if int(inp.case_key[1:]) < 4:
            raise RuntimeError("x")
        return "o"

    run = make(kit, n=10, target=CallableTarget(flaky, name="f", fingerprint="1"))
    page = render_report(kit, run.id)
    assert "withheld" in page and "below the required 95%" in page and "60.0%" in page


def test_judge_trust_panel_flags_an_uncalibrated_unchecked_judge(kit):
    class J:
        provider, model = "fake", "j"

        def call(self, req):
            return LLMResponse(payload=judged(quality=5), stop_reason="tool_use")

    j = J()
    kit.datasets.import_cases("qa", [case(f"c{i}", output="o") for i in range(3)])
    spec = llm_judge_spec("q", Rubric.from_dict({"quality": "Good?"}), j)
    run = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[spec])
    kit.controller.execute(run.id, clients=[j])
    page = render_report(kit, run.id)
    assert "Judge trust" in page and "UNCALIBRATED" in page and "not checked" in page
    assert "cannot be prevented by encoding" in page
    kit.store.save_judge_check(
        dict(evaluator_key=spec.key, fixture_name="mine", fixture_hash="a" * 64, n=10, correct=9,
             adversarial_n=4, adversarial_correct=3, failed=0, results=[])
    )  # fmt: skip
    assert "9/10 correct, adversarial 3/4" in render_report(kit, run.id)


def test_a_comparison_section_with_a_gate_confounders_and_exclusions(kit):
    kit.datasets.import_cases(
        "qa", [case(f"c{i:02}", output="o", reference="r" if i < 30 else "x") for i in range(40)]
    )
    a = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM])
    kit.controller.execute(a.id)
    b = kit.controller.create("qa", target=PrecomputedTarget(), evaluators=[EM])
    kit.controller.execute(b.id)
    cmp = compare(kit, a.id, b.id)
    gate = evaluate_gates(
        kit,
        b.id,
        Gates.from_mapping(
            {"gates": {"exact_match:*.match": {"direction": "higher", "delta": 0.02}}}
        ),
        cmp,
    )
    page = render_report(kit, b.id, comparison=cmp, gate=gate)
    needles = [
        "<h2>Comparison</h2>", "Paired metrics", "Gate:", "EQUIVALENT", "same identity",
        "Noise floor", "McNemar", a.id, "Comparing many metrics",
    ]  # fmt: skip
    for needle in needles:
        assert needle in page


def test_the_report_states_estimates_are_estimates_and_shows_usage(kit):
    run = make(kit)
    page = render_report(kit, run.id)
    assert "external calls" in page and "cost not estimated" in page


# --- self-contained and safe ------------------------------------------------------------------


def test_the_page_is_static_and_self_contained(kit):
    page = render_report(kit, make(kit).id)
    a = audit(page)
    assert set(a.tags) <= ALLOWED_TAGS, set(a.tags) - ALLOWED_TAGS
    assert not {"script", "link", "img", "iframe", "object", "embed", "form", "a", "base"} & set(
        a.tags
    )
    for tag, key, value in a.attrs:
        assert not key.startswith("on"), (tag, key)  # no event handlers
        assert not re.search(r"(?i)https?:|javascript:|data:", value or "") or key == "content", (
            tag,
            key,
            value,
        )
    assert f'content="{CSP}"'.replace("'", "&#x27;") in page
    assert "default-src 'none'" in CSP and "script-src" not in CSP and "connect-src" not in CSP
    assert "url(" not in page and "@import" not in page and "<script" not in page.lower()


@pytest.mark.parametrize("payload", XSS)
def test_hostile_content_in_every_field_is_escaped_and_inert(kit, payload):
    kit.datasets.import_cases(
        "qa",
        [
            {
                "case_key": "k",
                "prompt": payload,
                "reference": payload,
                "context": payload,
                "output": payload,
                "tags": [payload[:100]],
                "metadata": {"x": payload},
            }
        ],
    )
    run = kit.controller.create(
        "qa",
        target=PrecomputedTarget(),
        evaluators=[EM],
        name=payload[:100],
        idempotency_key=None,
    )
    kit.runs.plan(run.id)
    kit.controller.execute(run.id)
    page = render_report(kit, run.id)
    a = audit(page)
    assert set(a.tags) <= ALLOWED_TAGS, set(a.tags) - ALLOWED_TAGS
    for tag, key, _value in a.attrs:
        assert not key.startswith("on") and key not in ("href", "src"), (tag, key)
    # the payload text is there, as text: parsed back it is exactly what was stored
    assert payload in "".join(a.text)


def test_hostile_failure_messages_and_evaluator_detail_are_escaped(kit):
    kit.datasets.import_cases("qa", [case("a", output="o", reference="r")])
    run = kit.runs.create("qa", RunConfig(evaluators=[resolve(EM)]))
    kit.runs.start(run.id)
    cr = kit.runs.record_case_result(run.id, "a", CaseOutcome.complete("o"))
    kit.runs.record_evaluator_result(
        cr.id,
        EvaluatorOutcome(
            evaluator_key=resolve(EM).key,
            status="ok",
            verdict="PASS",
            metrics={"match": 1.0},
            detail={"reasoning": "<script>alert(1)</script>", "<b>key</b>": "v"},
        ),
        attempts=[
            RunAttempt.failed(
                1,
                datetime.now(UTC),
                1,
                Failure(
                    failure_class="infrastructure",
                    kind="timeout",
                    message="<img src=x onerror=alert(1)>",
                ),
                evidence="</pre><script>x</script>",
            )
        ],
    )
    page = render_report(kit, run.id)
    assert (
        "<script" not in page.lower()
        and "<img" not in page.lower()
        and "onerror=alert" not in page.replace("&lt;img src=x onerror=alert(1)&gt;", "")
    )
    assert set(audit(page).tags) <= ALLOWED_TAGS


def test_long_text_is_truncated_with_a_visible_note(kit):
    run = make(kit, n=1, output="x" * (MAX_TEXT + 5000))
    page = render_report(kit, run.id)
    assert "5000 more characters not shown" in page and page.count("x" * MAX_TEXT) >= 1
    assert "x" * (MAX_TEXT + 1) not in page


def test_max_cases_bounds_the_evidence_and_says_how_many_were_left_out(kit):
    run = make(kit, n=12)
    page = render_report(kit, run.id, max_cases=5)
    assert page.count("<details id='case-") == 5 and "7 more case(s) are in the database" in page
    none = render_report(kit, run.id, max_cases=0)
    assert "<details id='case-" not in none and "12 more case(s)" in none
    assert "<details id='case-" in render_report(kit, run.id, max_cases=100)
    for bad in (-1, True, "5"):
        with pytest.raises(RunError, match="max_cases"):
            render_report(kit, run.id, max_cases=bad)


def test_rendering_is_deterministic(kit):
    run = make(kit)
    assert render_report(kit, run.id) == render_report(kit, run.id)


def test_unknown_runs_raise(kit):
    with pytest.raises(RunError):
        render_report(kit, "nope")


# --- writing the file -------------------------------------------------------------------------


def test_write_report_is_atomic_private_and_refuses_to_overwrite(kit, tmp_path):
    run = make(kit)
    out = tmp_path / "r.html"
    assert write_report(kit, run.id, out) == out
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    if os.name == "posix":
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
    with pytest.raises(RunError, match="already exists"):
        write_report(kit, run.id, out)
    before = out.read_text()
    write_report(kit, run.id, out, overwrite=True)
    assert out.read_text() == before
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]


def test_a_failed_render_leaves_no_partial_file(kit, tmp_path):
    out = tmp_path / "r.html"
    with pytest.raises(RunError):
        write_report(kit, "nope", out)
    assert not out.exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".r.html")]


def test_the_report_opens_offline_it_references_nothing_outside_itself(kit, tmp_path):
    run = make(kit)
    out = write_report(kit, run.id, tmp_path / "r.html")
    text = out.read_text(encoding="utf-8")
    body_without_text = re.sub(r"<pre>.*?</pre>", "", text, flags=re.S)
    assert not re.search(r"(?i)\b(src|href|action|data)\s*=", body_without_text)
