"""A self-contained static HTML report (docs/TARGET-ARCHITECTURE.md §13.2).

One file, inline CSS, no JavaScript, no external resource of any kind: open it from disk, attach it
to a CI job. It is a *view* of stored, immutable data, so it needs no server.

Everything that comes from data (case text, model output, judge reasoning, error messages, names,
tags, config) is untrusted: it is HTML-escaped at every interpolation, placed in text nodes or
`<pre>`, never in an attribute that can carry behaviour, and the page ships with a CSP that allows
no script, no network and no embedding.
"""

from __future__ import annotations

import html
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evalkit.analysis import RunSummary, evalkit_version, summarize
from evalkit.calibration import calibrate
from evalkit.compare import Comparison, GateResult
from evalkit.errors import RunError

if TYPE_CHECKING:
    from evalkit.kit import EvalKit

CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
    "base-uri 'none'; form-action 'none'"
)
MAX_TEXT = 4000  # characters of any one text shown; the full text stays in the database
DEFAULT_MAX_CASES = 200

_CSS = """
:root { --bg:#fff; --fg:#1b1f24; --muted:#59636e; --line:#d0d7de; --card:#f6f8fa;
        --ok:#1a7f37; --bad:#cf222e; --warn:#9a6700; --info:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --card:#161b22;
          --ok:#3fb950; --bad:#f85149; --warn:#d29922; --info:#58a6ff; }
}
* { box-sizing: border-box; }
body { margin:0 auto; max-width:1180px; padding:16px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size:22px; margin:8px 0; }
h2 { font-size:17px; margin:28px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
h3 { font-size:14px; margin:16px 0 6px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding:8px 12px; min-width:120px; }
.card b { display:block; font-size:20px; } .card span { color:var(--muted); font-size:12px; }
table { border-collapse:collapse; width:100%; margin:6px 0 12px; font-size:13px; }
th, td { border:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }
th { background:var(--card); } td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
pre { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:6px 8px;
      margin:4px 0; white-space:pre-wrap; word-break:break-word;
      font:12px/1.4 ui-monospace, Menlo, monospace; }
details { border:1px solid var(--line); border-radius:6px; margin:6px 0; padding:4px 10px; }
summary { cursor:pointer; padding:2px 0; }
.badge { display:inline-block; border-radius:10px; padding:0 8px; font-size:12px;
         border:1px solid currentColor; }
.ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
.info { color:var(--info); }
.banner { border:1px solid var(--warn); border-left-width:4px; border-radius:4px;
          padding:6px 10px; margin:8px 0; }
.banner.bad { border-color:var(--bad); }
.small { font-size:12px; color:var(--muted); }
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def num(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}g}"


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def clip(text: str | None, limit: int = MAX_TEXT) -> str:
    if text is None:
        return "—"
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters not shown]"


def pre(text: str | None, limit: int = MAX_TEXT) -> str:
    return f"<pre>{esc(clip(text, limit))}</pre>"


def badge(text: Any, kind: str = "info") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def small(text: Any) -> str:
    return f"<span class=small>{esc(text)}</span>"


def banner(text: Any, kind: str = "") -> str:
    return f'<div class="banner {kind}">{text}</div>'


def table(
    headers: Sequence[str], rows: Iterable[Sequence[Any]], numeric: Iterable[int] = ()
) -> str:
    """`headers` and every cell are HTML (callers escape); `numeric` columns are right-aligned."""
    right = set(numeric)

    def cell(tag: str, i: int, value: Any) -> str:
        return f"<{tag}{' class=n' if i in right else ''}>{value}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _json(value: Any, limit: int = MAX_TEXT) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    return pre(text, limit)


def _status_kind(status: str) -> str:
    return {"succeeded": "ok", "failed": "bad", "cancelled": "warn", "partial": "warn"}.get(
        status, "info"
    )


# ---- sections -------------------------------------------------------------------------------


def _overview(kit: EvalKit, run_id: str) -> str:
    run = kit.runs.get(run_id)
    tags = kit.store.run_tags(run_id)
    target = json.dumps(run.config.target.identity, sort_keys=True)[:300]
    env = ", ".join(f"{k}={v}" for k, v in run.environment.items() if k != "preflight")
    stopped = f" {esc(run.stop_reason)}" if run.stop_reason else ""
    started = run.started_at.isoformat() if run.started_at else "—"
    finished = run.finished_at.isoformat() if run.finished_at else "—"
    rows = [
        ["Run", esc(run.id)],
        ["Name", esc(run.name or "—")],
        ["Status", badge(run.status, _status_kind(run.status)) + stopped],
        ["Dataset version", f"{esc(run.dataset_ref)} {small(run.dataset_version_id)}"],
        ["Identity hash", f"<code>{esc(run.identity_hash)}</code> {small('what was measured')}"],
        ["Exec hash", f"<code>{esc(run.exec_hash)}</code> {small('speed and cost only')}"],
        ["Created", esc(run.created_at.isoformat())],
        ["Started / finished", f"{esc(started)} / {esc(finished)}"],
        ["Tags", esc(", ".join(tags) or "—")],
        ["Target", f"{esc(run.config.target.kind)} {small(target)}"],
        [
            "Evaluators",
            "<br>".join(f"<code>{esc(e.key)}</code>" for e in run.config.evaluators) or "—",
        ],
        ["Scoring version", esc(run.config.scoring_version)],
        ["Environment", esc(env)],
    ]
    config = _json(run.config.model_dump(mode="json"))
    preflight = ""
    report = run.environment.get("preflight")
    if report:
        issues = table(
            ["level", "code", "message", "count"],
            [
                [
                    badge(i["level"], "bad" if i["level"] == "error" else "warn"),
                    esc(i["code"]),
                    esc(i["message"]),
                    esc(i["count"]),
                ]
                for i in report["issues"]
            ],
            {3},
        )
        preflight = (
            "<details><summary>Preflight (what was known before anything was spent)</summary>"
            f"{issues}{_json(report['estimate'])}</details>"
        )
    return (
        f"<h2>Run overview</h2>{table(['', ''], rows)}"
        f"<details><summary>Frozen configuration</summary>{config}</details>{preflight}"
        f"<p>{small('evalkit ' + evalkit_version() + '; every figure is recomputed from rows')}</p>"
    )


def _metric_rows(s: RunSummary) -> list[list[str]]:
    rows = []
    for ev in s.evaluators.values():
        coverage = badge(pct(ev.coverage), "ok" if ev.sufficient_coverage else "bad")
        if not ev.metrics:
            rows.append([esc(ev.key), "—", "—", "—", "—", "—", coverage])
        for m in ev.metrics.values():
            value = num(m.value) if m.value is not None else badge("withheld", "bad")
            note = f"<br>{small(m.note)}" if m.note else ""
            interval = f"[{num(m.ci_low)}, {num(m.ci_high)}] {esc(m.ci_method)}"
            rows.append(
                [esc(ev.key), esc(m.name), value + note, num(m.observed_mean), interval,
                 esc(m.n), coverage]
            )  # fmt: skip
    return rows


def _pass_rates(s: RunSummary) -> str:
    rows = []
    for ev in s.evaluators.values():
        pr = ev.pass_rate
        if not pr:
            continue
        value = num(pr["value"]) if pr["value"] is not None else badge("withheld", "bad")
        span = f"[{num(pr['lower_bound'])}, {num(pr['upper_bound'])}]"
        bounds = f"{span} ({esc(pr['uncertain'])} uncertain)"
        rows.append(
            [esc(ev.key), value, f"[{num(pr['ci_low'])}, {num(pr['ci_high'])}]", esc(pr["n"]),
             bounds, num(ev.pass_rate_strict)]
        )  # fmt: skip
    if not rows:
        return ""
    headers = [
        "evaluator", "pass rate (confident verdicts)", "95% CI", "n",
        "bounds if UNCERTAIN counted", "strict (all cases)",
    ]  # fmt: skip
    return "<h3>Pass rate</h3>" + table(headers, rows, {3})


def _usage(s: RunSummary) -> str:
    u = s.usage
    if u.get("cost_usd_estimate") is not None:
        cost = (
            f"estimated cost ${u['cost_usd_estimate']:.4f} from user-supplied prices "
            f"({esc(u['unpriced_attempts'])} unpriced attempts): an ESTIMATE"
        )
    else:
        cost = "cost not estimated (no price table)"
    text = (
        f"{esc(u['attempts'])} external calls ({esc(u['failed_attempts'])} failed, "
        f"{esc(u['retries'])} retries); tokens in {esc(u['input_tokens'])} / "
        f"out {esc(u['output_tokens'])}; {cost}."
    )
    latency = ""
    if s.target_latency_ms:
        t = s.target_latency_ms
        latency = (
            f"<p>Target latency (ms): p50 {num(t['p50'])}, p95 {num(t['p95'])}, "
            f"mean {num(t['mean'])} over {esc(int(t['n']))} completed cases.</p>"
        )
    return f"{latency}<p>{text}</p>"


def _summary(s: RunSummary) -> str:
    st = s.case_stage
    failed = st["failed_target"] + st["failed_input"] + st["failed_infrastructure"]
    cards = [
        ("Cases", s.cases),
        ("Completed", st["complete"]),
        ("Failed at target stage", failed),
        ("Pending / missing", st["pending"] + st["missing"]),
        ("Case coverage", pct(s.case_coverage)),
        ("Target failure rate", pct(s.target_failure_rate)),
    ]
    card_html = "".join(
        f"<div class=card><b>{esc(v)}</b><span>{esc(k)}</span></div>" for k, v in cards
    )
    banners = "".join(banner(esc(w)) for w in s.warnings)
    headers = ["evaluator", "metric", "value", "observed mean", "95% CI", "n", "coverage"]
    metrics = table(headers, _metric_rows(s), {2, 3, 5})
    slices = ""
    if s.slices:
        rows = [
            [esc(tag), esc(name), esc(v["n"]), num(v["mean"])]
            for tag, m in sorted(s.slices.items())
            for name, v in sorted(m.items())
        ]
        slices = "<h3>Slices by tag</h3>" + table(["tag", "metric", "n", "mean"], rows, {2, 3})
    return (
        f"<h2>Summary</h2><div class=cards>{card_html}</div>{banners}{metrics}"
        f"{_pass_rates(s)}{_usage(s)}{slices}"
    )


_ACCOUNT_KEYS = (
    "ok", "not_applicable", "skipped", "failed_evaluator", "failed_infrastructure",
    "failed_input", "missing",
)  # fmt: skip


def _failures(s: RunSummary) -> str:
    classes = ["input", "target", "evaluator", "infrastructure"]
    counts = dict.fromkeys(classes, 0)
    for f in s.failures:
        counts[f["class"]] += f["count"]
    cards = "".join(f"<div class=card><b>{counts[c]}</b><span>{c}</span></div>" for c in classes)
    rows = [[esc(f["scope"]), esc(f["class"]), esc(f["kind"]), esc(f["count"])] for f in s.failures]
    detail = (
        table(["stage", "class", "kind", "count"], rows, {3}) if rows else "<p>No failures.</p>"
    )
    headers = [
        "evaluator", "scored", "not applicable", "skipped (target failed)", "evaluator failed",
        "infrastructure failed", "input failed", "missing",
    ]  # fmt: skip
    accounting = table(
        headers,
        [[esc(e.key), *[esc(e.counts[k]) for k in _ACCOUNT_KEYS]] for e in s.evaluators.values()],
        range(1, 8),
    )
    return (
        "<h2>Failure breakdown</h2>"
        "<p class=small>A target failure is the AI system's; an evaluator failure is the "
        "rubric's or judge's; infrastructure is the platform or provider; input is the case or "
        "configuration. None is ever turned into a score.</p>"
        f"<div class=cards>{cards}</div>{detail}<h3>Accounting per evaluator</h3>{accounting}"
    )


def _trust(kit: EvalKit, run_id: str, s: RunSummary) -> str:
    rows = []
    for ev in s.evaluators.values():
        if ev.kind != "llm_judge":
            continue
        cal = calibrate(kit, run_id, ev.key)
        if cal.uncalibrated:
            cal_text = (
                f"{badge('UNCALIBRATED', 'bad')} {cal.n_paired} human-reviewed random-sample "
                f"cases (need {cal.n_min})"
            )
        else:
            cal_text = (
                f"accuracy {pct(cal.accuracy)} (n={cal.n_paired}), kappa {num(cal.kappa)}, "
                f"FAIL precision {num(cal.fail_precision)}, recall {num(cal.fail_recall)}"
            )
        jc = kit.store.latest_judge_check(ev.key)
        if jc:
            jc_text = (
                f"{jc['correct']}/{jc['n']} correct, adversarial "
                f"{jc['adversarial_correct']}/{jc['adversarial_n']}, {jc['failed']} unjudged "
                f"(fixture {esc(jc['fixture_name'])}, {esc(jc['created_at'][:10])})"
            )
        else:
            jc_text = f"{badge('not checked', 'warn')} no judge-check stored for this evaluator"
        lb = ev.length_bias
        rho = num(lb["spearman"]) if lb and lb["spearman"] is not None else "—"
        rows.append([esc(ev.key), cal_text, jc_text, rho])
    if not rows:
        return ""
    headers = [
        "judge evaluator", "calibration vs humans", "judge-check (golden set)",
        "length correlation (Spearman)",
    ]  # fmt: skip
    return (
        "<h2>Judge trust</h2>"
        + table(headers, rows)
        + "<p class=small>A judge is a measurement instrument with unknown error, not ground "
        'truth. Semantic prompt injection ("score this 5") cannot be prevented by encoding; it '
        "is made detectable by the adversarial golden set, human calibration and deterministic "
        "evaluators. The golden set is small and hand-labelled: it catches gross failures, not "
        "subtle bias.</p>"
    )


_DECISION_KIND = {
    "regression": "bad", "improvement": "ok", "equivalent": "ok", "inconclusive": "warn",
}  # fmt: skip


def _gate(g: GateResult) -> str:
    kind = {"pass": "ok", "fail": "bad", "inconclusive": "warn"}[g.status]
    out = [f"<h3>Gate: {badge(g.status.upper(), kind)} {small(f'exit code {g.exit_code}')}</h3>"]
    out += [banner(f"<b>{esc(f['code'])}:</b> {esc(f['message'])}", "bad") for f in g.failures]
    out += [banner(esc(n)) for n in g.notes]
    rows = [
        [
            esc(d.ref),
            badge(d.decision.upper(), _DECISION_KIND[d.decision]),
            esc(d.n_paired),
            num(d.diff),
            f"[{num(d.ci_low)}, {num(d.ci_high)}]",
            num(d.mde),
            esc(d.reason),
        ]
        for d in g.decisions
    ]
    headers = ["gate", "decision", "n paired", "difference", "95% CI", "MDE", "why"]
    out.append(table(headers, rows, {2, 3, 5}))
    return "".join(out)


def _paired_table(c: Comparison) -> str:
    rows = []
    for m in c.metrics:
        rows.append(
            [
                esc(m.evaluator),
                esc(m.metric),
                esc(m.n_paired),
                num(m.mean_baseline),
                num(m.mean_candidate),
                num(m.diff),
                f"[{num(m.ci_low)}, {num(m.ci_high)}] {esc(m.ci_method)}",
                num(m.mde),
                f"{m.candidate_higher} / {m.candidate_lower} / {m.tied}",
                num(m.p_value),
            ]
        )
    headers = [
        "evaluator", "metric", "n paired", "baseline", "candidate", "difference", "95% CI",
        "min detectable", "cand. higher / lower / tied", "McNemar p",
    ]  # fmt: skip
    return "<h3>Paired metrics</h3>" + table(headers, rows, {2, 3, 4, 5, 7})


def _comparison(c: Comparison, g: GateResult | None) -> str:
    out = [
        f"<h2>Comparison</h2><p>Baseline <code>{esc(c.baseline_id)}</code> "
        f"({esc(c.baseline_ref)}) → candidate <code>{esc(c.candidate_id)}</code> "
        f"({esc(c.candidate_ref)}); {esc(c.common_cases)} cases in common, paired by case.</p>"
    ]
    out += [
        banner(f"<b>Confounder ({esc(x.kind)}):</b> {esc(x.detail)}", "bad") for x in c.confounders
    ]
    out += [banner(esc(w)) for w in c.warnings]
    out += [f"<p>{small(i)}</p>" for i in c.informational]
    if g is not None:
        out.append(_gate(g))
    out.append(_paired_table(c))
    excluded = [
        [esc(side), esc(reason), esc(n)]
        for side, reasons in c.exclusions.items()
        for reason, n in sorted(reasons.items())
    ]
    if excluded:
        out.append(
            "<h3>Cases excluded from pairing</h3>"
            + table(["side", "reason", "cases"], excluded, {2})
        )
    if c.noise_floor:
        rows = [[esc(k), num(v)] for k, v in c.noise_floor.items()]
        out.append(
            "<h3>Noise floor (same identity: a rerun)</h3>"
            + table(["metric", "suggested tolerance ≈"], rows, {1})
        )
    worst = [(m, w) for m in c.metrics if m.evaluator != "run" for w in m.worst_cases[:5]]
    if worst:
        rows = [
            [esc(f"{m.evaluator}.{m.metric}"), esc(w["case_key"]), num(w["baseline"]),
             num(w["candidate"]), num(w["diff"])]
            for m, w in worst
        ]  # fmt: skip
        headers = ["metric", "case", "baseline", "candidate", "difference"]
        out.append("<h3>Most regressed cases</h3>" + table(headers, rows, {2, 3, 4}))
    out.append(
        "<p class=small>Comparing many metrics at once inflates the chance of a spurious "
        "difference: only declared gates can fail a build; the rest is informational.</p>"
    )
    return "".join(out)


def _attempts(attempts: Sequence[Any]) -> str:
    if not attempts:
        return "<p class=small>No external calls (no attempts).</p>"
    rows = []
    for a in attempts:
        failed = a.outcome == "failed" and a.error_class is not None
        error = f"{a.error_class.value}.{a.error_kind}: {a.error}" if failed else ""
        rows.append(
            [
                esc(a.n),
                badge(a.outcome, "ok" if a.outcome == "ok" else "bad"),
                esc(f"{a.provider or ''} {a.model or ''}"),
                esc(a.duration_ms),
                esc(f"{a.input_tokens or 0}/{a.output_tokens or 0}"),
                esc(a.request_id or ""),
                esc(clip(error, 500)),
            ]
        )
    headers = ["#", "outcome", "provider / model", "ms", "tokens in/out", "request id", "error"]
    out = table(headers, rows, {0, 3})
    for a in attempts:
        if a.raw:
            cut = " (truncated)" if a.raw_truncated else ""
            out += (
                f"<details><summary>evidence of failed attempt {esc(a.n)}{cut}</summary>"
                f"{pre(a.raw, 2000)}</details>"
            )
    return out


def _failure_banner(f: Any) -> str:
    where = f"{esc(f.failure_class.value)}.{esc(f.kind)}"
    return banner(f"<b>{where}</b> (retryable: {esc(f.retryable)}): {esc(f.message)}", "bad")


def _evaluator_block(e: Any) -> str:
    kind = {"ok": "ok", "failed": "bad"}.get(e.status, "warn")
    head = f"<h3>Evaluator <code>{esc(e.evaluator_key)}</code> {badge(e.status, kind)}"
    if e.verdict:
        head += " " + badge(e.verdict, {"PASS": "ok", "FAIL": "bad"}.get(e.verdict, "warn"))
    parts = [head + "</h3>"]
    if e.metrics:
        parts.append(
            table(["metric", "value"], [[esc(k), num(v)] for k, v in e.metrics.items()], {1})
        )
    if e.failure:
        parts.append(_failure_banner(e.failure))
    if e.detail:
        parts.append(f"<details><summary>evidence</summary>{_json(e.detail)}</details>")
    parts.append(_attempts(e.attempts))
    return "".join(parts)


def _case_block(kit: EvalKit, run_id: str, version_id: str, index: int, key: str) -> str:
    case = kit.store.get_case(version_id, key)
    cr = kit.runs.case_result(run_id, key)
    if case is None or cr is None:
        return ""
    results = kit.runs.evaluator_results(cr.id)
    kind = {"failed": "bad", "complete": "ok"}.get(cr.status, "warn")
    flagged = cr.status == "failed" or any(e.status == "failed" for e in results)
    marks = badge(cr.status, kind) + (" " + badge("has failures", "bad") if flagged else "")
    parts = [
        f"<summary><code>{esc(key)}</code> {marks}</summary>",
        "<h3>Input</h3>" + pre(case.prompt),
    ]
    if case.context:
        parts.append("<h3>Context</h3>" + pre(case.context))
    if case.reference:
        parts.append("<h3>Reference</h3>" + pre(case.reference))
    if case.tags:
        parts.append(f"<p>{small('tags: ' + ', '.join(case.tags))}</p>")
    parts.append("<h3>Target</h3>")
    if cr.failure:
        parts.append(_failure_banner(cr.failure))
    else:
        parts.append(pre(cr.output))
        if cr.retrieved:
            parts.append(f"<p>{small('retrieved: ' + ', '.join(cr.retrieved[:50]))}</p>")
    if cr.duration_ms is not None:
        parts.append(f"<p>{small(f'target time {cr.duration_ms} ms')}</p>")
    parts.append(_attempts(cr.attempts))
    parts += [_evaluator_block(e) for e in results]
    return f"<details id='case-{index}'>{''.join(parts)}</details>"


def _case_evidence(kit: EvalKit, run_id: str, keys: list[str]) -> str:
    version_id = kit.runs.get(run_id).dataset_version_id
    return "".join(_case_block(kit, run_id, version_id, i, k) for i, k in enumerate(keys))


def _select_cases(
    kit: EvalKit, run_id: str, comparison: Comparison | None, max_cases: int
) -> tuple[list[str], int]:
    """Failures first, then the most regressed, then the rest by key; and how many were left out."""
    chosen: dict[str, None] = {}
    for f in kit.runs.failures(run_id, limit=min(max_cases, 10_000)):
        chosen.setdefault(f.case_key)
    if comparison is not None:
        for m in comparison.metrics:
            for w in m.worst_cases:
                chosen.setdefault(w["case_key"])
    total = kit.runs.counts(run_id).total_cases
    for cr in kit.runs.case_results(run_id, batch_size=min(1000, max_cases + 1)):
        if len(chosen) >= max_cases:
            break
        chosen.setdefault(cr.case_key)
    keys = list(chosen)[:max_cases]
    return keys, max(0, total - len(keys))


def render_report(
    kit: EvalKit,
    run_id: str,
    *,
    comparison: Comparison | None = None,
    gate: GateResult | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
    min_coverage: float | None = None,
    seed: int = 0,
) -> str:
    """The report as one HTML string."""
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 0:
        raise RunError("max_cases must be a non-negative integer")
    options: dict[str, Any] = {"seed": seed}
    if min_coverage is not None:
        options["min_coverage"] = min_coverage
    summary = summarize(kit, run_id, **options)
    if max_cases:
        keys, left_out = _select_cases(kit, run_id, comparison, max_cases)
    else:
        keys, left_out = [], kit.runs.counts(run_id).total_cases
    omitted = ""
    if left_out:
        omitted = f"<p>{
            small(
                f'{left_out} more case(s) are in the database and not shown here '
                '(failures and the most regressed cases are shown first)'
            )
        }</p>"
    sections = [
        _overview(kit, run_id),
        _summary(summary),
        _failures(summary),
        _trust(kit, run_id, summary),
        _comparison(comparison, gate) if comparison is not None else "",
        f"<h2>Case-level evidence</h2>{_case_evidence(kit, run_id, keys)}{omitted}",
    ]
    title = f"EvalKit report {run_id}"
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(CSP)}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{esc(title)}</h1>{''.join(sections)}</body></html>\n"
    )


def write_report(
    kit: EvalKit,
    run_id: str,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> Path:
    """Write the report atomically (owner-only). An existing file is never overwritten silently."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise RunError(f"{target} already exists (overwrite=True / --force replaces it)")
    content = render_report(kit, run_id, **kwargs)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target
