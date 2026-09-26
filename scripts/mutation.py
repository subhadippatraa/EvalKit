"""Targeted mutation testing of the invariants that matter.

    uv run python scripts/mutation.py [--only SUBSTRING]

Each mutant changes one line of behaviour (a comparison, a constant, a guard, a trigger) and the
listed tests must then FAIL ("killed"). A mutant whose tests still pass ("survived") is a gap in the
suite. This is deliberate and small, not an exhaustive mutation framework.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "evalkit"

# (name, file, old, new, tests)
M = [
    # -- schema: run lifecycle and result invariants (SQL) ------------------------------------
    (
        "sql: running->succeeded illegal path allowed",
        "migrations.py",
        # the effective trigger is the one migration 7 re-created (identified by what follows it)
        "OR (OLD.status = 'running' AND NEW.status IN ('succeeded','partial','cancelled','failed'))\n  OR (OLD.status IN ('partial','cancelled','failed') AND NEW.status = 'running')\n  OR (OLD.status = 'succeeded'",
        "OR (OLD.status IN ('created','running') AND NEW.status IN ('succeeded','partial','cancelled','failed'))\n  OR (OLD.status IN ('partial','cancelled','failed') AND NEW.status = 'running')\n  OR (OLD.status = 'succeeded'",
        ["tests/test_runs.py"],
    ),
    (
        "sql: succeeded may be reopened",
        "migrations.py",
        "OR (OLD.status IN ('partial','cancelled','failed') AND NEW.status = 'running')\n  OR (OLD.status = 'succeeded'",
        "OR (OLD.status IN ('partial','cancelled','failed','succeeded') AND NEW.status = 'running')\n  OR (OLD.status = 'succeeded'",
        ["tests/test_runs.py"],
    ),
    (
        "sql: success without every case result",
        "migrations.py",
        "      <> (SELECT case_count FROM dataset_versions WHERE id = OLD.dataset_version_id)\n    OR",
        "      = (SELECT case_count FROM dataset_versions WHERE id = OLD.dataset_version_id) + 999999\n    OR",
        ["tests/test_runs.py"],
    ),
    (
        "sql: case from another dataset version accepted",
        "migrations.py",
        "WHERE r.id = NEW.run_id AND c.id = NEW.case_id\n)",
        "WHERE r.id = NEW.run_id OR c.id = NEW.case_id\n)",
        ["tests/test_run_results.py"],
    ),
    (
        "sql: target failure may be scored",
        "migrations.py",
        "  OR (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') = 'failed'\n     AND NEW.status = 'skipped')",
        "  OR (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') = 'failed'\n     AND NEW.status IN ('skipped','ok'))",
        ["tests/test_run_results.py"],
    ),
    (
        "sql: evaluator result on a pending case result",
        "migrations.py",
        "  (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') = 'complete'\n     AND NEW.status IN ('ok','not_applicable','failed'))",
        "  (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') IN ('complete','pending')\n     AND NEW.status IN ('ok','not_applicable','failed'))",
        ["tests/test_run_results.py"],
    ),
    (
        "sql: metric range check removed",
        "migrations.py",
        "CHECK (value BETWEEN -1.7976931348623157e308\n                                                     AND 1.7976931348623157e308)",
        "",
        ["tests/test_run_schema.py"],
    ),
    (
        "sql: attempt owner XOR removed",
        "migrations.py",
        "CHECK ((case_result_id IS NULL) + (evaluator_result_id IS NULL) = 1),",
        "",
        ["tests/test_run_attempts.py"],
    ),
    (
        "sql: failed case result may lack its class",
        "migrations.py",
        "AND failure_class IS NOT NULL AND failure_class IN ('input','target','infrastructure')",
        "AND failure_class IN ('input','target','infrastructure')",
        ["tests/test_run_schema.py"],
    ),
    (
        "sql: target class allowed on evaluator results",
        "migrations.py",
        "AND failure_class IN ('input','evaluator','infrastructure')",
        "AND failure_class IN ('input','target','evaluator','infrastructure')",
        ["tests/test_run_schema.py", "tests/test_run_results.py"],
    ),
    (
        "sql: run config mutable",
        "migrations.py",
        "OR NEW.name IS NOT OLD.name OR NEW.config_json IS NOT OLD.config_json",
        "OR NEW.name IS NOT OLD.name",
        ["tests/test_runs.py"],
    ),
    (
        "sql: results writable on an ended run",
        "migrations.py",
        "  )\n  OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id OR NEW.case_id IS NOT OLD.case_id\n  OR NEW.created_at IS NOT OLD.created_at OR NEW.ord IS NOT OLD.ord\n  OR COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') <> 'running'\nBEGIN SELECT RAISE(ABORT, 'case results are write-once, and only on a running run')",
        "  )\n  OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id OR NEW.case_id IS NOT OLD.case_id\n  OR NEW.created_at IS NOT OLD.created_at OR NEW.ord IS NOT OLD.ord\nBEGIN SELECT RAISE(ABORT, 'case results are write-once, and only on a running run')",
        ["tests/test_run_schema.py", "tests/test_run_results.py"],
    ),
    (
        "sql: reviews mutable",
        "migrations.py",
        "CREATE TRIGGER reviews_no_update BEFORE UPDATE ON reviews\nBEGIN SELECT RAISE(ABORT, 'reviews are append-only'); END;",
        "",
        ["tests/test_calibration.py"],
    ),
    # -- domain models ------------------------------------------------------------------------
    (
        "model: case outcome may have output and failure",
        "runs.py",
        "if (self.output is None) == (self.failure is None):",
        "if (self.output is None) and (self.failure is None):",
        ["tests/test_run_results.py"],
    ),
    (
        "model: ok evaluator result may have no metrics",
        "runs.py",
        'if not self.metrics:\n                raise ValueError("an ok evaluator result needs at least one metric")',
        "if False:\n                raise ValueError('x')",
        ["tests/test_run_results.py"],
    ),
    (
        "model: non-finite metric accepted",
        "runs.py",
        'if value != value or value in (float("inf"), float("-inf")):',
        "if False:",
        ["tests/test_run_results.py"],
    ),
    (
        "model: evaluator failure may carry a target class",
        "runs.py",
        'EVALUATOR_RESULT_CLASSES, "evaluator-result")',
        'CASE_RESULT_CLASSES | EVALUATOR_RESULT_CLASSES, "evaluator-result")',
        ["tests/test_run_results.py"],
    ),
    (
        "identity: evaluators ignored",
        "runs.py",
        '"evaluators": self.evaluator_keys,',
        '"evaluators": [],',
        ["tests/test_run_config.py"],
    ),
    (
        "identity: policy leaks into identity",
        "runs.py",
        '"scoring_version": self.scoring_version,\n            },',
        '"scoring_version": self.scoring_version, "policy": self.policy,\n            },',
        ["tests/test_run_config.py"],
    ),
    (
        "identity: dataset content ignored",
        "runs.py",
        '"dataset": dataset_content_hash,',
        '"dataset": "",',
        ["tests/test_run_config.py"],
    ),
    (
        "store: duplicate result accepted",
        "run_store.py",
        'if existing is not None and existing["status"] != "pending":',
        "if False:",
        ["tests/test_run_results.py"],
    ),
    (
        "store: attempts not numbered",
        "runs.py",
        "if [a.n for a in attempts] != list(range(1, len(attempts) + 1)):",
        "if False:",
        ["tests/test_run_attempts.py"],
    ),
    (
        "attempt: failed attempt keeps no evidence",
        "runs.py",
        "        text, sha, truncated = capture_evidence(evidence, keep_text=True)",
        "        text, sha, truncated = capture_evidence(evidence, keep_text=False)",
        ["tests/test_run_attempts.py"],
    ),
    # -- taxonomy -----------------------------------------------------------------------------
    (
        "taxonomy: target timeout not retryable",
        "failures.py",
        '"timeout": KindInfo(True),\n        "contract_violation"',
        '"timeout": KindInfo(False),\n        "contract_violation"',
        ["tests/test_failures.py"],
    ),
    (
        "taxonomy: auth not systemic",
        "failures.py",
        '"auth": KindInfo(False, systemic=True)',
        '"auth": KindInfo(False)',
        ["tests/test_failures.py"],
    ),
    # -- call layer ---------------------------------------------------------------------------
    (
        "calls: one extra retry",
        "calls.py",
        "return min(cap, self.max_attempts - 1)",
        "return min(cap + 1, self.max_attempts)",
        ["tests/test_calls.py"],
    ),
    # 'calls: systemic failures retried' is an equivalent mutant: no systemic kind is in the retry
    # table, so the guard is defence in depth (its test asserts the behaviour, not the mutant).
    (
        "calls: backoff uncapped",
        "calls.py",
        "ceiling = min(self.cap_s, self.base_s * 2**retry_no)",
        "ceiling = self.base_s * 2**retry_no",
        ["tests/test_calls.py"],
    ),
    (
        "calls: retry-after ignored",
        "calls.py",
        "wait = max(wait, min(retry_after, self.cap_s))",
        "pass",
        ["tests/test_calls.py"],
    ),
    (
        "calls: no feedback on validation retry",
        "calls.py",
        "feedback = str(f)  # the request changes, so the retry is not a repeat",
        "feedback = None",
        ["tests/test_calls.py"],
    ),
    (
        "calls: cancel during backoff ignored",
        "calls.py",
        "if r.token.wait(wait):",
        "if r.token.wait(wait) and False:",
        ["tests/test_calls.py"],
    ),
    (
        "calls: unit deadline ignored",
        "calls.py",
        "if r.monotonic() + wait > self._deadline_at:",
        "if False:",
        ["tests/test_calls.py"],
    ),
    (
        "calls: breaker never trips",
        "calls.py",
        "if self._consecutive_infra >= self.breaker_threshold and self._abort is None:",
        "if False:",
        ["tests/test_calls.py", "tests/test_engine.py"],
    ),
    (
        "calls: success does not reset the breaker",
        "calls.py",
        "                self._consecutive_infra = 0\n                self._systemic.pop(key, None)",
        "                self._systemic.pop(key, None)",
        ["tests/test_calls.py"],
    ),
    (
        "calls: budget tokens ignored",
        "calls.py",
        "if b.max_tokens is not None and tokens >= b.max_tokens:",
        "if False:",
        ["tests/test_calls.py"],
    ),
    (
        "calls: attempts not numbered from first_n",
        "calls.py",
        "return self._first_n + len(self.attempts)",
        "return 1 + len(self.attempts)",
        ["tests/test_calls.py"],
    ),
    # -- targets ------------------------------------------------------------------------------
    (
        "targets: view leaks the reference",
        "targets.py",
        "return TargetInput(case.case_key, case.prompt, case.context, provided)",
        "return TargetInput(case.case_key, case.prompt, (case.context or '') + (case.reference or ''), provided)",
        ["tests/test_targets.py"],
    ),
    (
        "targets: template fields unchecked",
        "targets.py",
        "if n is not None and (n not in _TEMPLATE_FIELDS or spec or conv)",
        "if False",
        ["tests/test_targets.py"],
    ),
    (
        "targets: exception classed as evaluator",
        "targets.py",
        'FailureClass.TARGET, "exception", f"{type(e).__name__}: {e}"',
        'FailureClass.EVALUATOR, "internal_error", f"{type(e).__name__}: {e}"',
        ["tests/test_targets.py"],
    ),
    # -- evaluators ---------------------------------------------------------------------------
    (
        "eval: ndcg gain linear",
        "evaluators.py",
        "(2.0 ** relevance.get(d, 0) - 1.0) / math.log2(i + 1) for i, d in enumerate(top, 1)",
        "(relevance.get(d, 0)) / math.log2(i + 1) for i, d in enumerate(top, 1)",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: precision uses retrieved length",
        "evaluators.py",
        'metrics[f"precision@{k}"] = hits / k',
        'metrics[f"precision@{k}"] = hits / max(1, len(top))',
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: mrr uses 0-based rank",
        "evaluators.py",
        'metrics["mrr"] = 0.0 if first is None else 1.0 / first',
        'metrics["mrr"] = 0.0 if first is None else 1.0 / (first + 1)',
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: recall over retrieved not relevant",
        "evaluators.py",
        'metrics[f"recall@{k}"] = hits / len(relevant)',
        'metrics[f"recall@{k}"] = hits / max(1, len(top))',
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: idcg from retrieved only",
        "evaluators.py",
        "idcg = sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))",
        "idcg = sum((2.0 ** relevance.get(d, 0) - 1.0) / math.log2(i + 1) for i, d in enumerate(top, 1))",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: no relevant docs scored zero",
        "evaluators.py",
        'raise NotApplicable("no relevant documents are labelled for this case")',
        "return {f'{m}@{k}': 0.0 for m in ('recall', 'precision', 'hit', 'ndcg') for k in ks} | {'mrr': 0.0}",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: retrieved not deduplicated",
        "evaluators.py",
        "ranked = list(dict.fromkeys(retrieved))",
        "ranked = list(retrieved)",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: exact match ignores normalization",
        "evaluators.py",
        "a, b = _normalize(inp.output, steps), _normalize(inp.reference, steps)",
        "a, b = inp.output, inp.reference",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: citation validity over resolved only",
        "evaluators.py",
        'metrics = {"citation_validity": len(resolved) / len(tokens)}',
        'metrics = {"citation_validity": 1.0}',
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: malformed json is a failure",
        "evaluators.py",
        '{"valid": 0.0}, "FAIL", {"parse_error"',
        '{"valid": 1.0}, "FAIL", {"parse_error"',
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: must-pass gate ignored",
        "models.py",
        "passed = overall >= self.threshold and not self.failed_gates(scores)",
        "passed = overall >= self.threshold",
        ["tests/test_evaluators.py"],
    ),
    (
        "eval: remote $ref allowed",
        "evaluators.py",
        'and not v.startswith("#")',
        "and False",
        ["tests/test_evaluators.py"],
    ),
    # -- engine -------------------------------------------------------------------------------
    (
        "engine: failed target still evaluated",
        "engine.py",
        "            if outcome.failure is not None:\n                # a failed target and its skipped",
        "            if False:\n                # a failed target and its skipped",
        ["tests/test_engine.py"],
    ),
    (
        "engine: missing fields ignored",
        "engine.py",
        "        missing = sorted(f for f in ev.requires if getattr(einp, f) is None)",
        "        missing = []",
        ["tests/test_engine.py"],
    ),
    (
        "engine: budget never stops",
        "engine.py",
        "if (reason := guard.budget_stop()) is not None:",
        "if False:",
        ["tests/test_engine.py"],
    ),
    (
        "engine: cancel never stops",
        "engine.py",
        '            if token.cancelled:\n                return "cancelled"',
        '            if False:\n                return "cancelled"',
        ["tests/test_engine.py"],
    ),
    (
        "engine: systemic failure does not fail the run",
        "engine.py",
        'if guard.abort is not None and guard.abort[0] == "failed":',
        "if False:",
        ["tests/test_engine.py"],
    ),
    (
        "engine: success without checking evaluators",
        "engine.py",
        "            if sum(states.values()) != terminal:\n                return False",
        "            if False:\n                return False",
        ["tests/test_engine.py"],
    ),
    (
        "engine: spill replay order not fixed",
        "engine.py",
        'items.sort(key=lambda item: item.kind != "case")',
        "pass",
        ["tests/test_engine.py"],
    ),
    (
        "engine: recovery of half-finished units skipped",
        "engine.py",
        "owed = {u.result_id: u for u in self.store.pending_units(run_id, n_evaluators)}",
        "owed = {}",
        ["tests/test_engine.py"],
    ),
    (
        "engine: target mismatch accepted",
        "engine.py",
        "if spec_of(supplied) != spec:",
        "if False:",
        ["tests/test_engine.py"],
    ),
    (
        "engine: oversize output accepted",
        "engine.py",
        'if len(out.output.encode("utf-8", "replace")) > limit:',
        "if False:",
        ["tests/test_engine.py"],
    ),
    (
        "engine: unseeded order",
        "store.py",
        'digest = hashlib.sha256(f"{seed}\\x00{case_key}".encode()).digest()',
        'digest = hashlib.sha256(f"{case_key}".encode()).digest()',
        ["tests/test_engine.py"],
    ),
    (
        "engine: window not enforced",
        "engine.py",
        "while len(in_flight) >= policy.in_flight:",
        "while False:",
        ["tests/test_engine.py"],
    ),
    (
        "engine: preflight errors ignored",
        "engine.py",
        "        if not report.ok:\n            raise PreflightError(report)",
        "",
        ["tests/test_engine.py"],
    ),
    # -- analysis -----------------------------------------------------------------------------
    (
        "analysis: interval depends on storage order",
        "analysis.py",
        "    values = sorted(values)\n",
        "",
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: coverage ignores not_applicable",
        "analysis.py",
        'applicable = total - counts["not_applicable"]',
        "applicable = total",
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: headline given at low coverage",
        "analysis.py",
        "value=mean if reliable else None,",
        "value=mean,",
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: insufficient coverage always sufficient",
        "analysis.py",
        "sufficient = coverage is not None and coverage >= min_coverage",
        "sufficient = True",
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: strict pass rate over confident only",
        "analysis.py",
        "ev.pass_rate_strict = passes / total if total else None",
        "ev.pass_rate_strict = passes / (passes + fails) if passes + fails else None",
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: uncertain bounds swapped",
        "analysis.py",
        '"lower_bound": passes / (confident + unsure),',
        '"lower_bound": (passes + unsure) / (confident + unsure),',
        ["tests/test_analysis.py"],
    ),
    (
        "analysis: target failures not counted",
        "analysis.py",
        'stage[f"failed_{cls}"] += n',
        "pass",
        ["tests/test_analysis.py"],
    ),
    # -- comparison and gates -----------------------------------------------------------------
    (
        "compare: regression threshold sign",
        "compare.py",
        "if hi < -delta:",
        "if hi < delta:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: improvement threshold",
        "compare.py",
        "if lo > delta:",
        "if lo > -delta:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: equivalence needs the whole interval",
        "compare.py",
        "if delta > 0 and lo > -delta and hi < delta:",
        "if delta > 0 and lo > -delta or hi < delta:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: min_n ignored",
        "compare.py",
        "if mc.n_paired < min_n:",
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: direction ignored",
        "compare.py",
        'sign = 1.0 if gate.direction == "higher" else -1.0\n    delta =',
        "sign = 1.0\n    delta =",
        ["tests/test_compare.py"],
    ),
    (
        "compare: exclusions not counted",
        "compare.py",
        "for case_key in common - scored:",
        "for case_key in set():",
        ["tests/test_compare.py"],
    ),
    (
        "compare: confounders allowed silently",
        "compare.py",
        "if confounders and not allow_confounders:",
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: scoring version ignored",
        "compare.py",
        "if base.config.scoring_version != cand.config.scoring_version:",
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: survivorship threshold",
        "compare.py",
        "if cb is not None and cc is not None and abs(cb - cc) > COVERAGE_GAP:",
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "gates: coverage shortfall does not fail",
        "compare.py",
        "if not ev.sufficient_coverage:\n            shown",
        "if False:\n            shown",
        ["tests/test_compare.py"],
    ),
    (
        "gates: floor comparison off by one",
        "compare.py",
        "elif m.value < floor:",
        "elif m.value < floor - 1.0:",
        ["tests/test_compare.py"],
    ),
    (
        "gates: must-pass rate ignored",
        "compare.py",
        "if rate > max_rate:",
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "gates: inconclusive strict ignored",
        "compare.py",
        "(EXIT_INCONCLUSIVE if gates.strict else EXIT_PASS)",
        "EXIT_PASS",
        ["tests/test_compare.py", "tests/test_cli_platform.py"],
    ),
    (
        "gates: regression does not fail",
        "compare.py",
        'if d.decision == "regression":',
        "if False:",
        ["tests/test_compare.py"],
    ),
    (
        "compare: tag baseline may be the candidate",
        "analysis_store.py",
        'sql += " AND r.id <> ?"',
        'sql += " AND r.id <> ?" if False else ""',
        ["tests/test_compare.py"],
    ),
    # -- statistics ---------------------------------------------------------------------------
    (
        "stats: mcnemar one-sided",
        "stats.py",
        "return min(1.0, 2 * tail)",
        "return min(1.0, tail)",
        ["tests/test_stats.py"],
    ),
    (
        "stats: wilson uses wrong z",
        "stats.py",
        "half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom",
        "half = z * math.sqrt(p * (1 - p) / n) / denom",
        ["tests/test_stats.py"],
    ),
    (
        "stats: quantile without interpolation",
        "stats.py",
        "return s[lo] + (s[hi] - s[lo]) * (pos - lo)",
        "return s[lo]",
        ["tests/test_stats.py"],
    ),
    (
        "stats: mde factor",
        "stats.py",
        "MDE_FACTOR = 2.8",
        "MDE_FACTOR = 1.0",
        ["tests/test_stats.py"],
    ),
    (
        "stats: spearman without tie ranks",
        "stats.py",
        "ranks[order[k]] = (i + j) / 2 + 1  # average rank for ties",
        "ranks[order[k]] = k + 1",
        ["tests/test_stats.py"],
    ),
    (
        "stats: bootstrap unseeded",
        "stats.py",
        "n, rng = len(values), random.Random(seed)",
        "n, rng = len(values), random.Random()",
        ["tests/test_stats.py"],
    ),
    # -- calibration --------------------------------------------------------------------------
    (
        "calibration: targeted reviews counted",
        "calibration.py",
        'random_reviews = [r for r in by_case[case_key] if r["sample"] == "random"]',
        "random_reviews = list(by_case[case_key])",
        ["tests/test_calibration.py"],
    ),
    (
        "calibration: ties resolved silently",
        "calibration.py",
        "return None if rest and rest[0][1] == n else top",
        "return top",
        ["tests/test_calibration.py"],
    ),
    (
        "calibration: uncertain counted",
        "calibration.py",
        'if verdict == "UNCERTAIN":',
        "if False:",
        ["tests/test_calibration.py"],
    ),
    (
        "calibration: n_min ignored",
        "calibration.py",
        "uncalibrated=n < n_min,",
        "uncalibrated=False,",
        ["tests/test_calibration.py"],
    ),
    (
        "judge-check: failures counted correct",
        "judgecheck.py",
        'row["correct"] = row["verdict"] == case.expected',
        'row["correct"] = row["verdict"] == case.expected or row["failure"] is not None',
        ["tests/test_calibration.py"],
    ),
    # -- report -------------------------------------------------------------------------------
    (
        "report: no escaping",
        "report.py",
        'return html.escape("" if value is None else str(value), quote=True)',
        'return "" if value is None else str(value)',
        ["tests/test_report.py"],
    ),
    (
        "report: csp removed",
        "report.py",
        'f\'<meta http-equiv="Content-Security-Policy" content="{esc(CSP)}">\'',
        "''",
        ["tests/test_report.py"],
    ),
    (
        "report: text not truncated",
        "report.py",
        "    if len(text) <= limit:\n        return text\n",
        "    return text\n",
        ["tests/test_report.py"],
    ),
    # -- providers ----------------------------------------------------------------------------
    (
        "llm: throttling classified as auth",
        "bedrock.py",
        '"ThrottlingException": (FailureClass.INFRA, "rate_limited")',
        '"ThrottlingException": (FailureClass.INFRA, "auth")',
        ["tests/test_llm_clients.py"],
    ),
    (
        "llm: validation error not role-aware",
        "llm.py",
        'if role == "target":',
        "if False:",
        ["tests/test_llm_clients.py"],
    ),
    (
        "llm: truncation not a failure",
        "judge_eval.py",
        'STOP_MAX_TOKENS: (FailureClass.EVALUATOR, "truncated"',
        'STOP_MAX_TOKENS: (FailureClass.EVALUATOR, "invalid_output"',
        ["tests/test_llm_clients.py"],
    ),
    # -- P1.1 hardening ------------------------------------------------------------------------
    (
        "pairing: comparison pairs on the full content again (audit P0-1)",
        "analysis_store.py",
        "SELECT c.case_key, c.input_hash, r.status",
        "SELECT c.case_key, c.content_hash, r.status",
        ["tests/test_regression_e2e.py", "tests/test_compare.py"],
    ),
    (
        "pairing: cases with a changed input are paired anyway",
        "compare.py",
        'if b_out[k]["input_hash"] is not None and b_out[k]["input_hash"] == c_out[k]["input_hash"]',
        "if True",
        ["tests/test_regression_e2e.py", "tests/test_compare.py"],
    ),
    (
        "resume: finished evaluators are run (and billed) again",
        "engine.py",
        "pending = [ev for ev in self.evaluators if ev.key not in done or ev.key in unit.retry_evals]",
        "pending = list(self.evaluators)",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "lock: the OS lock is never taken",
        "runlock.py",
        "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "pass",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "lock: the in-process registry admits a second holder",
        "runlock.py",
        "if self._key in _held:",
        "if False:",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "writer: an unexpected exception kills the thread unrecorded",
        "engine.py",
        "except BaseException as e:  # noqa: BLE001 - nothing may kill this thread unrecorded",
        "except sqlite3.Error as e:  # noqa: BLE001",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "execute: an exception leaves the run running with the writer open",
        "engine.py",
        "except BaseException as e:  # noqa: BLE001 - Ctrl-C or a bug: still flush and stop cleanly",
        "except SystemExit as e:  # noqa: BLE001",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "execute: a failed target and its skipped rows are written separately",
        "engine.py",
        "self._put_result(outcome.failure, attempts, case_item, *skipped)",
        "self._put_result(outcome.failure, attempts, case_item); self._put_result(None, (), *skipped)",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "recovery: stranded failed case results are not repaired",
        "run_store.py",
        "\"AND r.status IN ('complete', 'failed') \"",
        "\"AND r.status = 'complete' \"",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "success: an evaluator that scored nothing may succeed",
        "engine.py",
        "if scored == 0 or scored / applicable < policy.min_coverage:",
        "if scored / max(applicable, 1) < policy.min_coverage:",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "sql: success without every evaluator result",
        "migrations.py",
        "         * OLD.evaluator_count",
        "         * 0",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "verify: a succeeded run without evaluator results is fine",
        "run_store.py",
        'if evals != done * row["evaluator_count"]:',
        "if False:",
        ["tests/test_engine_hardening.py"],
    ),
    (
        "systemic: one failure stops the run",
        "calls.py",
        "if n >= self.systemic_threshold and (",
        "if n >= 1 and (",
        ["tests/test_systemic_failures.py"],
    ),
    (
        "systemic: one provider's success hides another's failures",
        "calls.py",
        "key = (attempt.provider, attempt.model)",
        "key = (None, None)",
        ["tests/test_systemic_failures.py"],
    ),
    (
        "systemic: held failures are never proven case-specific",
        "engine.py",
        "(proven if guard.success_seq(*entry[0]) > entry[1] else keep).append(entry)",
        "keep.append(entry)",
        ["tests/test_systemic_failures.py"],
    ),
    (
        "paging: the pending page sorts by an expression again",
        "run_store.py",
        '"ORDER BY r.ord, r.id LIMIT ?",',
        '"ORDER BY COALESCE(r.ord, 0), r.id LIMIT ?",',
        ["tests/test_pending_paging.py"],
    ),
    (
        "providers: an over-long input is systemic again",
        "llm.py",
        "if looks_oversize(message, code):",
        "if False:",
        ["tests/test_provider_realistic.py"],
    ),
    (
        "providers: the request timeout is not passed to the OpenAI SDK",
        "bedrock_openai.py",
        "\"timeout\": req.timeout_s,  # the run's request timeout, not the client's default",
        "",
        ["tests/test_provider_realistic.py"],
    ),
    (
        "providers: boto3 ignores the request timeout",
        "bedrock.py",
        "if self._injected:",
        "if True:",
        ["tests/test_provider_realistic.py"],
    ),
    (
        "validity: not-applicable asymmetry is never a gate failure",
        "compare.py",
        "if abs(gap) > gates.max_na_asymmetry:",
        "if False:",
        ["tests/test_validity_hardening.py"],
    ),
    (
        "validity: not-applicable asymmetry is never warned about",
        "compare.py",
        "if abs(nb - nc) / len(common) > NA_GAP:",
        "if False:",
        ["tests/test_validity_hardening.py"],
    ),
    (
        "validity: truncation metadata is dropped",
        "engine.py",
        "meta=out.meta or None",
        "meta=None",
        ["tests/test_validity_hardening.py"],
    ),
    (
        "validity: gates accept unpinned callables",
        "compare.py",
        "if declared and not gates.allow_unpinned:",
        "if False:",
        ["tests/test_validity_hardening.py"],
    ),
    (
        "judge-check: the built-in golden set is accepted for any rubric",
        "cli_platform.py",
        'if not args.cases and frozen["rubric_content_hash"] != BUILTIN_RUBRIC.content_hash:',
        "if False:",
        ["tests/test_cli_platform.py"],
    ),
    # -- P2.1: schema (migration 7) -----------------------------------------------------------
    (
        "p2 sql: a complete case result may be reopened",
        "migrations.py",
        "OR (OLD.status = 'failed' AND NEW.status = 'pending'\n        AND NEW.retry_round = OLD.retry_round + 1",
        "OR (OLD.status IN ('failed','complete') AND NEW.status = 'pending'\n        AND NEW.retry_round = OLD.retry_round + 1",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: a case result reopens without its failure archived",
        "migrations.py",
        "        AND EXISTS (SELECT 1 FROM result_history h WHERE h.case_result_id = OLD.id\n                    AND h.scope = 'case' AND h.retry_round = OLD.retry_round))",
        "        )",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: the case retry round may skip",
        "migrations.py",
        "AND NEW.retry_round = OLD.retry_round + 1\n        AND EXISTS",
        "AND NEW.retry_round > OLD.retry_round\n        AND EXISTS",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: a successful evaluator result may be replaced",
        "migrations.py",
        "  OLD.status = 'failed' AND NEW.status IN ('ok','not_applicable','failed')\n  AND NEW.retry_round = OLD.retry_round + 1",
        "  OLD.status IN ('failed','ok') AND NEW.status IN ('ok','not_applicable','failed')\n  AND NEW.retry_round = OLD.retry_round + 1",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: an evaluator result is replaced without its failure archived",
        "migrations.py",
        "  AND EXISTS (SELECT 1 FROM result_history h WHERE h.evaluator_result_id = OLD.id\n              AND h.scope = 'evaluator' AND h.retry_round = OLD.retry_round)\n",
        "",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: the skipped rows of a failed case may be deleted",
        "migrations.py",
        "  AND COALESCE((SELECT status FROM case_results WHERE id = OLD.case_result_id), '') = 'pending'\n  AND COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') = 'running'\n)\nBEGIN SELECT RAISE(ABORT, 'evaluator results are permanent records'); END;\n\"\"\"",
        "  AND COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') = 'running'\n)\nBEGIN SELECT RAISE(ABORT, 'evaluator results are permanent records'); END;\n\"\"\"",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: a succeeded run is reopened by any ticket",
        "migrations.py",
        "WHERE x.run_id = OLD.id AND x.reopens_finished_at = OLD.finished_at))",
        "WHERE x.run_id = OLD.id))",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: result history is mutable",
        "migrations.py",
        "CREATE TRIGGER result_history_no_update BEFORE UPDATE ON result_history\nBEGIN",
        "CREATE TRIGGER result_history_no_update BEFORE UPDATE ON result_history WHEN 0\nBEGIN",
        ["tests/test_p2_schema.py"],
    ),
    (
        "p2 sql: an execution record can be rewritten",
        "migrations.py",
        "WHEN OLD.finished_at IS NOT NULL OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id",
        "WHEN NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id",
        ["tests/test_p2_schema.py"],
    ),
    # -- P2.1: retry-failed -------------------------------------------------------------------
    (
        "p2 retry: an invalid judge output is retried",
        "failures.py",
        'RETRY_NEVER = frozenset({(_C.EVALUATOR, "invalid_output")})',
        "RETRY_NEVER = frozenset()",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: systemic failures are retried",
        "failures.py",
        "if key in RETRY_NEVER or check_kind(failure_class, kind).systemic:",
        "if key in RETRY_NEVER:",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: a deadline or budget cut is never retried",
        "failures.py",
        'RETRY_ALSO = frozenset({(_C.INFRA, "deadline_exceeded"), (_C.INFRA, "budget_exceeded")})',
        "RETRY_ALSO = frozenset()",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: a unit may be retried without limit",
        "run_store.py",
        'f"({alias}.retry_round < ? AND "',
        'f"({alias}.retry_round >= 0 AND ? > 0 AND "',
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: failed evaluators are never retried",
        "engine.py",
        "if ev.key not in done or ev.key in unit.retry_evals",
        "if ev.key not in done",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: a retried case restarts its attempt numbering",
        "engine.py",
        "first_n=unit.prior_attempts + 1,",
        "first_n=1,",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: a reuse target retries inherited failures",
        "engine.py",
        'eligible = self.store.count_retryable(run_id, rounds, cases=tgt.kind != "reuse")',
        "eligible = self.store.count_retryable(run_id, rounds, cases=True)",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: a reuse target reopens inherited failures",
        "engine.py",
        'if tgt.kind != "reuse":  # a reuse target copies',
        "if True:  # a reuse target copies",
        ["tests/test_retry_failed.py"],
    ),
    (
        "p2 retry: the flag is ignored by the CLI",
        "cli_platform.py",
        "retry_failed=args.retry_failed,",
        "retry_failed=False,",
        ["tests/test_cli_ops.py"],
    ),
    # -- P2.1: cache --------------------------------------------------------------------------
    (
        "p2 cache: the key ignores the evaluator scope",
        "cache.py",
        '"scope": scope,',
        '"scope": "",',
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: the key ignores the model",
        "cache.py",
        '"model": model,',
        '"model": None,',
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: the key ignores the endpoint",
        "cache.py",
        '"endpoint": endpoint,',
        '"endpoint": None,',
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: the key ignores the request",
        "cache.py",
        '"request": fields,',
        '"request": {},',
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: non-deterministic calls are cached",
        "cache.py",
        "if not self.active or req.temperature > self.policy.max_temperature:",
        "if not self.active:",
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: targets are cached by default",
        "cache.py",
        'return req.role == "evaluator" or self.policy.targets',
        "return True",
        ["tests/test_cache.py"],
    ),
    (
        "p2 cache: replay writes the cache",
        "cache.py",
        'if self.policy.mode == "readwrite":',
        "if True:",
        ["tests/test_cache.py", "tests/test_calls_ops.py"],
    ),
    (
        "p2 cache: replay calls the provider on a miss",
        "calls.py",
        "if cache.replay:",
        "if False:",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 cache: a cache hit counts as a provider call",
        "calls.py",
        "if attempt.cache_hit:\n                u.cache_hits += 1\n                return",
        "if False:\n                u.cache_hits += 1\n                return",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 cache: concurrent identical requests are not serialized",
        "cache.py",
        "        entry[0].acquire()\n",
        "",
        ["tests/test_calls_ops.py", "tests/test_cache.py"],
    ),
    # -- P2.1: budgets, tokens and cost -------------------------------------------------------
    (
        "p2 budget: calls are made without reserving",
        "calls.py",
        "reservation = r.guard.reserve(est)",
        "reservation = None",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 budget: the worst case ignores the output cap",
        "calls.py",
        "tokens_in + req.max_tokens, pricing.upper_bound",
        "tokens_in, pricing.upper_bound",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 budget: a refused call does not stop the run",
        "calls.py",
        '                    self.blocked = ("budget", why)',
        "                    pass",
        ["tests/test_ops_runs.py"],
    ),
    (
        "p2 budget: a resume spends the budget again",
        "calls.py",
        "u.budget_tokens + p.budget_tokens,",
        "u.budget_tokens,",
        ["tests/test_calls_ops.py", "tests/test_ops_runs.py"],
    ),
    (
        "p2 budget: the USD limit is ignored",
        "calls.py",
        "if why is None and b.max_cost_usd is not None:",
        "if False:",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 budget: unreported usage is free",
        "calls.py",
        "u.assumed_tokens += reservation.tokens",
        "pass",
        ["tests/test_calls_ops.py"],
    ),
    (
        "p2 budget: an abandoned unit is recorded as a failure",
        "engine.py",
        "        except UnitAbandoned:\n            pass",
        "        except UnitAbandoned:\n            raise",
        ["tests/test_ops_runs.py"],
    ),
    (
        "p2 cost: an unpriced model costs zero",
        "pricing.py",
        'return Cost(None, None, "unpriced")',
        'return Cost(0.0, None, "unpriced")',
        ["tests/test_pricing.py"],
    ),
    (
        "p2 cost: missing usage costs zero",
        "pricing.py",
        'return Cost(None, None, "no_usage")',
        'return Cost(0.0, None, "no_usage")',
        ["tests/test_pricing.py"],
    ),
    (
        "p2 cost: a paid response that fails validation records no tokens",
        "calls.py",
        "usage = self._usage_of(resp, describe)",
        "usage = {}",
        ["tests/test_calls_ops.py"],
    ),
    # -- P2.1: environment, logging, reporting --------------------------------------------------
    (
        "p2 env: a different endpoint is not a confounder",
        "envsnapshot.py",
        "if be and ce and be != ce:",
        "if False:",
        ["tests/test_environment.py"],
    ),
    (
        "p2 env: a different validation-retry setting is not a confounder",
        "envsnapshot.py",
        "if bv is not None and cv is not None and bv != cv:",
        "if False:",
        ["tests/test_environment.py"],
    ),
    (
        "p2 log: any field may be logged",
        "events.py",
        "if name in FIELDS and",
        "if True and",
        ["tests/test_events.py"],
    ),
    (
        "p2 log: strings are not scrubbed",
        "events.py",
        "return scrub(value)[:_MAX_STR]",
        "return value[:_MAX_STR]",
        ["tests/test_events.py"],
    ),
    (
        "p2 report: data in the operations section is not escaped",
        "report.py",
        'return esc(", ".join(pairs) or "—")',
        'return ", ".join(pairs) or "—"',
        ["tests/test_operations.py"],
    ),
    (
        "p2 cli: budget flags are ignored",
        "cli_platform.py",
        '        out["budget"] = {**(base.get("budget") or {}), **budget}',
        "        pass",
        ["tests/test_cli_ops.py"],
    ),
]


def run_tests(tests: list[str]) -> tuple[bool, str]:
    """(tests pass, the first failing test id)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    failing = next((ln[7:] for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")), "")
    return proc.returncode == 0, failing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    killed, survived, missing = [], [], []
    for name, file, old, new, tests in M:
        if args.only not in name:
            continue
        path = SRC / file
        original = path.read_text()
        if original.count(old) < 1:
            missing.append(name)
            print(f"NOT FOUND  {name}")
            continue
        path.write_text(original.replace(old, new, 1))
        t = time.time()
        try:
            passed, failing = run_tests(tests)
        except subprocess.TimeoutExpired:
            passed, failing = False, "(timeout)"  # a hang is a detected difference too
        finally:
            path.write_text(original)
        (survived if passed else killed).append(name)
        print(
            f"{'SURVIVED' if passed else 'killed  '}  {name}  ({time.time() - t:.0f}s)"
            f"  {failing.split('::')[-1][:70]}",
            flush=True,
        )
    total = len(killed) + len(survived)
    print(f"\n{len(killed)}/{total} killed, {len(survived)} survived, {len(missing)} not found")
    for n in survived:
        print("  survived:", n)
    for n in missing:
        print("  missing :", n)
    return 1 if survived or missing else 0


if __name__ == "__main__":
    sys.exit(main())
