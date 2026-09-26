"""CLI handlers for the evaluation platform: runs, compare, report, judge-check.

Each handler returns `(exit_code, json_able_output_or_None)`; `cli.main` prints it. Exit codes:
0 ok, 1 error (or a run that did not complete), 2 usage, 3 gate failed, 4 gate inconclusive
under `--strict`.

Trusted operator code (a `--target pkg.mod:fn` callable) is imported only from a CLI flag or the
operator's spec file, never from a dataset or the database.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evalkit import safejson
from evalkit.calibration import CaseReview  # noqa: F401  (validates the review shape)
from evalkit.calls import CancelToken
from evalkit.compare import (
    EXIT_INCONCLUSIVE,
    EXIT_PASS,
    EXIT_REGRESSION,
    Gates,
    compare,
    evaluate_gates,
    resolve_baseline,
)
from evalkit.env import client_from_env
from evalkit.errors import ConfigError, EvalKitError
from evalkit.evaluators import resolve
from evalkit.judgecheck import BUILTIN_RUBRIC, builtin_cases, judge_check, load_cases
from evalkit.kit import EvalKit
from evalkit.limits import MAX_INPUT_FILE_BYTES
from evalkit.operations import format_status, run_status
from evalkit.pricing import PriceTable
from evalkit.report import DEFAULT_MAX_CASES, write_report
from evalkit.runs import EvaluatorSpec, Run
from evalkit.targets import CallableTarget, ModelTarget, PrecomputedTarget, ReuseTarget, Target


class UsageError(EvalKitError):
    """A command-line or spec-file problem (exit code 2), as opposed to a run-time failure."""


_SPEC_KEYS = {"target", "evaluators", "policy"}
_TARGET_KEYS = {
    "precomputed": {"kind"},
    "reuse": {"kind", "source_run_id"},
    "callable": {"kind", "callable", "fingerprint", "name", "on_empty"},
    "model": {"kind", "template", "system", "temperature", "max_tokens", "on_empty"},
}


def _read_spec(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        with open(p, "rb") as f:
            data = f.read(MAX_INPUT_FILE_BYTES + 1)
        if len(data) > MAX_INPUT_FILE_BYTES:
            raise UsageError(f"{path} exceeds {MAX_INPUT_FILE_BYTES} bytes")
        text = data.decode("utf-8")
        parsed = tomllib.loads(text) if p.suffix == ".toml" else safejson.loads(text)
    except OSError as e:
        raise UsageError(f"cannot read {path}: {e}") from e
    except (ValueError, RecursionError) as e:
        raise UsageError(f"cannot parse {path}: {e}") from e
    if not isinstance(parsed, dict):
        raise UsageError(f"{path} must contain an object")
    unknown = set(parsed) - _SPEC_KEYS
    if unknown:
        raise UsageError(
            f"{path}: unknown key(s) {sorted(unknown)} (allowed: {sorted(_SPEC_KEYS)})"
        )
    return parsed


def load_callable(ref: str) -> Any:
    """`pkg.mod:fn` -> the function. Trusted operator code, imported from a flag only."""
    module, sep, name = ref.partition(":")
    if not module or not sep or not name:
        raise UsageError(f"a callable target is written 'package.module:function', got {ref!r}")
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"cannot load {ref!r}: {e}") from e
    if not callable(obj):
        raise ConfigError(f"{ref!r} is not callable")
    return obj


def build_target(
    spec: dict[str, Any], *, callable_ref: str | None = None, fingerprint: str | None = None
) -> Target:
    """A `Target` from a spec-file `target` table (and/or CLI flags)."""
    kind = spec.get("kind", "callable" if callable_ref else "precomputed")
    if kind not in _TARGET_KEYS:
        raise ConfigError(f"unknown target kind {kind!r} (known: {sorted(_TARGET_KEYS)})")
    unknown = set(spec) - _TARGET_KEYS[kind]
    if unknown:
        raise ConfigError(f"target kind {kind!r}: unknown key(s) {sorted(unknown)}")
    if kind == "precomputed":
        return PrecomputedTarget()
    if kind == "reuse":
        if not spec.get("source_run_id"):
            raise ConfigError("a reuse target needs source_run_id")
        return ReuseTarget(spec["source_run_id"])
    if kind == "callable":
        ref = callable_ref or spec.get("callable")
        if not ref:
            raise ConfigError('a callable target needs `callable = "pkg.mod:fn"` or --target')
        return CallableTarget(
            load_callable(ref),
            name=spec.get("name") or ref,
            fingerprint=fingerprint or spec.get("fingerprint"),
            on_empty=spec.get("on_empty", "allow"),
        )
    options = {k: v for k, v in spec.items() if k not in ("kind", "template")}
    return ModelTarget(
        client_from_env("EVALKIT_TARGET"), spec.get("template", "{prompt}"), **options
    )


def _judge_specs(raw: list[Any]) -> list[EvaluatorSpec]:
    """Evaluator specs from a spec file; a judge's provider/model default to the environment's."""
    specs = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or not {"kind", "name"} <= set(item)
            or set(item) - {"kind", "name", "params", "version"}
        ):
            raise UsageError("each evaluator is {kind, name, params?} and nothing else")
        params = dict(item.get("params", {}))
        if item["kind"] == "llm_judge" and not {"provider", "model"} <= set(params):
            client = client_from_env("EVALKIT_JUDGE")
            params = {"provider": client.provider, "model": client.model, **params}
        specs.append(EvaluatorSpec(kind=item["kind"], name=item["name"], params=params))
    return specs


def _target_for_run(run: Run, args: argparse.Namespace) -> Target | None:
    spec = run.config.target
    if spec.kind in ("precomputed", "reuse"):
        return None  # rebuilt by the controller
    if spec.kind == "callable":
        ref = args.target
        if not ref:
            raise ConfigError(
                "this run's target is a callable: pass --target package.module:function"
            )
        ident = spec.identity
        return CallableTarget(
            load_callable(ref),
            name=ident.get("name"),
            fingerprint=args.fingerprint or ident.get("fingerprint"),
            on_empty=ident.get("on_empty", "allow"),
        )
    ident = spec.identity
    return ModelTarget(
        client_from_env("EVALKIT_TARGET"),
        ident["template"],
        system=ident["system"],
        temperature=ident["temperature"],
        max_tokens=ident["max_tokens"],
        on_empty=ident["on_empty"],
    )


def _clients(run: Run) -> list[Any]:
    if any(e.kind == "llm_judge" for e in run.config.evaluators):
        return [client_from_env("EVALKIT_JUDGE")]
    return []


def _ops_settings(args: argparse.Namespace, base: dict[str, Any]) -> dict[str, Any]:
    """Pricing / cache / budget from the command line, merged over `base` (the spec file's policy
    at `create`, the run's frozen policy at `execute`): a flag changes only what it names."""
    out: dict[str, Any] = {}
    if getattr(args, "pricing", None):
        out["pricing"] = PriceTable.from_file(args.pricing).to_mapping()
    cache = {}
    if getattr(args, "cache", None):
        cache["mode"] = args.cache
    if getattr(args, "cache_targets", False):
        cache["targets"] = True
    if cache:
        out["cache"] = {**(base.get("cache") or {}), **cache}
    budget = {
        k: v
        for k, v in (
            ("max_tokens", getattr(args, "max_tokens", None)),
            ("max_cost_usd", getattr(args, "max_cost_usd", None)),
            ("max_calls", getattr(args, "max_calls", None)),
        )
        if v is not None
    }
    if budget:
        out["budget"] = {**(base.get("budget") or {}), **budget}
    return out


def _execute(kit: EvalKit, args: argparse.Namespace) -> tuple[int, Any]:
    run = kit.runs.get(args.id)
    target = _target_for_run(run, args)
    overrides = _ops_settings(args, run.config.policy)
    token = CancelToken()
    previous = signal.getsignal(signal.SIGINT)

    def on_sigint(signum: int, frame: Any) -> None:
        if token.cancelled:
            raise KeyboardInterrupt  # second Ctrl-C: give up waiting
        token.cancel("interrupted")
        print(
            "interrupted: finishing the units in flight (Ctrl-C again to stop waiting)",
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, on_sigint)
    try:
        report = kit.controller.execute(
            run.id,
            target=target,
            clients=_clients(run),
            token=token,
            overrides=overrides,
            retry_failed=args.retry_failed,
        )
    finally:
        signal.signal(signal.SIGINT, previous)
    status = run_status(kit, run.id)
    if not args.quiet:
        print(format_status(status), file=sys.stderr)
    out = {
        "run": report.run.model_dump(mode="json"),
        "counts": asdict(report.counts),
        "stop_reason": report.stop_reason,
        "stop_detail": report.stop_detail,
        "writer_error": report.writer_error,
        "units": report.units,
        "elapsed_s": round(report.elapsed_s, 3),
        "units_per_s": round(report.units_per_s, 1),
        "spend": asdict(report.spend),
        "run_spend": None if report.run_spend is None else asdict(report.run_spend),
        "retry": report.retry,
        "spilled": report.spilled,
        "operations": status,
    }
    return (0 if report.status == "succeeded" else 1), out


def runs_command(args: argparse.Namespace) -> tuple[int, Any]:
    """`evalkit runs ...`"""
    kit = EvalKit.from_env()
    try:
        rs = kit.runs
        match args.runs_command:
            case "create":
                spec = _read_spec(args.config)
                target = build_target(
                    spec.get("target", {}), callable_ref=args.target, fingerprint=args.fingerprint
                )
                policy = dict(spec.get("policy") or {})
                policy.update(_ops_settings(args, policy))
                run = kit.controller.create(
                    args.dataset,
                    target=target,
                    evaluators=_judge_specs(spec.get("evaluators", [])),
                    policy=policy,
                    name=args.name,
                    idempotency_key=args.idempotency_key,
                )
                return 0, run.model_dump(mode="json")
            case "execute" | "resume":
                return _execute(kit, args)
            case "list":
                runs = rs.list(dataset_ref=args.dataset, status=args.status, limit=args.limit)
                return 0, [r.model_dump(mode="json") for r in runs]
            case "show":
                summary = kit.summarize(args.id).to_dict() if args.summary else None
                out = {
                    "run": rs.get(args.id).model_dump(mode="json"),
                    "counts": asdict(rs.counts(args.id)),
                    "failure_counts": [asdict(c) for c in rs.failure_counts(args.id)],
                    "tags": kit.store.run_tags(args.id),
                    "retry": kit.store.retry_stats(args.id),
                    "executions": [
                        {k: e[k] for k in ("seq", "started_at", "finished_at", "retry_failed",
                                           "status_before", "outcome_status", "stop_reason",
                                           "units", "elapsed_s", "reopened_cases",
                                           "reopened_evaluators")}
                        for e in kit.store.list_executions(args.id)
                    ],
                }  # fmt: skip
                if summary is not None:
                    out["summary"] = summary
                return 0, out
            case "status":
                status = run_status(kit, args.id)
                if args.format == "text":
                    print(format_status(status))
                    return 0, None
                return 0, status
            case "failures" if args.history:
                return 0, kit.store.failure_history(args.id)
            case "failures":
                found = rs.failures(args.id, failure_class=args.failure_class, limit=args.limit)
                return 0, [
                    {**asdict(f), "failure": f.failure.model_dump(mode="json")} for f in found
                ]
            case "tag":
                if args.remove:
                    return 0, {"removed": kit.store.remove_run_tag(args.id, args.tag)}
                kit.store.add_run_tag(args.id, args.tag)
                return 0, {"tags": kit.store.run_tags(args.id)}
            case "review":
                review_id = kit.reviews.add(
                    args.id, args.case_key, reviewer=args.reviewer, verdict=args.verdict,
                    evaluator_key=args.evaluator, score=args.score, comment=args.comment,
                    sample=args.sample,
                )  # fmt: skip
                return 0, {"review_id": review_id}
            case "calibrate":
                return 0, kit.reviews.calibrate(args.id, args.evaluator, n_min=args.n_min).to_dict()
            case "queue":
                queue = kit.reviews.queue(
                    args.id, args.evaluator, strategy=args.strategy, n=args.n, seed=args.seed
                )
                return 0, queue
            case "disagreements":
                return 0, kit.reviews.disagreements(args.id, tau=args.tau)
            case _:  # verify
                report = rs.verify(args.id)
                return (0 if report.ok else 1), asdict(report)
    finally:
        kit.close()


def cache_command(args: argparse.Namespace) -> tuple[int, Any]:
    """`evalkit cache stats | clear [--older-than-days N]`"""
    from datetime import UTC, datetime, timedelta

    kit = EvalKit.from_env()
    try:
        if args.cache_command == "stats":
            return 0, kit.store.cache_stats()
        cutoff = None
        if args.older_than_days is not None:
            if not 0 <= args.older_than_days < 36_500:
                raise UsageError("--older-than-days must be between 0 and 36500")
            cutoff = (datetime.now(UTC) - timedelta(days=args.older_than_days)).isoformat()
        return 0, {"deleted": kit.store.cache_clear(cutoff)}
    finally:
        kit.close()


def compare_command(args: argparse.Namespace) -> tuple[int, Any]:
    """`evalkit compare CANDIDATE --baseline RUN|tag:NAME [--gates FILE]`"""
    kit = EvalKit.from_env()
    try:
        candidate = kit.runs.get(args.candidate).id
        baseline = resolve_baseline(kit, args.baseline, candidate)
        cmp = compare(
            kit, baseline, candidate, allow_confounders=args.allow_confounders, seed=args.seed
        )
        out: dict[str, Any] = {"comparison": cmp.to_dict()}
        code = EXIT_PASS
        if args.gates:
            gates = Gates.from_file(args.gates)
            if args.strict:
                gates = Gates(
                    gates.metrics,
                    gates.floors,
                    gates.must_pass,
                    gates.min_coverage,
                    gates.min_n,
                    True,
                )
            result = evaluate_gates(kit, candidate, gates, cmp, seed=args.seed)
            out["gate"] = result.to_dict()
            code = result.exit_code
        assert code in (EXIT_PASS, EXIT_REGRESSION, EXIT_INCONCLUSIVE)
        return code, out
    except ValueError as e:  # bad gates file
        raise UsageError(str(e)) from e
    finally:
        kit.close()


def report_command(args: argparse.Namespace) -> tuple[int, Any]:
    """`evalkit report RUN --out FILE [--compare BASELINE] [--gates FILE]`"""
    kit = EvalKit.from_env()
    try:
        run_id = kit.runs.get(args.id).id
        comparison = gate = None
        if args.compare:
            baseline = resolve_baseline(kit, args.compare, run_id)
            comparison = compare(kit, baseline, run_id, allow_confounders=args.allow_confounders)
            if args.gates:
                gate = evaluate_gates(kit, run_id, Gates.from_file(args.gates), comparison)
        path = write_report(
            kit, run_id, args.out, overwrite=args.force, comparison=comparison, gate=gate,
            max_cases=args.max_cases,
        )  # fmt: skip
        return 0, {"path": str(path), "bytes": path.stat().st_size,
                   "gate": None if gate is None else gate.status}  # fmt: skip
    except ValueError as e:
        raise UsageError(str(e)) from e
    finally:
        kit.close()


def _judge_spec_of_run(kit: EvalKit, run_id: str, selector: str | None) -> EvaluatorSpec:
    """The run's own frozen llm_judge spec: its key is what reports look the check up by."""
    run = kit.runs.get(run_id)
    judges = [s for s in run.config.evaluators if s.kind == "llm_judge"]
    if selector is not None:
        judges = [s for s in judges if selector in (s.name, s.key)]
        if not judges:
            raise ConfigError(
                f"run {run.id} has no llm_judge evaluator named or keyed {selector!r}"
            )
    if not judges:
        raise ConfigError(f"run {run.id} has no llm_judge evaluator to check")
    if len(judges) > 1:
        names = ", ".join(s.name for s in judges)
        raise ConfigError(
            f"run {run.id} has several llm_judge evaluators ({names}): pass --evaluator"
        )
    return judges[0]


def judge_check_command(args: argparse.Namespace) -> tuple[int, Any]:
    """`evalkit judge-check [--run RUN [--evaluator NAME]] [--cases FILE] [--rubric FILE]`

    With `--run` the check is of *that run's* judge evaluator, built from its frozen spec, so its
    result is stored under (and shown for) the evaluator key the run actually used. Without it, a
    standalone check of the environment-configured judge under `--name`."""
    from evalkit.evaluators import llm_judge_spec
    from evalkit.models import Rubric

    if args.run and args.rubric:
        raise UsageError("--rubric cannot be combined with --run (the run's rubric is used)")
    if not args.run and args.evaluator:
        raise UsageError("--evaluator needs --run")
    client = client_from_env("EVALKIT_JUDGE")
    kit = EvalKit.from_env()
    try:
        if args.run:
            spec = _judge_spec_of_run(kit, args.run, args.evaluator)
            frozen = spec.params
            if (client.provider, client.model) != (frozen["provider"], frozen["model"]):
                raise ConfigError(
                    f"the run's judge is {frozen['provider']}/{frozen['model']} but the configured "
                    f"judge is {client.provider}/{client.model}: a check of another judge would be "
                    "stored under a key it does not belong to"
                )
            if not args.cases and frozen["rubric_content_hash"] != BUILTIN_RUBRIC.content_hash:
                raise ConfigError(
                    "the built-in golden set was written for the built-in rubric only; this run's "
                    "rubric differs, so supply known-verdict cases for it with --cases"
                )
        else:
            rubric = (
                Rubric.model_validate(_read_json(args.rubric)) if args.rubric else BUILTIN_RUBRIC
            )
            spec = resolve(llm_judge_spec(args.name or "judge-check", rubric, client))
        cases = load_cases(args.cases) if args.cases else builtin_cases()
        name = Path(args.cases).name if args.cases else "builtin-v1"
        result = judge_check(kit, spec, client, cases, fixture_name=name)
        out = result.to_dict()
        if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
            return EXIT_REGRESSION, out
        return 0, out
    finally:
        kit.close()


def _read_json(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return json.loads(f.read(MAX_INPUT_FILE_BYTES).decode("utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e


__all__ = [
    "DEFAULT_MAX_CASES",
    "EvalKitError",
    "build_target",
    "cache_command",
    "compare_command",
    "judge_check_command",
    "load_callable",
    "report_command",
    "runs_command",
    "UsageError",
]
