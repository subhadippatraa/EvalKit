"""Thin CLI over Evaluator and EvalKit.

Exit codes: 0 ok (incl. a FAIL verdict), 1 error (or a run that did not complete), 2 usage,
3 gate failed, 4 gate inconclusive under --strict.
"""

import argparse
import inspect
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from itertools import islice

from evalkit import cli_platform, events, safejson
from evalkit.errors import EvalKitError
from evalkit.evaluator import Evaluator
from evalkit.failures import FailureClass
from evalkit.kit import EvalKit
from evalkit.limits import MAX_INPUT_FILE_BYTES
from evalkit.redact import scrub
from evalkit.report import DEFAULT_MAX_CASES
from evalkit.runs import TRANSITIONS

# keys evaluate() accepts, for a clear "unknown key" message before calling it
_RUN_SIGNATURE = inspect.signature(Evaluator.evaluate)
_RUN_PARAMS = set(_RUN_SIGNATURE.parameters) - {"self"}


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _ops_flags(p: argparse.ArgumentParser, *, execute: bool) -> None:
    """Pricing, cache and budget options (at `create` they are frozen into the run's policy; at
    `execute` they apply to that execution only)."""
    p.add_argument("--pricing", metavar="FILE", help="price table (.toml/.json), versioned")
    p.add_argument(
        "--cache",
        choices=["off", "readwrite", "replay"],
        help="response cache: off (default), readwrite, or replay (never calls the provider)",
    )
    p.add_argument("--cache-targets", action="store_true", help="also cache model-target calls")
    p.add_argument("--max-tokens", type=_positive_int, help="token budget for the whole run")
    p.add_argument("--max-cost-usd", type=float, help="USD budget for the whole run (needs prices)")
    p.add_argument("--max-calls", type=_positive_int, help="provider-call budget per execution")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description="LLM-as-judge evaluation")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="emit structured run events at this level (default: off; or EVALKIT_LOG_LEVEL)",
    )
    parser.add_argument(
        "--log-format", choices=["json", "text"], help="event format (default json)"
    )
    parser.add_argument("--log-file", help="append events to this file (owner-only) not stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate one input file")
    run.add_argument("--input", required=True, help="JSON file with evaluate() arguments")

    get = sub.add_parser("get", help="show one evaluation with its reviews")
    get.add_argument("id")

    ls = sub.add_parser("list", help="list recent evaluations")
    ls.add_argument("--tag")
    ls.add_argument("--limit", type=_positive_int, default=20)
    ls.add_argument("--cursor", help="continue after a previous page (its next_cursor)")

    sub.add_parser("recover", help="save results a failed database write left in <db>.spill/")

    ds = sub.add_parser("dataset", help="versioned, immutable evaluation datasets")
    dsub = ds.add_subparsers(dest="dataset_command", required=True)
    imp = dsub.add_parser("import", help="import a JSONL file as a new dataset version")
    imp.add_argument("name")
    imp.add_argument("file")
    imp.add_argument("--description")
    dsub.add_parser("list", help="list datasets")
    show = dsub.add_parser("show", help="show a version (name, name@latest, name@3, name@<hash>)")
    show.add_argument("ref")
    show.add_argument("--cases", type=int, default=0, metavar="N", help="also print N cases")
    exp = dsub.add_parser("export", help="write a version as JSONL")
    exp.add_argument("ref")
    exp.add_argument("file")
    exp.add_argument("--force", action="store_true", help="overwrite an existing file")
    lint = dsub.add_parser("lint", help="report duplicates, empty fields, missing labels")
    lint.add_argument("ref")
    ver = dsub.add_parser("verify", help="recompute hashes from stored rows (exit 1 on mismatch)")
    ver.add_argument("ref")

    runs = sub.add_parser("runs", help="create, execute and inspect runs")
    rsub = runs.add_subparsers(dest="runs_command", required=True)
    create = rsub.add_parser("create", help="preflight, freeze and plan a run from a spec file")
    create.add_argument("dataset", help="dataset ref: name, name@latest, name@3, name@<hash>")
    create.add_argument("--config", required=True, help="spec file (.json or .toml)")
    create.add_argument("--name")
    create.add_argument("--idempotency-key")
    create.add_argument("--target", metavar="PKG.MOD:FN", help="a callable target (trusted code)")
    create.add_argument("--fingerprint", help="declared identity of the callable (e.g. a git sha)")
    _ops_flags(create, execute=False)
    for name, help_text in (
        ("execute", "execute a created run"),
        ("resume", "resume a stopped run"),
    ):
        ex = rsub.add_parser(name, help=help_text)
        ex.add_argument("id")
        ex.add_argument("--target", metavar="PKG.MOD:FN", help="the run's callable target")
        ex.add_argument("--fingerprint")
        ex.add_argument(
            "--retry-failed",
            action="store_true",
            help="also give failed units whose failure is retryable another try (a succeeded run "
            "is reopened); nothing that succeeded is repeated",
        )
        ex.add_argument("--quiet", action="store_true", help="no status summary on stderr")
        _ops_flags(ex, execute=True)
    rls = rsub.add_parser("list", help="list runs, newest first")
    rls.add_argument("--dataset", help="only runs of this dataset version")
    rls.add_argument("--status", choices=sorted(TRANSITIONS))
    rls.add_argument("--limit", type=_positive_int, default=20)
    rshow = rsub.add_parser("show", help="show a run with its progress")
    rshow.add_argument("id")
    rshow.add_argument("--summary", action="store_true", help="include the aggregated summary")
    rfail = rsub.add_parser("failures", help="list failures, optionally of one class")
    rfail.add_argument("id")
    rfail.add_argument("--class", dest="failure_class", choices=[c.value for c in FailureClass])
    rfail.add_argument("--limit", type=_positive_int, default=100)
    rfail.add_argument(
        "--history", action="store_true", help="list superseded failures (retried units) instead"
    )
    rstat = rsub.add_parser("status", help="calls, cache, retries, tokens, cost, budget, coverage")
    rstat.add_argument("id")
    rstat.add_argument("--format", choices=["json", "text"], default="json")
    rver = rsub.add_parser("verify", help="recheck a run (exit 1 on mismatch)")
    rver.add_argument("id")
    rtag = rsub.add_parser("tag", help="tag a run (a baseline is `tag:main`)")
    rtag.add_argument("id")
    rtag.add_argument("tag")
    rtag.add_argument("--remove", action="store_true")
    rrev = rsub.add_parser("review", help="add a human review of one case's evaluator verdict")
    rrev.add_argument("id")
    rrev.add_argument("case_key")
    rrev.add_argument("--evaluator", required=True, help="the evaluator key being graded")
    rrev.add_argument("--reviewer", required=True)
    rrev.add_argument("--verdict", required=True, type=str.upper, choices=["PASS", "FAIL"])
    rrev.add_argument("--sample", choices=["random", "targeted"], default="targeted")
    rrev.add_argument("--score", type=float)
    rrev.add_argument("--comment")
    rcal = rsub.add_parser("calibrate", help="judge-vs-human agreement for one evaluator")
    rcal.add_argument("id")
    rcal.add_argument("--evaluator", required=True)
    rcal.add_argument("--n-min", type=_positive_int, default=30)
    rq = rsub.add_parser("queue", help="cases to review next")
    rq.add_argument("id")
    rq.add_argument("--evaluator", required=True)
    rq.add_argument(
        "--strategy",
        choices=["random", "stratified", "uncertain", "disagreement"],
        default="random",
    )
    rq.add_argument("-n", type=_positive_int, default=30)
    rq.add_argument("--seed", type=int, default=0)
    rdis = rsub.add_parser("disagreements", help="cases where evaluators disagree")
    rdis.add_argument("id")
    rdis.add_argument("--tau", type=float, default=0.25)

    cache = sub.add_parser("cache", help="the provider-response cache")
    csub = cache.add_subparsers(dest="cache_command", required=True)
    csub.add_parser("stats", help="entries, size and models in the cache")
    cclear = csub.add_parser("clear", help="delete cached responses")
    cclear.add_argument(
        "--older-than-days", type=float, help="only entries older than this (default: all)"
    )

    cmp_ = sub.add_parser(
        "compare", help="paired comparison with a baseline; gates set the exit code"
    )
    cmp_.add_argument("candidate")
    cmp_.add_argument("--baseline", required=True, help="a run id, or tag:NAME")
    cmp_.add_argument("--gates", help="gates file (.toml or .json)")
    cmp_.add_argument("--allow-confounders", action="store_true")
    cmp_.add_argument("--strict", action="store_true", help="an inconclusive gate exits 4")
    cmp_.add_argument("--seed", type=int, default=0)
    rep = sub.add_parser("report", help="write a self-contained static HTML report")
    rep.add_argument("id")
    rep.add_argument("--out", required=True)
    rep.add_argument("--compare", metavar="BASELINE", help="a run id, or tag:NAME")
    rep.add_argument("--gates")
    rep.add_argument("--allow-confounders", action="store_true")
    rep.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    rep.add_argument("--force", action="store_true", help="overwrite an existing file")
    jc = sub.add_parser("judge-check", help="run the golden set through the configured judge")
    jc.add_argument("--cases", help="JSONL of known-verdict cases (default: the built-in set)")
    jc.add_argument("--rubric", help="rubric JSON for --cases (default: the built-in rubric)")
    jc.add_argument("--min-accuracy", type=float, help="exit 3 if accuracy is lower")
    jc.add_argument("--run", help="check the llm_judge evaluator of this run (its exact spec)")
    jc.add_argument("--evaluator", help="with --run: the judge's name or key (if several)")
    jc.add_argument("--name", help="without --run: the evaluator name to check under")

    review = sub.add_parser("review", help="add a human review")
    review.add_argument("id")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", required=True, type=str.upper, choices=["PASS", "FAIL"])
    review.add_argument("--score", type=float, help="optional overall score in [0, 1]")
    review.add_argument("--comment")
    return parser


def _read_input(path: str) -> dict | None:
    """The parsed input object, or None after printing why the file is unusable."""
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_INPUT_FILE_BYTES + 1)
        if len(data) > MAX_INPUT_FILE_BYTES:
            print(f"error: input file exceeds {MAX_INPUT_FILE_BYTES} bytes", file=sys.stderr)
            return None
        payload = safejson.loads(data.decode("utf-8"))
    except (OSError, ValueError, RecursionError) as e:  # ValueError: bad JSON/UTF-8/NaN/duplicates
        print(f"error: cannot read input file: {scrub(str(e))}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print("error: input file must contain a JSON object", file=sys.stderr)
        return None
    return payload


def _dataset_command(args: argparse.Namespace) -> tuple[int, object]:
    """(exit code, JSON-able output) for `evalkit dataset ...`."""
    kit = EvalKit.from_env()
    try:
        ds = kit.datasets
        match args.dataset_command:
            case "import":
                if not os.path.isfile(args.file):
                    print(f"error: cannot read dataset file: {args.file}", file=sys.stderr)
                    return 2, None
                r = ds.import_jsonl(args.name, args.file, description=args.description)
                return 0, {
                    "created": r.created,
                    "version": r.version.model_dump(mode="json") | {"ref": r.version.ref},
                }
            case "list":
                return 0, [d.model_dump(mode="json") for d in ds.list()]
            case "show":
                v = ds.resolve(args.ref)
                out = v.model_dump(mode="json") | {"ref": v.ref}
                if args.cases > 0:
                    cases = ds.cases(args.ref, batch_size=min(args.cases, 1000))
                    out["cases"] = [c.to_record() for c in islice(cases, args.cases)]
                return 0, out
            case "export":
                r = ds.export_jsonl(args.ref, args.file, overwrite=args.force)
                return 0, {
                    "path": str(r.path),
                    "cases": r.case_count,
                    "content_hash": r.content_hash,
                }
            case "lint":
                return 0, [asdict(f) for f in ds.lint(args.ref)]
            case _:  # verify
                report = ds.verify(args.ref)
                return (0 if report.ok else 1), asdict(report)
    finally:
        kit.close()


_PLATFORM = {
    "runs": cli_platform.runs_command,
    "compare": cli_platform.compare_command,
    "report": cli_platform.report_command,
    "judge-check": cli_platform.judge_check_command,
    "cache": cli_platform.cache_command,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.log_level or args.log_file:
            events.configure_logging(
                (args.log_level or "info").upper(),
                args.log_format or os.environ.get("EVALKIT_LOG_FORMAT") or "json",
                file=args.log_file,
            )
        else:
            events.configure_from_env()
    except (ValueError, OSError) as e:
        print(f"error: cannot configure logging: {scrub(str(e))}", file=sys.stderr)
        return 2
    try:
        return _run(args)
    finally:
        events.reset_logging()


def _run(args: argparse.Namespace) -> int:
    try:
        if args.command == "run":
            payload = _read_input(args.input)
            if payload is None:
                return 2
            if unknown := sorted(set(payload) - _RUN_PARAMS):
                print(f"error: invalid input file: unknown key(s) {unknown}", file=sys.stderr)
                return 2
            try:  # e.g. a required key missing; bound here so a TypeError raised *inside*
                _RUN_SIGNATURE.bind(None, **payload)  # evaluate() is a bug, not a bad input file
            except TypeError as e:
                print(f"error: invalid input file: {e}", file=sys.stderr)
                return 2
            evaluator = Evaluator.from_env()
            output = evaluator.evaluate(**payload).model_dump(mode="json")
        elif args.command == "get":
            output = Evaluator.from_env(with_judge=False).get(args.id).model_dump(mode="json")
        elif args.command == "list":
            results, next_cursor = Evaluator.from_env(with_judge=False).list_page(
                tag=args.tag, limit=args.limit, cursor=args.cursor
            )
            output = [r.model_dump(mode="json") for r in results]
            if next_cursor:
                print(f"next_cursor: {next_cursor}", file=sys.stderr)
        elif args.command == "dataset":
            code, output = _dataset_command(args)
            if output is None:
                return code
            print(json.dumps(output, indent=2))
            return code
        elif args.command in _PLATFORM:
            code, output = _PLATFORM[args.command](args)
            if output is None:
                return code
            print(json.dumps(output, indent=2))
            return code
        elif args.command == "recover":
            report = Evaluator.from_env(with_judge=False).recover_spilled()
            output = {"recovered": report.recovered, "failed": report.failed}
            print(json.dumps(output, indent=2))
            return 1 if report.failed else 0
        else:
            output = (
                Evaluator.from_env(with_judge=False)
                .review(args.id, args.reviewer, args.verdict, args.score, args.comment)
                .model_dump(mode="json")
            )
    except cli_platform.UsageError as e:
        print(f"error: {scrub(str(e))}", file=sys.stderr)
        return 2
    except EvalKitError as e:
        print(f"error: {scrub(str(e))}", file=sys.stderr)
        if evaluation_id := getattr(e, "evaluation_id", None):
            print(f"evaluation_id: {evaluation_id}", file=sys.stderr)
        if spill_path := getattr(e, "spill_path", None):
            print(f"spilled_to: {spill_path}", file=sys.stderr)
        return 1
    except sqlite3.Error as e:
        print(f"error: database error: {scrub(str(e))}", file=sys.stderr)
        return 1
    except OSError as e:  # an unreadable file argument is a usage problem
        print(f"error: cannot read or write a file: {scrub(str(e))}", file=sys.stderr)
        return 2

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
