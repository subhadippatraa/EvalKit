"""Thin CLI over Evaluator. Exit codes: 0 ok (incl. FAIL verdict), 1 evaluation error, 2 usage."""

import argparse
import inspect
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from itertools import islice

from evalkit import safejson
from evalkit.errors import EvalKitError
from evalkit.evaluator import Evaluator
from evalkit.kit import EvalKit
from evalkit.limits import MAX_INPUT_FILE_BYTES
from evalkit.redact import scrub

# keys evaluate() accepts, for a clear "unknown key" message before calling it
_RUN_SIGNATURE = inspect.signature(Evaluator.evaluate)
_RUN_PARAMS = set(_RUN_SIGNATURE.parameters) - {"self"}


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description="LLM-as-judge evaluation")
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
