"""Thin CLI over Evaluator. Exit codes: 0 ok (incl. FAIL verdict), 1 evaluation error, 2 usage."""

import argparse
import inspect
import json
import sqlite3
import sys

from evalkit import safejson
from evalkit.errors import EvalKitError
from evalkit.evaluator import Evaluator
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
