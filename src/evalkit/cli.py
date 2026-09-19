"""Thin CLI over Evaluator. Exit codes: 0 ok (incl. FAIL verdict), 1 evaluation error, 2 usage."""

import argparse
import inspect
import json
import sys

from evalkit.errors import EvalKitError
from evalkit.evaluator import Evaluator

# keys evaluate() accepts, for a clear "unknown key" message before calling it
_RUN_PARAMS = set(inspect.signature(Evaluator.evaluate).parameters) - {"self"}


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

    review = sub.add_parser("review", help="add a human review")
    review.add_argument("id")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", required=True, type=str.upper, choices=["PASS", "FAIL"])
    review.add_argument("--score", type=float, help="optional overall score in [0, 1]")
    review.add_argument("--comment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            try:
                with open(args.input) as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"error: cannot read input file: {e}", file=sys.stderr)
                return 2
            if not isinstance(payload, dict):
                print("error: input file must contain a JSON object", file=sys.stderr)
                return 2
            if unknown := sorted(set(payload) - _RUN_PARAMS):
                print(f"error: invalid input file: unknown key(s) {unknown}", file=sys.stderr)
                return 2
            evaluator = Evaluator.from_env()
            try:
                output = evaluator.evaluate(**payload).model_dump(mode="json")
            except TypeError as e:  # e.g. a required key missing from the input file
                print(f"error: invalid input file: {e}", file=sys.stderr)
                return 2
        elif args.command == "get":
            output = Evaluator.from_env(with_judge=False).get(args.id).model_dump(mode="json")
        elif args.command == "list":
            results = Evaluator.from_env(with_judge=False).list(tag=args.tag, limit=args.limit)
            output = [r.model_dump(mode="json") for r in results]
        else:
            output = (
                Evaluator.from_env(with_judge=False)
                .review(args.id, args.reviewer, args.verdict, args.score, args.comment)
                .model_dump(mode="json")
            )
    except EvalKitError as e:
        print(f"error: {e}", file=sys.stderr)
        if evaluation_id := getattr(e, "evaluation_id", None):
            print(f"evaluation_id: {evaluation_id}", file=sys.stderr)
        return 1

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
