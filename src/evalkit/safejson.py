"""Strict JSON parsing for untrusted input: no NaN/Infinity (including overflowing floats such
as 1e999), no duplicate object keys."""

import json
import math
from typing import Any


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {text!r} is not allowed")
    return value


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def loads(text: str) -> Any:
    """Raises ValueError (incl. JSONDecodeError) or RecursionError on hostile input."""
    return json.loads(
        text,
        parse_constant=_reject_constant,
        parse_float=_parse_float,
        object_pairs_hook=_no_duplicates,
    )
