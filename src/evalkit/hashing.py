"""Canonical JSON and stable hashes (docs/TARGET-ARCHITECTURE.md §3.3).

Everything that must be comparable across time -- dataset content, and later evaluator keys and
run configs -- is hashed through `canonical_json`, so the same logical value always yields the same
bytes: sorted keys, ASCII-only escapes (so lone surrogates cannot fail), compact separators, no
NaN/Infinity. Integers and floats stay distinct (`1` and `1.0` hash differently), which keeps the
hash a function of exactly what was stored.
"""

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any


def _plain(obj: Any, path: str = "$") -> Any:
    """Reduce `obj` to plain JSON types, refusing anything that would hash ambiguously."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise TypeError(f"non-finite number at {path}")
        return obj
    if isinstance(obj, list | tuple):
        return [_plain(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        for key in obj:
            if not isinstance(key, str):  # json would silently stringify 1 and "1" into a clash
                raise TypeError(f"non-string key {key!r} at {path}")
        return {k: _plain(v, f"{path}.{k}") for k, v in obj.items()}
    raise TypeError(f"{type(obj).__name__} at {path} is not JSON-serializable")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        _plain(obj), sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def stable_hash(domain: str, obj: Any) -> str:
    """sha256 hex of `obj`'s canonical JSON, namespaced by `domain` (e.g. "evalkit-case-v1"), so
    hashes of different kinds of thing cannot collide and a scheme change is a new domain."""
    digest = hashlib.sha256(domain.encode() + b"\n" + canonical_json(obj).encode("ascii"))
    return digest.hexdigest()


CASE_DOMAIN = "evalkit-case-v1"
CASE_INPUT_DOMAIN = "evalkit-case-input-v1"
DATASET_DOMAIN = "evalkit-dataset-v1"


def dataset_hash(pairs: Iterable[tuple[str, str]]) -> str:
    """Hash of a dataset: its (case_key, case_content_hash) pairs in ascending case_key order.

    Equivalent in strength to hashing the canonical JSONL of the sorted cases (each case hash
    already commits to its content) but streamable: it needs no re-serialization of the cases.
    Keys cannot contain tab or newline, so the encoding is unambiguous. The caller supplies the
    order; both import and verification take it from `ORDER BY case_key` (byte order).
    """
    digest = hashlib.sha256(DATASET_DOMAIN.encode() + b"\n")
    for case_key, case_hash in pairs:
        digest.update(f"{case_key}\t{case_hash}\n".encode())
    return digest.hexdigest()


def case_input_hash(
    prompt: str,
    reference: str | None,
    context: str | None,
    relevance: dict[str, int] | None,
) -> str:
    """Hash of what a case *asks* and what counts as right: prompt, context, reference and
    relevance labels. Deliberately NOT the system's `output`, its `retrieved` documents, nor free
    metadata or tags (which describe an execution, not the question): two runs of different systems
    over the same questions must pair up case by case, whatever the systems answered. Comparison
    pairs on `(case_key, input_hash)`."""
    return stable_hash(
        CASE_INPUT_DOMAIN,
        {"prompt": prompt, "context": context, "reference": reference, "relevance": relevance},
    )
