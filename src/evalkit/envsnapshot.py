"""The reproducibility snapshot of a run (docs/TARGET-ARCHITECTURE.md §10): what a run was measured
*with*, recorded so that when two runs differ, the reason can be found and a difference that can
change scores is visible as a confounder.

Two records:

* the **frozen environment** (`runs.environment_json`, written at creation): the configuration
  that defines the measurement -- dataset identity, target and judge provider / model / generation
  settings, evaluator and scoring versions, retry and timeout settings, cache and pricing
  configuration -- plus the runtime it was created under;
* one **execution environment** per `execute` (`run_executions.environment_json`): the runtime
  that actually made the calls (EvalKit version and commit, Python, SDK versions) and the
  endpoints (region / host) of the clients. A resume on another machine or version is therefore
  visible.

Never recorded: credentials, tokens, keys, prompts, outputs, or a URL beyond its host. Every value
is a version string, a number, a label or an identifier.

Which differences are *confounders* (they can change what is measured or how invalid outputs are
handled) and which are merely informational (they change speed, cost or nothing) is decided in
`environment_differences`; see docs/EVALUATION-METHODOLOGY.md.
"""

from __future__ import annotations

import functools
import importlib.metadata
import platform
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evalkit.hashing import stable_hash
from evalkit.pricing import PriceTable

SCHEMA = 2  # the frozen environment's layout (runs created before P2.1 have none)
_SDKS = ("boto3", "openai", "pydantic", "jsonschema")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """The commit of the EvalKit checkout this code runs from. Only when this package *is* a
    checkout's `src/evalkit` (an installed wheel, or a copy inside some other project's virtualenv,
    has no EvalKit commit and must not report the enclosing repository's). Best effort: any failure
    is None."""
    here = Path(__file__).resolve().parent
    try:
        top = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2, check=False,
        )  # fmt: skip
        if top.returncode != 0 or (Path(top.stdout.strip()) / "src" / "evalkit").resolve() != here:
            return None
        out = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


def runtime_facts() -> dict[str, Any]:
    return {
        "evalkit_version": _version("evalkit") or "unknown",
        "evalkit_commit": git_commit(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": f"{platform.system()} {platform.machine()}",
        "sqlite": sqlite3.sqlite_version,
        "sdk": {name: v for name in _SDKS if (v := _version(name)) is not None},
    }


def endpoints_of(clients: Iterable[Any]) -> dict[str, str | None]:
    """`provider/model` -> the endpoint (region or host) the client reports, if it does."""
    return {f"{c.provider}/{c.model}": getattr(c, "endpoint", None) for c in clients}


def _generation(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in ("temperature", "max_tokens") if k in params}


def frozen_environment(
    *,
    dataset: Mapping[str, Any],
    target_kind: str,
    target_identity: Mapping[str, Any],
    evaluators: Sequence[Any],  # resolved EvaluatorSpec
    scoring_version: int,
    policy: Mapping[str, Any],
    retry: Mapping[str, Any],
    pricing: PriceTable,
    cache: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        **runtime_facts(),
        "dataset": dict(dataset),
        "scoring_version": scoring_version,
        "target": {
            "kind": target_kind,
            "provider": target_identity.get("provider"),
            "model": target_identity.get("model"),
            **_generation(target_identity),
            "identity_sha256": stable_hash("evalkit-env-target-v1", dict(target_identity)),
        },
        "evaluators": [
            {
                "key": e.key,
                "kind": e.kind,
                "name": e.name,
                "version": e.version,
                "provider": e.params.get("provider"),
                "model": e.params.get("model"),
                **_generation(e.params),
                "prompt_version": e.params.get("prompt_version"),
                "scoring_version": e.params.get("scoring_version"),
                "rubric_content_hash": e.params.get("rubric_content_hash"),
            }
            for e in evaluators
        ],
        "retry": dict(retry),
        "request_timeout_s": retry.get("timeout_s"),
        "unit_deadline_s": retry.get("unit_deadline_s") or 3 * (retry.get("timeout_s") or 0),
        "pricing": {
            "version": pricing.version,
            "sha256": pricing.content_hash,
            "models": len(pricing.models),
        },
        "cache": dict(cache),
        "budget": dict(budget),
    }


def execution_environment(clients: Iterable[Any], *, cache_mode: str, pricing: PriceTable):
    return {
        **runtime_facts(),
        "endpoints": endpoints_of(clients),
        "cache_mode": cache_mode,
        "pricing_version": pricing.version,
    }


# ---- comparing two environments ----------------------------------------------------------------

# Confounders: a difference here can change what is measured or how a bad answer is handled, so a
# score difference cannot be attributed to what you varied. Everything else is informational.
#   endpoints           region / host a model was served from (a different deployment can answer
#                       differently under the same model id)
#   validation_retries  how many times an unusable judge answer is retried with feedback: changes
#                       which cases are scored at all
_INFO_KEYS = (
    ("evalkit_version", "EvalKit version"),
    ("evalkit_commit", "EvalKit commit"),
    ("python", "Python"),
    ("python_implementation", "Python implementation"),
    ("platform", "platform"),
    ("sqlite", "SQLite"),
)


def _endpoints(executions: Sequence[Mapping[str, Any]]) -> set[str]:
    seen: set[str] = set()
    for env in executions:
        for model, endpoint in (env.get("endpoints") or {}).items():
            if endpoint:
                seen.add(f"{model} @ {endpoint}")
    return seen


def environment_differences(
    base: Mapping[str, Any],
    cand: Mapping[str, Any],
    base_executions: Sequence[Mapping[str, Any]] = (),
    cand_executions: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[tuple[str, str]], list[str]]:
    """(confounders as (kind, detail), informational notes). A side with no recorded value for
    something (a run from before the snapshot existed, or a client that reports no endpoint) is
    unknown, not different: it produces a note, never a confounder."""
    confounders: list[tuple[str, str]] = []
    notes: list[str] = []

    be, ce = _endpoints(base_executions), _endpoints(cand_executions)
    if be and ce and be != ce:
        confounders.append(
            (
                "environment.endpoint",
                f"served from different endpoints ({', '.join(sorted(be))} vs "
                f"{', '.join(sorted(ce))}): the same model id can answer differently",
            )
        )
    elif bool(be) != bool(ce):
        notes.append("the endpoint is recorded for only one of the runs")

    bv = (base.get("retry") or {}).get("validation_retries")
    cv = (cand.get("retry") or {}).get("validation_retries")
    if bv is not None and cv is not None and bv != cv:
        confounders.append(
            (
                "environment.validation_retries",
                f"unusable judge answers are retried {bv} vs {cv} time(s): this changes which "
                "cases are scored",
            )
        )
    for key, label in (
        ("max_attempts", "transport attempts"),
        ("unit_deadline_s", "unit deadline"),
    ):
        b, c = (base.get("retry") or {}).get(key), (cand.get("retry") or {}).get(key)
        if b is not None and c is not None and b != c:
            notes.append(f"{label} differ ({b} vs {c}): shows up as coverage, not as a score")

    for key, label in _INFO_KEYS:
        b, c = base.get(key), cand.get(key)
        if b is not None and c is not None and b != c:
            notes.append(f"{label} differs ({b} vs {c})")
    bs, cs = base.get("sdk") or {}, cand.get("sdk") or {}
    for name in sorted(set(bs) | set(cs)):
        if bs.get(name) and cs.get(name) and bs[name] != cs[name]:
            notes.append(f"{name} version differs ({bs[name]} vs {cs[name]})")
    bp, cp = (base.get("pricing") or {}), (cand.get("pricing") or {})
    if bp and cp and bp.get("sha256") != cp.get("sha256"):
        notes.append(
            f"price tables differ ({bp.get('version')} vs {cp.get('version')}): cost estimates are "
            "not comparable"
        )
    if not base.get("schema") or not cand.get("schema"):
        notes.append("the environment snapshot is missing or older for one of the runs")
    return confounders, notes
