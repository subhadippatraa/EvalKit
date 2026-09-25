"""Targets: where a case's output comes from (docs/TARGET-ARCHITECTURE.md §5).

    precomputed  the case already carries its output (dataset scoring, the core path)
    reuse        copy the outputs of an earlier run (re-scoring with new evaluators)
    callable     a Python function (trusted operator code), wrapped for timeout and classification
    model        a prompt template over an `LLMClient`

A target sees a *view* of the case (`TargetInput`): prompt and context, never the reference,
relevance labels, tags or metadata. The only output it ever receives is `provided`, and only the
engine sets it, only for precomputed / reuse targets. Every failure is a classified `EvalFailure`
of class `target` (or `input` / `infrastructure` where that is the truth) -- never a score.
Every external call goes through `UnitCalls` and so leaves an Attempt.

No HTTP target exists (P3, needs the SSRF policy of design §14.4): wrap a service in a `callable`.
"""

from __future__ import annotations

import hashlib
import json
import string
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from evalkit.calls import UnitCalls, describe_llm_response
from evalkit.datasets import EvaluationCase
from evalkit.errors import ConfigError
from evalkit.failures import EvalFailure, Failure, FailureClass
from evalkit.limits import MAX_DOC_ID_CHARS, MAX_DOC_IDS
from evalkit.llm import (
    STOP_CONTENT_FILTER,
    STOP_CONTEXT_WINDOW,
    STOP_GUARDRAIL,
    STOP_MALFORMED,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Usage,
)
from evalkit.runs import TargetSpec

MAX_META_BYTES = 4096
MAX_TEMPLATE_BYTES = 16 * 1024
_TEMPLATE_FIELDS = frozenset({"prompt", "context"})


@dataclass(frozen=True)
class TargetOutput:
    output: str
    retrieved: list[str] | None = None  # ranked doc ids, for RAG targets
    usage: Usage | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # non-secret, size-capped


@dataclass(frozen=True)
class TargetInput:
    """What a target may see. There is deliberately no field that could carry a reference answer,
    relevance labels or any evaluator-only information."""

    case_key: str
    prompt: str
    context: str | None = None
    # the output the case (precomputed) or a source run (reuse) already has; never set for
    # callable / model targets, and it is the system's own output, not evaluator information
    provided: TargetOutput | None = None
    inherited_failure: Failure | None = None  # reuse: the source run's failure for this case


def view(case: EvaluationCase, *, provide_output: bool = False) -> TargetInput:
    """The target's view of a case. `provide_output` is for the precomputed kind only."""
    provided = None
    if provide_output and case.output is not None:
        provided = TargetOutput(case.output, retrieved=case.retrieved)
    return TargetInput(case.case_key, case.prompt, case.context, provided)


class Target(Protocol):
    kind: str
    identity: dict[str, Any]  # goes into the run's identity hash

    def generate(self, inp: TargetInput, call: UnitCalls) -> TargetOutput: ...


def spec_of(target: Target) -> TargetSpec:
    return TargetSpec(kind=target.kind, identity=target.identity)  # type: ignore[arg-type]


def _contract(message: str, raw: Any = None) -> EvalFailure:
    return EvalFailure(FailureClass.TARGET, "contract_violation", message, raw=raw)


def coerce_output(value: Any) -> TargetOutput:
    """What a callable returned -> `TargetOutput`, or `target.contract_violation`."""
    if isinstance(value, str):
        return TargetOutput(value)
    if not isinstance(value, TargetOutput):
        raise _contract(
            f"a target must return str or TargetOutput, got {type(value).__name__}", raw=repr(value)
        )
    if not isinstance(value.output, str):
        raise _contract(f"TargetOutput.output must be str, got {type(value.output).__name__}")
    try:
        value.output.encode("utf-8")
    except UnicodeEncodeError:
        raise _contract(
            "TargetOutput.output is not valid Unicode text (unpaired surrogate)"
        ) from None
    ids = value.retrieved
    if ids is not None and (
        not isinstance(ids, list)
        or len(ids) > MAX_DOC_IDS
        or not all(isinstance(d, str) and 0 < len(d) <= MAX_DOC_ID_CHARS for d in ids)
    ):
        raise _contract(
            f"TargetOutput.retrieved must be a list of at most {MAX_DOC_IDS} non-empty doc-id "
            f"strings (each up to {MAX_DOC_ID_CHARS} characters)"
        )
    try:
        meta = json.dumps(value.meta, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise _contract(f"TargetOutput.meta must be plain finite JSON: {e}") from e
    if len(meta.encode("utf-8", "replace")) > MAX_META_BYTES:
        raise _contract(f"TargetOutput.meta is larger than {MAX_META_BYTES} bytes")
    if value.usage is not None and not isinstance(value.usage, Usage):
        raise _contract("TargetOutput.usage must be an evalkit.llm.Usage")
    return value


class PrecomputedTarget:
    """The output is in the dataset. Makes no call, so it has no latency and no attempts."""

    kind = "precomputed"

    def __init__(self) -> None:
        self.identity: dict[str, Any] = {}

    def generate(self, inp: TargetInput, call: UnitCalls) -> TargetOutput:
        if inp.provided is None:
            raise EvalFailure(
                FailureClass.INPUT,
                "missing_field",
                f"case {inp.case_key!r} has no `output` to score",
            )
        return inp.provided


class ReuseTarget:
    """Outputs copied from an earlier run (design 5.3). A source target failure is inherited: this
    run records the same failure for that case, and its evaluators are `skipped`."""

    kind = "reuse"

    def __init__(self, source_run_id: str):
        self.source_run_id = source_run_id
        self.identity = {"source_run_id": source_run_id}

    def generate(self, inp: TargetInput, call: UnitCalls) -> TargetOutput:
        if inp.inherited_failure is not None:
            f = inp.inherited_failure
            raise EvalFailure(
                f.failure_class,
                f.kind,
                f"inherited from source run {self.source_run_id}: {f.message}",
                retryable=f.retryable,
            )
        if inp.provided is None:
            raise EvalFailure(
                FailureClass.INPUT,
                "missing_field",
                f"source run {self.source_run_id} has no result for case {inp.case_key!r}",
            )
        return inp.provided


def _invoke(fn: Callable[[TargetInput], Any], inp: TargetInput, timeout_s: float) -> Any:
    """Run trusted code with a wall-clock limit. Python cannot kill a thread: on timeout the
    function keeps running in the background (documented); the result is discarded."""
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn(inp)
        except BaseException as e:  # noqa: BLE001 - whatever the target raised is the target's failure
            box["error"] = e

    thread = threading.Thread(target=work, name="evalkit-target", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise EvalFailure(
            FailureClass.TARGET,
            "timeout",
            f"target did not answer within {timeout_s:g}s (its function may still be running)",
        )
    if "error" in box:
        e = box["error"]
        raise EvalFailure(
            FailureClass.TARGET, "exception", f"{type(e).__name__}: {e}", exc_type=type(e).__name__
        ) from e
    return box["value"]


def _describe_returned(value: Any) -> dict[str, Any]:
    """Attempt fields from what a callable returned: the usage it declared, if any."""
    usage = getattr(value, "usage", None)
    if isinstance(usage, Usage):
        return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    return {}


class CallableTarget:
    """A Python function `(TargetInput) -> str | TargetOutput` (trusted operator code: loaded only
    from code or CLI flags, never from a dataset or the database).

    EvalKit cannot hash code, so the identity is what you declare: pass `fingerprint` (e.g. a git
    SHA) to make two runs of different code distinguishable; without one the run is an "unpinned
    target" and gates refuse it. The function may be called more than once for a case (a timeout is
    retried once), so it should be safe to call twice.
    """

    kind = "callable"

    def __init__(
        self,
        fn: Callable[[TargetInput], Any],
        *,
        name: str | None = None,
        fingerprint: str | None = None,
        on_empty: Literal["allow", "fail"] = "allow",
    ):
        if not callable(fn):
            raise ConfigError("a callable target needs a callable")
        if on_empty not in ("allow", "fail"):
            raise ConfigError("on_empty must be 'allow' or 'fail'")
        self.fn = fn
        self.on_empty = on_empty
        self.name = name or f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}"
        self.fingerprint = fingerprint
        self.identity = {"name": self.name, "fingerprint": fingerprint, "on_empty": on_empty}

    @property
    def pinned(self) -> bool:
        return self.fingerprint is not None

    def generate(self, inp: TargetInput, call: UnitCalls) -> TargetOutput:
        out = call.run(
            lambda _feedback: _invoke(self.fn, inp, call.timeout_s),
            validate=coerce_output,
            describe=_describe_returned,
            provider="callable",
            model=self.name[:128],
        )
        if out.output == "" and self.on_empty == "fail":
            raise EvalFailure(FailureClass.TARGET, "empty_output", "the target returned nothing")
        return out


class ModelTarget:
    """A prompt template over an `LLMClient`: "same dataset, different model or prompt" is a
    one-line change. Template fields are limited to `{prompt}` and `{context}` (a literal brace is
    `{{`); the case's reference and labels cannot reach the prompt."""

    kind = "model"

    def __init__(
        self,
        client: LLMClient,
        template: str = "{prompt}",
        *,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        on_empty: Literal["allow", "fail"] = "allow",
    ):
        self.client = client
        self.template = template
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.on_empty = on_empty
        if on_empty not in ("allow", "fail"):
            raise ConfigError("on_empty must be 'allow' or 'fail'")
        if len(template.encode("utf-8", "replace")) > MAX_TEMPLATE_BYTES:
            raise ConfigError(f"template is larger than {MAX_TEMPLATE_BYTES} bytes")
        try:
            parsed = list(string.Formatter().parse(template))
        except ValueError as e:
            raise ConfigError(f"invalid template: {e}") from e
        bad = {
            n
            for _, n, spec, conv in parsed
            if n is not None and (n not in _TEMPLATE_FIELDS or spec or conv)
        }
        if bad:
            raise ConfigError(
                f"template fields are limited to {sorted(_TEMPLATE_FIELDS)}; got {sorted(bad)}"
            )
        self.identity = {
            "provider": client.provider,
            "model": client.model,
            "template": template,
            "template_sha256": hashlib.sha256(template.encode("utf-8", "replace")).hexdigest(),
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "on_empty": on_empty,
        }

    def render(self, inp: TargetInput) -> str:
        return self.template.format(prompt=inp.prompt, context=inp.context or "")

    def generate(self, inp: TargetInput, call: UnitCalls) -> TargetOutput:
        request = LLMRequest(
            system=self.system,
            user=self.render(inp),
            tool=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_s=call.timeout_s,
            role="target",
        )
        return call.run(
            lambda _feedback: self.client.call(request),
            validate=self._output,
            describe=describe_llm_response,
            provider=self.client.provider,
            model=self.client.model,
        )

    def _output(self, resp: LLMResponse) -> TargetOutput:
        blocked = {
            STOP_CONTENT_FILTER: "content filter",
            STOP_GUARDRAIL: "guardrail",
        }.get(resp.stop_reason)
        suffix = f" ({resp.provider_stop})" if resp.provider_stop else ""
        if blocked:
            raise EvalFailure(
                FailureClass.TARGET,
                "blocked",
                f"the model's output was blocked by a {blocked}{suffix}",
                raw=resp.raw,
            )
        if resp.stop_reason == STOP_MALFORMED:
            raise _contract(f"the model response was malformed{suffix}", raw=resp.raw)
        if resp.stop_reason == STOP_CONTEXT_WINDOW:
            raise EvalFailure(
                FailureClass.INPUT,
                "oversize",
                f"the prompt exceeds the model's context window{suffix}",
            )
        text = resp.text or ""
        if text == "" and self.on_empty == "fail":
            raise EvalFailure(
                FailureClass.TARGET, "empty_output", "the model returned no text", raw=resp.raw
            )
        # a max_tokens stop is still an answer (a truncated one): kept, and flagged in the meta
        meta: dict[str, Any] = {"stop_reason": resp.stop_reason}
        if resp.request_id:
            meta["request_id"] = resp.request_id
        return TargetOutput(text, usage=resp.usage, meta=meta)
