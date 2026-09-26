"""Case evaluators (docs/TARGET-ARCHITECTURE.md §6): one lifecycle for every way of scoring.

    exact_match    output equals the reference after declared normalization
    regex          output matches an operator-authored pattern
    json_schema    output is JSON valid against a schema (optional `jsonschema` extra)
    retrieval      recall / precision / hit / MRR / nDCG of the target's retrieved doc ids
    citation_check every citation marker in the output resolves to a retrieved document
    llm_judge      a rubric-scored LLM judge (one implementation, not the architecture)

Rules:

* Deterministic evaluators are pure functions of their input: no clock, no randomness, no calls.
* An evaluator's identity is `EvaluatorSpec.key`, a hash of its kind, name, version and *every*
  parameter that can change a score. `resolve()` fills defaults and derived facts (rubric hash,
  judge prompt fingerprint) into the spec first, so the key cannot drift from behaviour.
* An evaluator sees an `EvalInput` and nothing else: no run, target identity, tags, metadata or
  other evaluators' results.
* A malformed *target* output is a quality signal (score 0 / `valid=0`), never an evaluator
  failure; an evaluator failure is a classified `EvalFailure`; an undefined metric raises
  `NotApplicable` (excluded from means, counted in coverage), never a zero.
* Every metric is finite (`EvaluatorOutcome` enforces it).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from evalkit import safejson
from evalkit.calls import UnitCalls, describe_llm_response
from evalkit.errors import ConfigError, JudgeOutputError, ScoringError
from evalkit.failures import EvalFailure, FailureClass
from evalkit.judge import PROMPT_VERSION
from evalkit.judge_eval import MAX_TOKENS, build_request, judge_payload
from evalkit.llm import LLMClient
from evalkit.models import Rubric
from evalkit.runs import SCORING_VERSION, EvaluatorSpec

MAX_PATTERN_CHARS = 2000
MAX_SCHEMA_BYTES = 128 * 1024
MAX_K = 100
_EVIDENCE_CHARS = 200
_MAX_SCHEMA_ERRORS = 10


@dataclass(frozen=True)
class EvalInput:
    """Everything an evaluator may see."""

    case_key: str
    prompt: str
    output: str
    reference: str | None = None
    context: str | None = None
    relevance: dict[str, int] | None = None
    retrieved: list[str] | None = None  # what the target retrieved, in rank order
    # what the target reported about this output: {"stop_reason", "truncated", ...} or None. It is
    # data for evaluators, never part of a judge's prompt (provenance must not reach the judge).
    target_meta: dict[str, Any] | None = None


class NotApplicable(Exception):
    """The metric is undefined for this case (e.g. recall with no relevant documents)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class EvalOutcome:
    metrics: dict[str, float]
    verdict: str | None = None  # PASS | FAIL | UNCERTAIN
    detail: dict[str, Any] = field(default_factory=dict)


class CaseEvaluator(Protocol):
    kind: str
    key: str
    requires: frozenset[str]  # case fields needed besides the output: reference, context, ...

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome: ...


@dataclass(frozen=True)
class EvalContext:
    """Runtime resources evaluators may need (not part of any identity)."""

    clients: Sequence[LLMClient] = ()


@dataclass(frozen=True)
class EvaluatorKind:
    kind: str
    version: int  # of this kind's scoring logic; part of every key
    requires: frozenset[str]  # case fields needed besides the output (checked by preflight)
    canonical: Callable[[dict[str, Any]], dict[str, Any]]  # validate + fill defaults
    build: Callable[[EvaluatorSpec, EvalContext], CaseEvaluator]


EVALUATOR_KINDS: dict[str, EvaluatorKind] = {}


def register(kind: EvaluatorKind) -> EvaluatorKind:
    EVALUATOR_KINDS[kind.kind] = kind
    return kind


def get_kind(name: str) -> EvaluatorKind:
    try:
        return EVALUATOR_KINDS[name]
    except KeyError:
        raise ConfigError(
            f"unknown evaluator kind {name!r} (known: {', '.join(sorted(EVALUATOR_KINDS))})"
        ) from None


def resolve(spec: EvaluatorSpec) -> EvaluatorSpec:
    """The canonical form of `spec`: validated, defaults filled, derived facts added, version set.
    Idempotent. Runs are created from resolved specs so a key names actual behaviour."""
    kind = get_kind(spec.kind)
    if spec.version != kind.version:
        raise ConfigError(
            f"evaluator kind {spec.kind!r} is at version {kind.version}; the spec says "
            f"{spec.version} (omit `version`)"
        )
    return EvaluatorSpec(
        kind=spec.kind,
        name=spec.name,
        version=kind.version,
        params=kind.canonical(dict(spec.params)),
    )


def build(spec: EvaluatorSpec, ctx: EvalContext | None = None) -> CaseEvaluator:
    """The evaluator for a *resolved* spec (its key must be the resolved key)."""
    resolved = resolve(spec)
    if resolved.key != spec.key:
        raise ConfigError(
            f"evaluator {spec.name!r} is not in canonical form (key {spec.key} vs "
            f"{resolved.key}); build specs with evalkit.evaluators.resolve()"
        )
    return get_kind(spec.kind).build(spec, ctx or EvalContext())


def _only(params: dict[str, Any], allowed: set[str], kind: str) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ConfigError(
            f"{kind}: unknown parameter(s) {sorted(unknown)} (allowed: {sorted(allowed)})"
        )


class _Evaluator:
    requires: frozenset[str] = frozenset()

    def __init__(self, spec: EvaluatorSpec):
        self.spec = spec
        self.kind = spec.kind
        self.key = spec.key
        self.params = spec.params


# ---- exact_match ----------------------------------------------------------------------------

_NORMALIZERS = ("nfkc", "casefold", "strip", "collapse_whitespace")  # application order


def _exact_canonical(p: dict[str, Any]) -> dict[str, Any]:
    _only(p, {"normalize"}, "exact_match")
    steps = p.get("normalize", [])
    if not isinstance(steps, list) or any(s not in _NORMALIZERS for s in steps):
        raise ConfigError(f"exact_match.normalize must be a list drawn from {list(_NORMALIZERS)}")
    return {"normalize": [s for s in _NORMALIZERS if s in steps]}  # order-independent identity


def _normalize(text: str, steps: Sequence[str]) -> str:
    if "nfkc" in steps:
        text = unicodedata.normalize("NFKC", text)
    if "casefold" in steps:
        text = text.casefold()
    if "collapse_whitespace" in steps:
        text = " ".join(text.split())
    if "strip" in steps:
        text = text.strip()
    return text


class ExactMatch(_Evaluator):
    requires = frozenset({"reference"})

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        if inp.reference is None:
            raise EvalFailure(FailureClass.INPUT, "missing_field", "exact_match needs a reference")
        steps = self.params["normalize"]
        a, b = _normalize(inp.output, steps), _normalize(inp.reference, steps)
        match = a == b
        detail: dict[str, Any] = {"normalize": steps}
        if not match:
            detail["first_difference"] = next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
        return EvalOutcome({"match": 1.0 if match else 0.0}, "PASS" if match else "FAIL", detail)


# ---- regex ----------------------------------------------------------------------------------

_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def _compile(pattern: str, flags: Sequence[str]) -> re.Pattern[str]:
    value = 0
    for f in flags:
        value |= _FLAGS[f]
    return re.compile(pattern, value)


def _regex_canonical(p: dict[str, Any]) -> dict[str, Any]:
    _only(p, {"pattern", "flags", "mode"}, "regex")
    pattern = p.get("pattern")
    if not isinstance(pattern, str) or not 0 < len(pattern) <= MAX_PATTERN_CHARS:
        raise ConfigError(f"regex.pattern must be 1-{MAX_PATTERN_CHARS} characters")
    flags = p.get("flags", [])
    if not isinstance(flags, list) or any(f not in _FLAGS for f in flags):
        raise ConfigError(f"regex.flags must be a list drawn from {sorted(_FLAGS)}")
    mode = p.get("mode", "search")
    if mode not in ("search", "fullmatch"):
        raise ConfigError("regex.mode must be 'search' or 'fullmatch'")
    flags = sorted(set(flags))
    try:
        _compile(pattern, flags)
    except re.error as e:
        raise ConfigError(f"regex.pattern is not a valid regular expression: {e}") from e
    return {"pattern": pattern, "flags": flags, "mode": mode}


class RegexMatch(_Evaluator):
    """Note: Python's `re` cannot be interrupted, so a pathological pattern is bounded by the
    output size cap (256 KB), not by time. Patterns are operator-authored, never from data."""

    def __init__(self, spec: EvaluatorSpec):
        super().__init__(spec)
        self._re = _compile(self.params["pattern"], self.params["flags"])

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        m = (self._re.fullmatch if self.params["mode"] == "fullmatch" else self._re.search)(
            inp.output
        )
        detail: dict[str, Any] = {"mode": self.params["mode"]}
        if m is not None:
            detail |= {"span": [m.start(), m.end()], "text": m.group(0)[:_EVIDENCE_CHARS]}
        return EvalOutcome({"match": 1.0 if m else 0.0}, "PASS" if m else "FAIL", detail)


# ---- json_schema ----------------------------------------------------------------------------


def _load_jsonschema() -> Any:
    try:
        import jsonschema
    except ImportError:
        raise ConfigError(
            "the json_schema evaluator needs the optional dependency: "
            "pip install 'evalkit[jsonschema]'"
        ) from None
    return jsonschema


def _local_refs_only(node: Any, path: str = "$") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if (
                k in ("$ref", "$dynamicRef", "$recursiveRef")
                and isinstance(v, str)
                and not v.startswith("#")
            ):
                raise ConfigError(
                    f"json_schema: remote reference {v!r} at {path} is not allowed "
                    "(remote $ref is disabled; inline the definition)"
                )
            _local_refs_only(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _local_refs_only(v, f"{path}[{i}]")


def _validator(schema: dict[str, Any]) -> Any:
    jsonschema = _load_jsonschema()
    from referencing import Registry
    from referencing.exceptions import NoSuchResource

    def refuse(uri: str) -> Any:
        raise NoSuchResource(ref=uri)  # nothing is ever fetched: no network, no file access

    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, registry=Registry(retrieve=refuse))


def _schema_canonical(p: dict[str, Any]) -> dict[str, Any]:
    _only(p, {"schema", "strip_code_fence"}, "json_schema")
    schema = p.get("schema")
    if not isinstance(schema, dict):
        raise ConfigError("json_schema.schema must be a JSON Schema object")
    if len(safejson_dumps(schema)) > MAX_SCHEMA_BYTES:
        raise ConfigError(f"json_schema.schema is larger than {MAX_SCHEMA_BYTES} bytes")
    _local_refs_only(schema)
    jsonschema = _load_jsonschema()
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.SchemaError as e:
        raise ConfigError(f"json_schema.schema is not a valid JSON Schema: {e.message}") from e
    fence = p.get("strip_code_fence", False)
    if not isinstance(fence, bool):
        raise ConfigError("json_schema.strip_code_fence must be true or false")
    return {"schema": schema, "strip_code_fence": fence}


def safejson_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


class JsonSchemaCheck(_Evaluator):
    """Structured-output validity of the *target's* output. Malformed or non-conforming output is
    the AI system's quality (`valid = 0`), not an evaluator failure."""

    def __init__(self, spec: EvaluatorSpec):
        super().__init__(spec)
        self._validator = _validator(self.params["schema"])

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        text = inp.output
        if self.params["strip_code_fence"] and (m := _FENCE.match(text)):
            text = m.group(1)
        try:
            instance = safejson.loads(text)
        except (ValueError, RecursionError) as e:
            return EvalOutcome(
                {"valid": 0.0}, "FAIL", {"parse_error": str(e)[:_EVIDENCE_CHARS], "parsed": False}
            )
        try:
            errors = sorted(
                self._validator.iter_errors(instance), key=lambda e: list(map(str, e.path))
            )
        except RecursionError:
            return EvalOutcome(
                {"valid": 0.0},
                "FAIL",
                {"parsed": True, "error_count": None, "errors": ["too deeply nested"]},
            )
        if not errors:
            return EvalOutcome({"valid": 1.0}, "PASS", {"parsed": True})
        shown = [
            {
                "path": "/" + "/".join(map(str, e.absolute_path)),
                "message": e.message[:_EVIDENCE_CHARS],
            }
            for e in errors[:_MAX_SCHEMA_ERRORS]
        ]
        return EvalOutcome(
            {"valid": 0.0}, "FAIL", {"parsed": True, "error_count": len(errors), "errors": shown}
        )


# ---- retrieval ------------------------------------------------------------------------------


def _ks(p: dict[str, Any], kind: str) -> list[int]:
    ks = p.get("k", [5])
    if isinstance(ks, int) and not isinstance(ks, bool):
        ks = [ks]
    if (
        not isinstance(ks, list)
        or not ks
        or any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K for k in ks)
    ):
        raise ConfigError(f"{kind}.k must be an integer or list of integers in 1..{MAX_K}")
    return sorted(set(ks))


def _retrieval_canonical(p: dict[str, Any]) -> dict[str, Any]:
    _only(p, {"k"}, "retrieval")
    return {"k": _ks(p, "retrieval")}


def retrieval_metrics(
    retrieved: Sequence[str], relevance: dict[str, int], ks: Sequence[int]
) -> dict[str, float]:
    """The metric semantics of design 6.2 (the single specification the tests check).

    relevant = {d : relevance[d] > 0}; the ranking is `retrieved` de-duplicated (first occurrence).
    recall@k = |relevant & top_k| / |relevant|; precision@k = |relevant & top_k| / k (k is the
    denominator even when fewer were retrieved); hit@k = 1 if any relevant in top k; mrr =
    1/rank(first relevant) else 0; ndcg@k = DCG@k / IDCG@k with gain 2**rel - 1 and discount
    log2(rank + 1), IDCG from *all* labelled documents (so relevant documents that were not
    retrieved lower the score). Unlabelled retrieved documents count as not relevant.
    Raises NotApplicable when there are no relevant documents (the metrics are undefined).
    """
    relevant = {d for d, g in relevance.items() if g > 0}
    if not relevant:
        raise NotApplicable("no relevant documents are labelled for this case")
    ranked = list(dict.fromkeys(retrieved))
    ideal = sorted((g for g in relevance.values() if g > 0), reverse=True)
    metrics: dict[str, float] = {}
    first = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
    metrics["mrr"] = 0.0 if first is None else 1.0 / first
    for k in ks:
        top = ranked[:k]
        hits = sum(1 for d in top if d in relevant)
        dcg = sum(
            (2.0 ** relevance.get(d, 0) - 1.0) / math.log2(i + 1) for i, d in enumerate(top, 1)
        )
        idcg = sum((2.0**g - 1.0) / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))
        metrics[f"recall@{k}"] = hits / len(relevant)
        metrics[f"precision@{k}"] = hits / k
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"ndcg@{k}"] = dcg / idcg
    return metrics


class RetrievalEval(_Evaluator):
    requires = frozenset({"relevance", "retrieved"})

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        if inp.relevance is None or inp.retrieved is None:
            raise EvalFailure(
                FailureClass.INPUT,
                "missing_field",
                "retrieval needs `relevance` labels and retrieved ids",
            )
        metrics = retrieval_metrics(inp.retrieved, inp.relevance, self.params["k"])
        detail = {
            "retrieved": len(inp.retrieved),
            "relevant": sum(1 for g in inp.relevance.values() if g > 0),
            "k": self.params["k"],
        }
        return EvalOutcome(metrics, None, detail)


# ---- citation_check -------------------------------------------------------------------------

_DEFAULT_MARKER = r"\[([^\[\]\s,]+)\]"


def _citation_canonical(p: dict[str, Any]) -> dict[str, Any]:
    _only(p, {"style", "marker"}, "citation_check")
    style = p.get("style", "index")
    if style not in ("index", "id"):
        raise ConfigError("citation_check.style must be 'index' (marker n = nth retrieved) or 'id'")
    marker = p.get("marker", _DEFAULT_MARKER)
    if not isinstance(marker, str) or not 0 < len(marker) <= MAX_PATTERN_CHARS:
        raise ConfigError(f"citation_check.marker must be 1-{MAX_PATTERN_CHARS} characters")
    try:
        compiled = re.compile(marker)
    except re.error as e:
        raise ConfigError(f"citation_check.marker is not a valid regular expression: {e}") from e
    if compiled.groups != 1:
        raise ConfigError("citation_check.marker must have exactly one capture group")
    return {"style": style, "marker": marker}


class CitationCheck(_Evaluator):
    """Deterministic *structure* check: does each citation marker in the output resolve to a
    retrieved document, and are the cited documents the relevant ones. It does NOT judge whether a
    cited document supports the claim -- that is faithfulness, which needs a judge."""

    requires = frozenset({"retrieved"})

    def __init__(self, spec: EvaluatorSpec):
        super().__init__(spec)
        self._re = re.compile(self.params["marker"])

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        if inp.retrieved is None:
            raise EvalFailure(
                FailureClass.INPUT, "missing_field", "citation_check needs retrieved ids"
            )
        tokens = list(dict.fromkeys(self._re.findall(inp.output)))
        if not tokens:
            raise NotApplicable("the output contains no citations")
        retrieved = list(dict.fromkeys(inp.retrieved))
        resolved: dict[str, str] = {}
        for t in tokens:
            if self.params["style"] == "index":
                if t.isascii() and t.isdigit() and 1 <= int(t) <= len(retrieved):
                    resolved[t] = retrieved[int(t) - 1]
            elif t in retrieved:
                resolved[t] = t
        cited = set(resolved.values())
        metrics = {"citation_validity": len(resolved) / len(tokens)}
        detail: dict[str, Any] = {
            "cited": tokens,
            "unresolved": [t for t in tokens if t not in resolved],
            "documents": sorted(cited),
        }
        if inp.relevance is not None:
            relevant = {d for d, g in inp.relevance.items() if g > 0}
            if cited:
                metrics["citation_precision"] = len(cited & relevant) / len(cited)
            if relevant:
                metrics["citation_recall"] = len(cited & relevant) / len(relevant)
        verdict = "PASS" if len(resolved) == len(tokens) else "FAIL"
        return EvalOutcome(metrics, verdict, detail)


# ---- llm_judge ------------------------------------------------------------------------------

_FEEDBACK = (
    "\n\nYour previous evaluation was rejected: {error}\n"
    "Submit a corrected evaluation that satisfies the requirements exactly."
)


def _judge_canonical(p: dict[str, Any]) -> dict[str, Any]:
    # the derived keys are accepted (a resolved spec is resolved again) but always recomputed, so a
    # spec made under an older judge prompt no longer matches its key and is refused by build()
    derived = {"rubric_content_hash", "prompt_version", "scoring_version"}
    _only(p, {"rubric", "provider", "model", "temperature", "max_tokens"} | derived, "llm_judge")
    try:
        rubric = Rubric.model_validate(p.get("rubric"))
    except Exception as e:  # noqa: BLE001 - any invalid rubric is a config error
        raise ConfigError(f"llm_judge.rubric is invalid: {e}") from e
    for key in ("provider", "model"):
        if not isinstance(p.get(key), str) or not p[key]:
            raise ConfigError(f"llm_judge.{key} is required (it names the judge in the identity)")
    temperature = p.get("temperature", 0.0)
    max_tokens = p.get("max_tokens", MAX_TOKENS)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not 0 <= temperature <= 2
    ):
        raise ConfigError("llm_judge.temperature must be a number in [0, 2]")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 100_000
    ):
        raise ConfigError("llm_judge.max_tokens must be an integer in 1..100000")
    assert rubric.version is not None
    return {
        "rubric": rubric.model_dump(mode="json"),
        "rubric_content_hash": rubric.content_hash,  # derived: what the rubric *is*
        "provider": p["provider"],
        "model": p["model"],
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "prompt_version": PROMPT_VERSION,  # derived: what the judge is *asked*
        "scoring_version": SCORING_VERSION,
    }


def llm_judge_spec(
    name: str,
    rubric: Rubric | dict[str, Any],
    client: LLMClient,
    *,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> EvaluatorSpec:
    """A resolved `llm_judge` spec naming the client's provider and model."""
    rubric = rubric if isinstance(rubric, Rubric) else Rubric.model_validate(rubric)
    spec = EvaluatorSpec(
        kind="llm_judge",
        name=name,
        params={
            "rubric": rubric.model_dump(mode="json"),
            "provider": client.provider,
            "model": client.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    return resolve(spec)


class LLMJudgeEvaluator(_Evaluator):
    """Owns judge semantics (prompt, `Rubric.score`, must-pass gates, failure attribution); the
    `LLMClient` owns the request. The judge sees prompt, output, reference and context only."""

    def __init__(self, spec: EvaluatorSpec, client: LLMClient):
        super().__init__(spec)
        self.client = client
        self.rubric = Rubric.model_validate(self.params["rubric"])

    def evaluate(self, inp: EvalInput, call: UnitCalls) -> EvalOutcome:
        p = self.params
        base = build_request(
            inp.prompt, inp.output, inp.reference, inp.context, self.rubric,
            temperature=p["temperature"], timeout_s=call.timeout_s, max_tokens=p["max_tokens"],
        )  # fmt: skip

        def request(feedback: str | None) -> Any:
            return (
                base
                if feedback is None
                else replace(base, user=base.user + _FEEDBACK.format(error=feedback))
            )

        def send(feedback: str | None) -> Any:
            return self.client.call(request(feedback))

        def validate(resp: Any) -> Any:
            payload = judge_payload(resp)
            try:
                return self.rubric.score(payload)
            except JudgeOutputError as e:
                raise EvalFailure(
                    FailureClass.EVALUATOR, "invalid_output", str(e), raw=payload
                ) from e
            except ScoringError as e:
                raise EvalFailure(
                    FailureClass.EVALUATOR, "internal_error", str(e), raw=payload
                ) from e

        scores, overall, verdict = call.run(
            send,
            validate=validate,
            describe=describe_llm_response,
            provider=self.client.provider,
            model=self.client.model,
            feedback_retry=True,
            request=request,
            endpoint=getattr(self.client, "endpoint", None),
        )
        metrics = {"score": overall}
        for c in self.rubric.criteria:
            metrics[f"criterion.{c.name}"] = c.normalized(scores[c.name])
        detail = {
            "reasoning": {n: s.reasoning for n, s in scores.items()},
            "values": {n: s.label if s.label is not None else s.score for n, s in scores.items()},
            "threshold": self.rubric.threshold,
            "failed_gates": self.rubric.failed_gates(scores),
            "rubric_version": self.rubric.version,
            "prompt_version": PROMPT_VERSION,
        }
        return EvalOutcome(metrics, verdict, detail)

    requires = frozenset()


def _judge_build(spec: EvaluatorSpec, ctx: EvalContext) -> CaseEvaluator:
    p = spec.params
    matches = [c for c in ctx.clients if c.provider == p["provider"] and c.model == p["model"]]
    if not matches:
        raise ConfigError(
            f"no LLM client for provider {p['provider']!r} model {p['model']!r} was supplied "
            f"for evaluator {spec.name!r}"
        )
    return LLMJudgeEvaluator(spec, matches[0])


register(
    EvaluatorKind(
        "exact_match", 1, ExactMatch.requires, _exact_canonical, lambda s, c: ExactMatch(s)
    )
)
register(
    EvaluatorKind("regex", 1, RegexMatch.requires, _regex_canonical, lambda s, c: RegexMatch(s))
)
register(
    EvaluatorKind(
        "json_schema",
        1,
        JsonSchemaCheck.requires,
        _schema_canonical,
        lambda s, c: JsonSchemaCheck(s),
    )
)
register(
    EvaluatorKind(
        "retrieval", 1, RetrievalEval.requires, _retrieval_canonical, lambda s, c: RetrievalEval(s)
    )
)
register(
    EvaluatorKind(
        "citation_check",
        1,
        CitationCheck.requires,
        _citation_canonical,
        lambda s, c: CitationCheck(s),
    )
)
register(EvaluatorKind("llm_judge", 1, LLMJudgeEvaluator.requires, _judge_canonical, _judge_build))
