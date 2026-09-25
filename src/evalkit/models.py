import hashlib
import json
import math
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalkit.errors import JudgeOutputError, ScoringError
from evalkit.limits import (
    MAX_CRITERIA,
    MAX_DESCRIPTION_CHARS,
    MAX_LABEL_CHARS,
    MAX_LABELS,
    MAX_REVIEW_COMMENT_CHARS,
    MAX_REVIEWER_CHARS,
    MAX_SCALE_ABS,
    MAX_VERSION_CHARS,
    MAX_WEIGHT,
    MIN_WEIGHT,
)

Verdict = Literal["PASS", "FAIL"]

# The judge prompt (evalkit.judge) delimits untrusted content with these markers. Trusted rubric
# text must not contain them, so it can never be mistaken for (or forge) a delimiter line.
DELIMITER_OPEN = "<<<EVALKIT:"
DELIMITER_CLOSE = "<<<END:"


def _no_delimiters(text: str, what: str) -> str:
    try:
        text.encode("utf-8")  # unpaired surrogates would fail later, at persist/send time
    except UnicodeEncodeError as e:
        raise ValueError(f"{what} is not valid Unicode text (unpaired surrogate)") from e
    if DELIMITER_OPEN in text or DELIMITER_CLOSE in text:
        raise ValueError(f"{what} must not contain {DELIMITER_OPEN!r} or {DELIMITER_CLOSE!r}")
    return text


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


def format_validation_error(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'input'}: {err['msg']}" for err in e.errors()
    )


# each criterion name becomes a tool-schema property name (evalkit.judge.output_schema);
# keep it to a safe, boring identifier so it's valid across judge providers
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Criterion(BaseModel):
    name: str = Field(min_length=1, pattern=_NAME_RE.pattern)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    # Exactly one of scale/labels ends up set (see _check_scale_or_labels). Both default to
    # None so the validator can tell "neither given" (-> default numeric scale, the pre-labels
    # behavior) apart from "both given" (-> error), which isn't possible if `scale` keeps a
    # non-None default.
    scale: tuple[int, int] | None = None
    labels: tuple[str, ...] | None = None
    # bounded + finite: keeps Rubric.score's weighted mean exactly representable (no inf/inf)
    weight: float = Field(default=1.0, ge=MIN_WEIGHT, le=MAX_WEIGHT, allow_inf_nan=False)

    @field_validator("description")
    @classmethod
    def _description_safe(cls, v: str) -> str:
        return _no_delimiters(v, "criterion description")

    @model_validator(mode="after")
    def _check_scale_or_labels(self) -> "Criterion":
        if self.scale is not None and self.labels is not None:
            raise ValueError("a criterion cannot set both `scale` and `labels`")
        if self.scale is None and self.labels is None:
            self.scale = (1, 5)  # unchanged default from before `labels` existed
        if self.scale is not None:
            if self.scale[0] >= self.scale[1]:
                raise ValueError(f"scale min must be < max, got {self.scale}")
            if max(abs(self.scale[0]), abs(self.scale[1])) > MAX_SCALE_ABS:
                raise ValueError(
                    f"scale bounds must be within +/-{MAX_SCALE_ABS}, got {self.scale}"
                )
        if self.labels is not None:
            if len(self.labels) < 2:
                raise ValueError(f"labels must have at least 2 entries, got {self.labels}")
            if len(self.labels) > MAX_LABELS:
                raise ValueError(f"at most {MAX_LABELS} labels allowed, got {len(self.labels)}")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"labels must be unique, got {self.labels}")
            for label in self.labels:
                # labels are rendered into the trusted criteria section: keep them one boring line
                if not label or len(label) > MAX_LABEL_CHARS or not label.isprintable():
                    raise ValueError(
                        f"labels must be 1-{MAX_LABEL_CHARS} printable characters, got {label!r}"
                    )
                _no_delimiters(label, "label")
        return self


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(min_length=1)
    # Exactly one of score/label is set -- whichever the matching Criterion requires (checked
    # against the specific criterion in Rubric.score, since a bare CriterionScore doesn't know
    # which Criterion it belongs to).
    score: int | None = None
    label: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one(self) -> "CriterionScore":
        if (self.score is None) == (self.label is None):
            raise ValueError("exactly one of `score` or `label` must be set")
        return self


class Rubric(BaseModel):
    criteria: list[Criterion] = Field(min_length=1, max_length=MAX_CRITERIA)
    threshold: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    version: str | None = Field(default=None, min_length=1, max_length=MAX_VERSION_CHARS)

    @field_validator("version")
    @classmethod
    def _version_is_one_printable_line(cls, v: str | None) -> str | None:
        if v is not None:
            if not v.isprintable():
                raise ValueError("version must be printable, single-line text")
            _no_delimiters(v, "version")
        return v

    @model_validator(mode="after")
    def _check_and_version(self) -> "Rubric":
        names = [c.name for c in self.criteria]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate criterion names: {dupes}")
        if self.version is None:
            self.version = self.content_hash[:12]
        return self

    @property
    def content_hash(self) -> str:
        """sha256 of the rubric's canonical content (everything except `version`).

        The store records it per `version` so one version label can never silently refer to
        two different rubrics. The auto version is its first 12 hex characters.
        """
        canonical = json.dumps(self.model_dump(exclude={"version"}), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def from_dict(cls, criteria: dict[str, str]) -> "Rubric":
        """Default rubric (scale 1-5, weight 1.0, threshold 0.75) from {name: description}."""
        return cls(criteria=[Criterion(name=n, description=d) for n, d in criteria.items()])

    def score(self, raw: Any) -> tuple[dict[str, CriterionScore], float, Verdict]:
        """Validate raw judge output strictly and compute (scores, overall 0-1, verdict)."""
        if not isinstance(raw, dict):
            raise JudgeOutputError(f"judge output must be an object, got {type(raw).__name__}")
        expected = {c.name for c in self.criteria}
        missing, extra = expected - raw.keys(), raw.keys() - expected
        if missing or extra:
            raise JudgeOutputError(
                f"judge output criteria mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(map(str, extra))}"
            )

        scores: dict[str, CriterionScore] = {}
        contributions: list[float] = []
        for c in self.criteria:
            try:
                # strict: no coercion of "4", 4.0 or True into an int score
                cs = CriterionScore.model_validate(raw[c.name], strict=True)
            except ValidationError as e:
                raise JudgeOutputError(
                    f"invalid judge output for {c.name!r}: {format_validation_error(e)}"
                ) from e
            if c.labels is not None:
                if cs.label is None:
                    raise JudgeOutputError(f"{c.name!r} expects a label, got a numeric score")
                if cs.label not in c.labels:
                    raise JudgeOutputError(
                        f"label {cs.label!r} for {c.name!r} is not one of {list(c.labels)}"
                    )
                # ordinal position in the declared (worst -> best) order, same normalization
                # shape as a numeric scale: (value - min) / (max - min)
                normalized = c.labels.index(cs.label) / (len(c.labels) - 1)
            else:
                if cs.score is None:
                    raise JudgeOutputError(f"{c.name!r} expects a numeric score, got a label")
                assert c.scale is not None  # guaranteed by _check_scale_or_labels
                lo, hi = c.scale
                if not lo <= cs.score <= hi:
                    raise JudgeOutputError(
                        f"score {cs.score} for {c.name!r} is outside scale {lo}-{hi}"
                    )
                normalized = (cs.score - lo) / (hi - lo)
            scores[c.name] = cs
            contributions.append(c.weight * normalized)

        # fsum: correctly-rounded sums, so the mean does not depend on criterion order.
        # Rounding keeps exact thresholds (e.g. all 4s on 1-5 == 0.75) free of float noise.
        overall = round(math.fsum(contributions) / math.fsum(c.weight for c in self.criteria), 10)
        # unreachable with bounded finite weights and validated scores; kept so a future change
        # can never turn NaN/inf/out-of-range into an "ok" result (audit F-1)
        if not math.isfinite(overall) or not 0.0 <= overall <= 1.0:
            raise ScoringError(f"computed overall score is not a finite value in [0, 1]: {overall}")
        return scores, overall, "PASS" if overall >= self.threshold else "FAIL"


class Review(BaseModel):
    id: str = Field(default_factory=_new_id)
    evaluation_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1, max_length=MAX_REVIEWER_CHARS)
    verdict: Verdict
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    comment: str | None = Field(default=None, max_length=MAX_REVIEW_COMMENT_CHARS)
    created_at: datetime = Field(default_factory=_now)


AttemptOutcome = Literal[
    "ok", "invalid_output", "timeout", "provider_error", "scoring_error", "unexpected_error"
]


class Attempt(BaseModel):
    """One judge call and what came of it. Every call is recorded, so a result that succeeded
    on its retry is distinguishable from one that succeeded first time, and a rejected output
    is kept for debugging (failed attempts only; truncated -- see evalkit.limits)."""

    n: int = Field(ge=1)
    outcome: AttemptOutcome
    started_at: datetime
    duration_ms: int = Field(ge=0)
    error_type: str | None = None  # exception class name
    error: str | None = None  # scrubbed and truncated
    kind: str | None = None  # provider-detected sub-case, see JudgeError
    raw: str | None = None  # rejected/partial judge output as text (failed attempts only)
    raw_sha256: str | None = None  # of the full, untruncated text
    raw_truncated: bool = False


class EvaluationResult(BaseModel):
    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    status: Literal["ok", "error"]
    error: str | None = None
    prompt: str
    model_output: str
    reference_output: str | None = None
    context: str | None = None
    rubric: Rubric
    rubric_version: str
    judge_provider: str
    judge_model: str
    judge_temperature: float = Field(ge=0, allow_inf_nan=False)
    judge_prompt_version: str
    scores: dict[str, CriterionScore] = Field(default_factory=dict)
    overall_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    verdict: Verdict | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    # empty for rows written before attempt evidence existed
    attempts: list[Attempt] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)

    @field_validator("metadata")
    @classmethod
    def _json_serializable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metadata must be JSON-serializable (no NaN/Infinity): {e}") from e
        return v

    @model_validator(mode="after")
    def _coherent(self) -> "EvaluationResult":
        """An "ok" result carries a score and a verdict that follow from its rubric; an "error"
        result carries neither. Nothing else can be persisted or returned (audit F-1)."""
        if self.rubric_version != self.rubric.version:
            raise ValueError("rubric_version does not match rubric.version")
        if self.status == "error":
            if self.error is None:
                raise ValueError("an error result needs an error message")
            if self.scores or self.overall_score is not None or self.verdict is not None:
                raise ValueError("an error result must not carry scores, overall_score or verdict")
            return self
        if self.error is not None:
            raise ValueError("an ok result must not carry an error")
        if self.overall_score is None or self.verdict is None:
            raise ValueError("an ok result needs overall_score and verdict")
        if set(self.scores) != {c.name for c in self.rubric.criteria}:
            raise ValueError("an ok result needs exactly one score per rubric criterion")
        expected: Verdict = "PASS" if self.overall_score >= self.rubric.threshold else "FAIL"
        if self.verdict != expected:
            raise ValueError(
                f"verdict {self.verdict} contradicts overall_score {self.overall_score} "
                f"and threshold {self.rubric.threshold}"
            )
        return self

    @field_validator("tags")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))
