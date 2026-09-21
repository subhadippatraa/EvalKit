import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalkit.errors import JudgeOutputError

Verdict = Literal["PASS", "FAIL"]


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
    description: str = Field(min_length=1)
    # Exactly one of scale/labels ends up set (see _check_scale_or_labels). Both default to
    # None so the validator can tell "neither given" (-> default numeric scale, the pre-labels
    # behavior) apart from "both given" (-> error), which isn't possible if `scale` keeps a
    # non-None default.
    scale: tuple[int, int] | None = None
    labels: tuple[str, ...] | None = None
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _check_scale_or_labels(self) -> "Criterion":
        if self.scale is not None and self.labels is not None:
            raise ValueError("a criterion cannot set both `scale` and `labels`")
        if self.scale is None and self.labels is None:
            self.scale = (1, 5)  # unchanged default from before `labels` existed
        if self.scale is not None and self.scale[0] >= self.scale[1]:
            raise ValueError(f"scale min must be < max, got {self.scale}")
        if self.labels is not None:
            if len(self.labels) < 2:
                raise ValueError(f"labels must have at least 2 entries, got {self.labels}")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"labels must be unique, got {self.labels}")
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
    criteria: list[Criterion] = Field(min_length=1)
    threshold: float = Field(default=0.75, ge=0, le=1)
    version: str | None = None

    @model_validator(mode="after")
    def _check_and_version(self) -> "Rubric":
        names = [c.name for c in self.criteria]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate criterion names: {dupes}")
        if self.version is None:
            canonical = json.dumps(self.model_dump(exclude={"version"}), sort_keys=True)
            self.version = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return self

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
        weighted_sum = 0.0
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
            weighted_sum += c.weight * normalized

        # rounding keeps exact thresholds (e.g. all 4s on 1-5 == 0.75) free of float noise
        overall = round(weighted_sum / sum(c.weight for c in self.criteria), 10)
        return scores, overall, "PASS" if overall >= self.threshold else "FAIL"


class Review(BaseModel):
    id: str = Field(default_factory=_new_id)
    evaluation_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    verdict: Verdict
    score: float | None = Field(default=None, ge=0, le=1)
    comment: str | None = None
    created_at: datetime = Field(default_factory=_now)


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
    judge_temperature: float
    judge_prompt_version: str
    scores: dict[str, CriterionScore] = Field(default_factory=dict)
    overall_score: float | None = None
    verdict: Verdict | None = None
    latency_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    reviews: list[Review] = Field(default_factory=list)

    @field_validator("metadata")
    @classmethod
    def _json_serializable(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"metadata must be JSON-serializable: {e}") from e
        return v

    @field_validator("tags")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))
