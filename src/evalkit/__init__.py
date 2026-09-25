"""evalkit: a small LLM-as-judge evaluation library."""

from evalkit.datasets import (
    Dataset,
    DatasetService,
    DatasetVersion,
    EvaluationCase,
    ImportResult,
)
from evalkit.errors import (
    ConfigError,
    DatasetError,
    DuplicateResultError,
    EvalKitError,
    JudgeError,
    JudgeOutputError,
    JudgeTimeoutError,
    MigrationError,
    RubricError,
    RunError,
    ScoringError,
    StoreError,
)
from evalkit.evaluator import Evaluator
from evalkit.failures import EvalFailure, Failure, FailureClass
from evalkit.judge import Judge
from evalkit.kit import EvalKit
from evalkit.limits import Limits
from evalkit.models import Attempt, Criterion, EvaluationResult, Review, Rubric
from evalkit.runs import (
    CaseOutcome,
    CaseResult,
    EvaluatorOutcome,
    EvaluatorResult,
    EvaluatorSpec,
    Run,
    RunAttempt,
    RunConfig,
    RunService,
    TargetSpec,
)

__all__ = [
    "Attempt",
    "CaseOutcome",
    "CaseResult",
    "ConfigError",
    "Criterion",
    "Dataset",
    "DatasetError",
    "DatasetService",
    "DatasetVersion",
    "DuplicateResultError",
    "EvalFailure",
    "EvalKit",
    "EvalKitError",
    "EvaluationCase",
    "EvaluationResult",
    "Evaluator",
    "EvaluatorOutcome",
    "EvaluatorResult",
    "EvaluatorSpec",
    "Failure",
    "FailureClass",
    "ImportResult",
    "Judge",
    "JudgeError",
    "JudgeOutputError",
    "JudgeTimeoutError",
    "Limits",
    "MigrationError",
    "Review",
    "Rubric",
    "RubricError",
    "Run",
    "RunAttempt",
    "RunConfig",
    "RunError",
    "RunService",
    "ScoringError",
    "StoreError",
    "TargetSpec",
]
