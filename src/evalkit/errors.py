import sqlite3
from collections.abc import Sequence
from typing import Any

from evalkit.redact import scrub


class EvalKitError(Exception):
    """Base class for all evalkit errors."""


class ConfigError(EvalKitError):
    """Missing or invalid configuration (env vars, no judge/store configured)."""


class RubricError(EvalKitError):
    """Invalid rubric or evaluate() inputs. Raised before the judge is called; never persisted."""


class MigrationError(EvalKitError):
    """The database schema is unrecognized, newer than this evalkit, or a migration failed."""


class DatasetError(EvalKitError):
    """A dataset could not be imported, found, read or verified.

    `issues` lists every validation problem found (each with a line/index and case_key where
    known) so a bad file can be fixed in one pass; nothing is persisted when there are issues.
    """

    def __init__(self, message: str, issues: Sequence[Any] = ()):
        super().__init__(scrub(message))
        self.issues = list(issues)


class RunError(EvalKitError):
    """A run or one of its results could not be created, changed, recorded or read.

    Raised for anything the run domain forbids: a case from another dataset version, a second
    result for the same case, a result on a run that is not running, an illegal status change.
    """


class DuplicateResultError(RunError):
    """A result already exists for this case (or evaluator) in this run; results are write-once."""


class JudgeError(EvalKitError):
    """Judge provider/API/connection failure. Persisted as an error row, then raised.

    `kind` (optional) names a provider-detected sub-case ("truncated", "refused",
    "malformed_output", "context_window", "no_tool_call", "invalid_json", ...) and `raw` any
    partial provider output worth keeping as evidence; both are recorded on the attempt.
    The message is scrubbed of secrets/ARNs at construction.
    """

    def __init__(
        self,
        message: str,
        evaluation_id: str | None = None,
        *,
        kind: str | None = None,
        raw: Any = None,
        failure: Any = None,
    ):
        super().__init__(scrub(message))
        self.evaluation_id = evaluation_id
        self.kind = kind
        self.raw = raw
        self.failure = failure  # the classified `evalkit.failures.Failure`, when there is one


class JudgeTimeoutError(JudgeError):
    """The judge request timed out."""


class JudgeOutputError(JudgeError):
    """Judge output was missing, malformed, or out of range."""


class ScoringError(JudgeError):
    """Internal inconsistency while scoring/assembling a result (never a judge fault).

    Persisted as an error row like any judge-stage failure, but never retried.
    """


# also an sqlite3.Error: before P0 a failed save propagated the raw sqlite3.Error, and callers
# may still `except sqlite3.Error`
class StoreError(EvalKitError, sqlite3.Error):
    """A judged result could not be persisted.

    `result` is the complete EvaluationResult (the judge call was already paid for);
    `spill_path` is where it was written for later recovery, if the store supports that.
    """

    def __init__(self, message: str, *, result: Any = None, spill_path: Any = None):
        super().__init__(scrub(message))
        self.result = result
        self.spill_path = spill_path
        self.evaluation_id: str | None = getattr(result, "id", None)
