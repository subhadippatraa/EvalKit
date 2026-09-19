class EvalKitError(Exception):
    """Base class for all evalkit errors."""


class ConfigError(EvalKitError):
    """Missing or invalid configuration (env vars, no judge/store configured)."""


class RubricError(EvalKitError):
    """Invalid rubric or evaluate() inputs. Raised before the judge is called; never persisted."""


class JudgeError(EvalKitError):
    """Judge provider/API/connection failure. Persisted as an error row, then raised."""

    def __init__(self, message: str, evaluation_id: str | None = None):
        super().__init__(message)
        self.evaluation_id = evaluation_id


class JudgeTimeoutError(JudgeError):
    """The judge request timed out."""


class JudgeOutputError(JudgeError):
    """Judge output was missing, malformed, or out of range."""
