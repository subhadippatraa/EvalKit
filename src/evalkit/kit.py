"""`EvalKit`: the entry point to the dataset platform (docs/TARGET-ARCHITECTURE.md §13.1).

    kit = EvalKit.open("evalkit.db")
    result = kit.datasets.import_jsonl("support-qa", "cases.jsonl")
    for case in kit.datasets.cases("support-qa@latest"): ...

`kit.runs` records runs and their results (no execution yet); comparison and reports will
hang off the same object in later phases. The single-record
`Evaluator` API is separate and unchanged.
"""

from __future__ import annotations

from pathlib import Path

from evalkit.calibration import ReviewService
from evalkit.datasets import DatasetService
from evalkit.engine import RunController
from evalkit.limits import Limits
from evalkit.runs import RunService
from evalkit.store import SQLiteStore, default_db_path


class EvalKit:
    def __init__(self, store: SQLiteStore, limits: Limits | None = None):
        self.store = store
        self.limits = limits or Limits()
        self.datasets = DatasetService(store, self.limits)
        self.runs = RunService(store, self.datasets, self.limits)
        self.controller = RunController(self)
        self.reviews = ReviewService(self)

    @classmethod
    def open(cls, path: str | Path, limits: Limits | None = None) -> EvalKit:
        """Open (and migrate, if needed) the database at `path`."""
        return cls(SQLiteStore(path), limits)

    @classmethod
    def from_env(cls, limits: Limits | None = None) -> EvalKit:
        """The database named by EVALKIT_DB_PATH (default ./evalkit.db)."""
        return cls.open(default_db_path(), limits)

    def snapshot(self, run_id: str) -> str:
        """Store an immutable summary snapshot of the run (recomputable from its rows)."""
        from evalkit.analysis import take_snapshot

        return take_snapshot(self, run_id)

    def summarize(self, run_id: str, **kwargs):
        """Aggregate a run now (see `evalkit.analysis.summarize`)."""
        from evalkit.analysis import summarize

        return summarize(self, run_id, **kwargs)

    def close(self) -> None:
        self.store.close()
