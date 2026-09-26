"""Datasets: versioned, immutable collections of evaluation cases (docs/TARGET-ARCHITECTURE.md §3).

* A `Dataset` is a named lineage. Importing cases creates an immutable, content-addressed
  `DatasetVersion`; importing identical content again returns the existing version.
* An `EvaluationCase` is data only: everything any evaluator may need, no results. `case_key` is
  its stable identity across versions; `content_hash` detects change.
* Import is strict and atomic: every problem in the input is reported at once, and nothing is
  stored unless the whole input is valid. Content is stored losslessly.

This module is pure (models, validation, JSONL, hashing); persistence is in `dataset_store.py`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from evalkit import safejson
from evalkit.errors import DatasetError
from evalkit.hashing import CASE_DOMAIN, case_input_hash, stable_hash
from evalkit.limits import (
    MAX_DOC_ID_CHARS,
    MAX_DOC_IDS,
    MAX_ISSUES,
    MAX_JSON_DEPTH,
    MAX_JSONL_LINE_BYTES,
    MAX_RELEVANCE_GRADE,
    MAX_TAG_CHARS,
    Limits,
)
from evalkit.models import format_validation_error

_KEY_RE = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")  # no '@': it separates name@version
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8,64}$")


def _now() -> datetime:
    return datetime.now(UTC)


def _encodable(text: str, where: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"{where} is not valid Unicode text (unpaired surrogate)") from e


def _check_json(value: Any, where: str, depth: int = 0) -> None:
    """Metadata must be plain JSON: string keys, finite numbers, encodable text, bounded depth."""
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"{where} is nested deeper than {MAX_JSON_DEPTH} levels")
    if isinstance(value, str):
        _encodable(value, where)
    elif isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{where} has a non-string key {k!r}")
            _encodable(k, where)
            _check_json(v, f"{where}.{k}", depth + 1)
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            _check_json(v, f"{where}[{i}]", depth + 1)
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{where} contains a non-finite number")
    elif value is not None and not isinstance(value, bool | int):
        raise ValueError(f"{where} contains a non-JSON value of type {type(value).__name__}")


class EvaluationCase(BaseModel):
    """One case: an input plus whatever any evaluator may need. Unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")

    case_key: str = Field(pattern=_KEY_RE)
    prompt: str = Field(min_length=1)
    output: str | None = None  # a pre-generated output to score
    reference: str | None = None  # a gold answer
    context: str | None = None  # grounding material the output may rely on
    retrieved: list[str] | None = Field(default=None, max_length=MAX_DOC_IDS)  # ranked doc ids
    relevance: dict[str, int] | None = None  # doc id -> graded relevance (0 = not relevant)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        for tag in v:
            if not tag or len(tag) > MAX_TAG_CHARS:
                raise ValueError(f"tags must be 1-{MAX_TAG_CHARS} characters")
        return list(dict.fromkeys(v))  # de-duplicate, keep order

    @model_validator(mode="after")
    def _content(self) -> EvaluationCase:
        for name in ("prompt", "output", "reference", "context"):
            text = getattr(self, name)
            if text is not None:
                _encodable(text, name)
        for tag in self.tags:
            _encodable(tag, "tags")
        for doc_id in self.retrieved or []:
            if not doc_id or len(doc_id) > MAX_DOC_ID_CHARS:
                raise ValueError(f"retrieved ids must be 1-{MAX_DOC_ID_CHARS} characters")
            _encodable(doc_id, "retrieved")
        if self.relevance is not None:
            if len(self.relevance) > MAX_DOC_IDS:
                raise ValueError(f"relevance has more than {MAX_DOC_IDS} entries")
            for doc_id, grade in self.relevance.items():
                if not doc_id or len(doc_id) > MAX_DOC_ID_CHARS:
                    raise ValueError(f"relevance ids must be 1-{MAX_DOC_ID_CHARS} characters")
                _encodable(doc_id, "relevance")
                if not 0 <= grade <= MAX_RELEVANCE_GRADE:
                    raise ValueError(
                        f"relevance grade for {doc_id!r} must be 0-{MAX_RELEVANCE_GRADE}"
                    )
        _check_json(self.metadata, "metadata")
        try:  # what is hashed and stored is exactly the JSON form (tuples become lists, ...)
            self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        except (TypeError, ValueError) as e:  # e.g. an integer too large to render
            raise ValueError(f"metadata is not JSON-serializable: {e}") from e
        return self

    @property
    def content_hash(self) -> str:
        """Hash of the case's content, excluding `case_key` (identity) and including everything an
        evaluator could see. `None` and `""` differ; `tags` are order-insensitive, `retrieved`
        order (a ranking) is not."""
        return stable_hash(
            CASE_DOMAIN,
            {
                "prompt": self.prompt,
                "output": self.output,
                "reference": self.reference,
                "context": self.context,
                "retrieved": self.retrieved,
                "relevance": self.relevance,
                "metadata": self.metadata,
                "tags": sorted(self.tags),
            },
        )

    @property
    def input_hash(self) -> str:
        """Hash of the input side only (see `hashing.case_input_hash`): what comparison pairs on."""
        return case_input_hash(self.prompt, self.reference, self.context, self.relevance)

    def to_record(self) -> dict[str, Any]:
        """The JSONL form: only what is set (absent == None / empty), stable field order."""
        record: dict[str, Any] = {"case_key": self.case_key, "prompt": self.prompt}
        for name in ("output", "reference", "context", "retrieved", "relevance"):
            if getattr(self, name) is not None:
                record[name] = getattr(self, name)
        if self.metadata:
            record["metadata"] = self.metadata
        if self.tags:
            record["tags"] = self.tags
        return record


class StoredCase(EvaluationCase):
    """A case as stored in one dataset version."""

    id: str
    dataset_version_id: str
    stored_hash: str  # the content_hash recorded at import; `content_hash` recomputes it
    stored_input_hash: str | None = None  # the input_hash recorded at import (None: not recorded)


class Dataset(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: datetime
    version_count: int = 0
    latest_version: int | None = None


class DatasetVersion(BaseModel):
    """An immutable snapshot. `content_hash` identifies its exact content."""

    id: str
    dataset_id: str
    dataset_name: str
    version_no: int
    content_hash: str
    case_count: int
    source: str | None = None
    created_at: datetime

    @property
    def ref(self) -> str:
        return f"{self.dataset_name}@{self.version_no}"


@dataclass(frozen=True)
class ImportResult:
    dataset: Dataset
    version: DatasetVersion
    created: bool  # False: identical content already existed, the existing version is returned


@dataclass(frozen=True)
class ExportResult:
    path: Path
    case_count: int
    content_hash: str


@dataclass(frozen=True)
class LintFinding:
    code: str
    count: int
    message: str
    examples: list[str] = field(default_factory=list)  # case_keys


@dataclass
class VerifyReport:
    ok: bool
    case_count: int
    problems: list[str]


@dataclass(frozen=True)
class Issue:
    line: int | None  # 1-based line (JSONL) or index (iterable) of the offending case
    message: str
    case_key: str | None = None

    def __str__(self) -> str:
        where = "input" if self.line is None else f"line {self.line}"
        return f"{where}{f' (case {self.case_key!r})' if self.case_key else ''}: {self.message}"


class IssueCollector:
    """Gathers validation problems, capped at MAX_ISSUES so a hostile file cannot make us
    build an unbounded report (scanning stops once full)."""

    def __init__(self, cap: int = MAX_ISSUES):
        self.cap = cap
        self.issues: list[Issue] = []

    @property
    def full(self) -> bool:
        return len(self.issues) >= self.cap

    def add(self, line: int | None, message: str, case_key: str | None = None) -> None:
        if not self.full:
            self.issues.append(Issue(line, message, case_key))

    def raise_if_any(self, what: str) -> None:
        if self.issues:
            more = f" (stopped after the first {self.cap})" if self.full else ""
            lines = "\n  ".join(str(i) for i in self.issues[:10])
            extra = f"\n  ... and {len(self.issues) - 10} more" if len(self.issues) > 10 else ""
            raise DatasetError(
                f"{what}: {len(self.issues)} problem(s){more}; nothing was imported.\n  "
                f"{lines}{extra}",
                self.issues,
            )


def validate_dataset_name(name: Any) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise DatasetError(
            f"invalid dataset name {name!r}: use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return name


def check_case_limits(case: EvaluationCase, limits: Limits) -> None:
    """Byte-size limits (UTF-8), applied on top of the model's structural checks."""
    total = len(case.case_key)
    for name in ("prompt", "output", "reference", "context"):
        text = getattr(case, name)
        if text is None:
            continue
        size = len(text.encode("utf-8"))
        if size > limits.max_field_bytes:
            raise ValueError(
                f"`{name}` is too large ({size} bytes; limit {limits.max_field_bytes})"
            )
        total += size
    meta = len(json.dumps(case.metadata, ensure_ascii=False).encode("utf-8"))
    if meta > limits.max_metadata_bytes:
        raise ValueError(
            f"`metadata` is too large ({meta} bytes; limit {limits.max_metadata_bytes})"
        )
    if len(case.tags) > limits.max_tags:
        raise ValueError(f"too many tags ({len(case.tags)}; limit {limits.max_tags})")
    total += meta
    total += sum(len(d.encode("utf-8")) for d in case.retrieved or [])
    total += sum(len(d.encode("utf-8")) + 4 for d in (case.relevance or {}))
    total += sum(len(t.encode("utf-8")) for t in case.tags)
    if total > limits.max_case_bytes:
        raise ValueError(f"case is too large ({total} bytes; limit {limits.max_case_bytes})")


def validate_cases(
    items: Iterable[tuple[int, Any]], limits: Limits, collector: IssueCollector
) -> Iterator[EvaluationCase]:
    """Yield each valid case; record every problem (bad shape, unknown field, limit, ...) in
    `collector` and keep going so one pass reports them all. Stops once the collector is full or
    the case-count limit is exceeded."""
    count = 0
    for line, raw in items:
        if collector.full:
            return
        count += 1
        if count > limits.max_import_cases:
            collector.add(line, f"more than {limits.max_import_cases} cases")
            return
        if isinstance(raw, EvaluationCase):
            raw = raw.model_dump()
        if not isinstance(raw, Mapping):
            collector.add(line, f"expected a JSON object, got {type(raw).__name__}")
            continue
        key = raw.get("case_key") if isinstance(raw.get("case_key"), str) else None
        try:
            case = EvaluationCase.model_validate(dict(raw), strict=True)
            check_case_limits(case, limits)
        except ValidationError as e:
            collector.add(line, format_validation_error(e), key)
        except ValueError as e:
            collector.add(line, str(e), key)
        else:
            yield case


def read_jsonl(f: BinaryIO, collector: IssueCollector) -> Iterator[tuple[int, Any]]:
    """Stream (line_number, parsed_object) from a binary JSONL file without loading it whole.

    Strict: UTF-8 only, no NaN/Infinity, no duplicate keys, each line capped at
    MAX_JSONL_LINE_BYTES (an oversized line is skipped without being read into memory).
    Blank lines are ignored. Unparseable lines are recorded and skipped.
    """
    line_no = 0
    while not collector.full:
        chunk = f.readline(MAX_JSONL_LINE_BYTES + 1)
        if not chunk:
            return
        line_no += 1
        if len(chunk) > MAX_JSONL_LINE_BYTES and not chunk.endswith(b"\n"):
            while chunk and not chunk.endswith(b"\n"):  # discard the rest of the line
                chunk = f.readline(1 << 16)
            collector.add(line_no, f"line exceeds {MAX_JSONL_LINE_BYTES} bytes")
            continue
        try:
            text = chunk.decode("utf-8")
            if not text.strip():
                continue
            yield line_no, safejson.loads(text)
        except (ValueError, RecursionError) as e:  # incl. UnicodeDecodeError, JSONDecodeError
            collector.add(line_no, f"invalid JSON: {e}")


class _DatasetStore(Protocol):  # what DatasetService needs; SQLiteStore implements it
    def import_version(self, name, cases, *, description, source, collector, limits): ...
    def resolve_version(self, ref: str) -> DatasetVersion: ...
    def list_datasets(self) -> list[Dataset]: ...
    def list_versions(self, name: str) -> list[DatasetVersion]: ...
    def iter_cases(self, version_id: str, batch_size: int = 1000) -> Iterator[StoredCase]: ...
    def get_case(self, version_id: str, case_key: str) -> StoredCase | None: ...
    def lint_version(self, version_id: str) -> list[LintFinding]: ...
    def verify_version(self, version_id: str) -> VerifyReport: ...


class DatasetService:
    """Import, read, export, lint and verify datasets. Obtain one via `EvalKit.datasets`."""

    def __init__(self, store: _DatasetStore, limits: Limits | None = None):
        self.store = store
        self.limits = limits or Limits()

    # -- import -----------------------------------------------------------------------------

    def import_cases(
        self,
        name: str,
        cases: Iterable[EvaluationCase | Mapping[str, Any]],
        *,
        description: str | None = None,
        source: str | None = None,
    ) -> ImportResult:
        """Validate and import `cases` as a new version of dataset `name`.

        All-or-nothing: on any problem a DatasetError lists them all and nothing (not even the
        dataset) is stored. If the content is identical to an existing version, that version is
        returned (`created=False`).
        """
        validate_dataset_name(name)
        collector = IssueCollector()
        valid = validate_cases(enumerate(cases, 1), self.limits, collector)
        return self.store.import_version(
            name,
            valid,
            description=description,
            source=source,
            collector=collector,
            limits=self.limits,
        )

    def import_jsonl(
        self, name: str, path: str | os.PathLike[str], *, description: str | None = None
    ) -> ImportResult:
        """Import a JSONL file (one case object per line), streaming."""
        validate_dataset_name(name)
        collector = IssueCollector()
        try:
            f = open(path, "rb")  # noqa: SIM115 - closed in the finally below
        except OSError as e:
            raise DatasetError(f"cannot read dataset file: {e}") from e
        try:
            valid = validate_cases(read_jsonl(f, collector), self.limits, collector)
            return self.store.import_version(
                name,
                valid,
                description=description,
                source=f"file:{Path(path).name}",
                collector=collector,
                limits=self.limits,
            )
        except OSError as e:
            raise DatasetError(f"cannot read dataset file: {e}") from e
        finally:
            f.close()

    # -- read -------------------------------------------------------------------------------

    def resolve(self, ref: str) -> DatasetVersion:
        """`name`, `name@latest`, `name@3` (version number) or `name@<hash prefix, 8+ hex>`."""
        return self.store.resolve_version(ref)

    def list(self) -> list[Dataset]:
        return self.store.list_datasets()

    def versions(self, name: str) -> list[DatasetVersion]:
        return self.store.list_versions(name)

    def cases(self, ref: str, *, batch_size: int = 1000) -> Iterator[StoredCase]:
        """Stream the cases of a version in ascending case_key order."""
        return self.store.iter_cases(self.resolve(ref).id, batch_size)

    def get_case(self, ref: str, case_key: str) -> StoredCase | None:
        return self.store.get_case(self.resolve(ref).id, case_key)

    def lint(self, ref: str) -> list[LintFinding]:
        """Non-fatal observations about a version (duplicates, empty fields, missing labels)."""
        return self.store.lint_version(self.resolve(ref).id)

    def verify(self, ref: str) -> VerifyReport:
        """Recompute every case hash and the dataset hash from the stored rows and compare with
        what was recorded at import: detects any modification made behind the database's back."""
        return self.store.verify_version(self.resolve(ref).id)

    # -- export -----------------------------------------------------------------------------

    def export_jsonl(
        self, ref: str, path: str | os.PathLike[str], *, overwrite: bool = False
    ) -> ExportResult:
        """Write a version as JSONL (ascending case_key), verifying each case against its
        recorded hash on the way out. Re-importing the file yields the same content hash, i.e.
        the same version. The file is written atomically with owner-only permissions."""
        version = self.resolve(ref)
        target = Path(path)
        if target.exists() and not overwrite:
            raise DatasetError(f"{target} already exists (overwrite=True / --force replaces it)")
        tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        count = 0
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
                for case in self.store.iter_cases(version.id):
                    if case.content_hash != case.stored_hash:
                        raise DatasetError(
                            f"stored case {case.case_key!r} does not match its recorded hash; "
                            "the database was modified outside evalkit (see `dataset verify`)"
                        )
                    out.write(json.dumps(case.to_record(), ensure_ascii=False) + "\n")
                    count += 1
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return ExportResult(target, count, version.content_hash)
