"""SQLite persistence for datasets, mixed into `SQLiteStore` (uses its connection and lock)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from pydantic import ValidationError

from evalkit.datasets import (
    _HASH_PREFIX_RE,
    Dataset,
    DatasetVersion,
    EvaluationCase,
    ImportResult,
    IssueCollector,
    LintFinding,
    StoredCase,
    VerifyReport,
    validate_dataset_name,
)
from evalkit.errors import DatasetError
from evalkit.hashing import dataset_hash
from evalkit.limits import Limits
from evalkit.models import format_validation_error

_EXAMPLES = 5
_CASE_COLUMNS = (
    "id, dataset_version_id, case_key, content_hash, prompt, output, reference, context, "
    "retrieved_json, relevance_json, metadata_json, tags_json, input_hash"
)


def _version(row: sqlite3.Row) -> DatasetVersion:
    return DatasetVersion(
        id=row["id"],
        dataset_id=row["dataset_id"],
        dataset_name=row["name"],
        version_no=row["version_no"],
        content_hash=row["content_hash"],
        case_count=row["case_count"],
        source=row["source"],
        created_at=row["created_at"],
    )


def _stored_case(row: sqlite3.Row) -> StoredCase:
    try:
        return StoredCase(
            id=row["id"],
            dataset_version_id=row["dataset_version_id"],
            stored_hash=row["content_hash"],
            stored_input_hash=row["input_hash"],
            case_key=row["case_key"],
            prompt=row["prompt"],
            output=row["output"],
            reference=row["reference"],
            context=row["context"],
            retrieved=json.loads(row["retrieved_json"]) if row["retrieved_json"] else None,
            relevance=json.loads(row["relevance_json"]) if row["relevance_json"] else None,
            metadata=json.loads(row["metadata_json"]),
            tags=json.loads(row["tags_json"]),
        )
    except (ValidationError, ValueError) as e:
        detail = format_validation_error(e) if isinstance(e, ValidationError) else str(e)
        raise DatasetError(f"stored case {row['case_key']!r} is corrupt: {detail}") from e


class DatasetStoreMixin:
    _conn: sqlite3.Connection
    _lock: object  # threading.Lock

    # -- import ---------------------------------------------------------------------------

    def import_version(
        self,
        name: str,
        cases: Iterable[EvaluationCase],
        *,
        description: str | None,
        source: str | None,
        collector: IssueCollector,
        limits: Limits,
    ) -> ImportResult:
        """Insert `cases` as a new sealed version of `name`, atomically.

        `cases` is consumed lazily (streamed); the validating generator upstream records any
        problem in `collector`. On any problem, or an empty input, the transaction is rolled back:
        no version, no cases, not even a new dataset row survives. If a version with identical
        content already exists it is returned instead (`created=False`).

        ponytail: the store lock is held for the whole import, so in-process readers wait; a
        dedicated connection would lift that if concurrent reads during huge imports matter.
        """
        validate_dataset_name(name)
        now = datetime.now(UTC).isoformat()
        with self._lock:  # type: ignore[attr-defined]
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")  # serializes writers; version numbers cannot race
            try:
                row = conn.execute("SELECT id FROM datasets WHERE name = ?", (name,)).fetchone()
                if row is None:
                    dataset_id = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO datasets (id, name, description, created_at) VALUES (?,?,?,?)",
                        (dataset_id, name, description, now),
                    )
                else:
                    dataset_id = row["id"]
                (version_no,) = conn.execute(
                    "SELECT COALESCE(MAX(version_no), 0) + 1 FROM dataset_versions "
                    "WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()
                version_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO dataset_versions (id, dataset_id, version_no, source, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (version_id, dataset_id, version_no, source, now),
                )
                inserted = 0
                for case in cases:
                    try:
                        conn.execute(
                            f"INSERT INTO cases ({_CASE_COLUMNS}) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                str(uuid.uuid4()),
                                version_id,
                                case.case_key,
                                case.content_hash,
                                case.prompt,
                                case.output,
                                case.reference,
                                case.context,
                                None if case.retrieved is None else json.dumps(case.retrieved),
                                None if case.relevance is None else json.dumps(case.relevance),
                                json.dumps(case.metadata, allow_nan=False),
                                json.dumps(case.tags),
                                case.input_hash,
                            ),
                        )
                        inserted += 1
                    except sqlite3.IntegrityError as e:
                        if "UNIQUE" not in str(e):
                            raise
                        collector.add(None, "duplicate case_key", case.case_key)
                collector.raise_if_any("invalid dataset")
                if inserted == 0:
                    raise DatasetError("dataset has no cases; nothing was imported")
                content_hash = self._hash_version(version_id)
                existing = conn.execute(
                    "SELECT v.*, d.name FROM dataset_versions v "
                    "JOIN datasets d ON d.id = v.dataset_id "
                    "WHERE v.dataset_id = ? AND v.content_hash = ?",
                    (dataset_id, content_hash),
                ).fetchone()
                if existing is not None:
                    conn.rollback()  # identical content: keep the existing version, add nothing
                    return ImportResult(
                        self._dataset_by_name(name), _version(existing), created=False
                    )
                conn.execute(
                    "UPDATE dataset_versions SET content_hash = ?, case_count = ?, sealed = 1 "
                    "WHERE id = ?",
                    (content_hash, inserted, version_id),
                )
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise
            return ImportResult(
                self._dataset_by_name(name), self._version_by_id(version_id), created=True
            )

    def _hash_version(self, version_id: str) -> str:
        """Must be called with the lock held. Order comes from the database (byte order)."""
        rows = self._conn.execute(
            "SELECT case_key, content_hash FROM cases WHERE dataset_version_id = ? "
            "ORDER BY case_key",
            (version_id,),
        )
        return dataset_hash((r["case_key"], r["content_hash"]) for r in rows)

    # -- read -----------------------------------------------------------------------------

    def _dataset_by_name(self, name: str) -> Dataset:
        row = self._conn.execute(
            "SELECT d.id, d.name, d.description, d.created_at, COUNT(v.id) AS versions, "
            "MAX(v.version_no) AS latest FROM datasets d "
            "LEFT JOIN dataset_versions v ON v.dataset_id = d.id AND v.sealed = 1 "
            "WHERE d.name = ? GROUP BY d.id",
            (name,),
        ).fetchone()
        if row is None:
            raise DatasetError(f"dataset {name!r} not found")
        return Dataset(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            version_count=row["versions"],
            latest_version=row["latest"],
        )

    def _version_by_id(self, version_id: str) -> DatasetVersion:
        row = self._conn.execute(
            "SELECT v.*, d.name FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id "
            "WHERE v.id = ? AND v.sealed = 1",
            (version_id,),
        ).fetchone()
        if row is None:
            raise DatasetError(f"dataset version {version_id!r} not found")
        return _version(row)

    def list_datasets(self) -> list[Dataset]:
        with self._lock:  # type: ignore[attr-defined]
            names = [r[0] for r in self._conn.execute("SELECT name FROM datasets ORDER BY name")]
            return [self._dataset_by_name(n) for n in names]

    def list_versions(self, name: str) -> list[DatasetVersion]:
        with self._lock:  # type: ignore[attr-defined]
            self._dataset_by_name(name)  # not found -> DatasetError
            rows = self._conn.execute(
                "SELECT v.*, d.name FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id "
                "WHERE d.name = ? AND v.sealed = 1 ORDER BY v.version_no",
                (name,),
            ).fetchall()
            return [_version(r) for r in rows]

    def resolve_version(self, ref: str) -> DatasetVersion:
        """`name`, `name@latest`, `name@<version number>` or `name@<hash prefix>`."""
        name, _, selector = str(ref).partition("@")
        validate_dataset_name(name)
        with self._lock:  # type: ignore[attr-defined]
            self._dataset_by_name(name)
            base = (
                "SELECT v.*, d.name FROM dataset_versions v JOIN datasets d ON d.id=v.dataset_id "
                "WHERE d.name = ? AND v.sealed = 1"
            )
            if selector in ("", "latest"):
                rows = self._conn.execute(base + " ORDER BY v.version_no DESC LIMIT 1", (name,))
            elif selector.isdigit():
                rows = self._conn.execute(base + " AND v.version_no = ?", (name, int(selector)))
            elif _HASH_PREFIX_RE.match(selector):
                rows = self._conn.execute(
                    base + " AND v.content_hash LIKE ? ESCAPE '\\'", (name, selector + "%")
                )
            else:
                raise DatasetError(
                    f"invalid version selector {selector!r} in {ref!r}: use 'latest', a version "
                    "number, or a content-hash prefix of at least 8 hex characters"
                )
            found = rows.fetchall()
        if not found:
            raise DatasetError(f"no version matches {ref!r}")
        if len(found) > 1:
            raise DatasetError(
                f"{ref!r} is ambiguous; use more hash characters or a version number"
            )
        return _version(found[0])

    def iter_cases(self, version_id: str, batch_size: int = 1000) -> Iterator[StoredCase]:
        """Ascending case_key, keyset-paged so memory stays bounded and the lock is released
        between batches."""
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 10_000:
            raise DatasetError("batch_size must be an integer between 1 and 10000")
        after = None
        while True:
            with self._lock:  # type: ignore[attr-defined]
                if after is None:
                    rows = self._conn.execute(
                        f"SELECT {_CASE_COLUMNS} FROM cases WHERE dataset_version_id = ? "
                        "ORDER BY case_key LIMIT ?",
                        (version_id, batch_size),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT {_CASE_COLUMNS} FROM cases WHERE dataset_version_id = ? "
                        "AND case_key > ? ORDER BY case_key LIMIT ?",
                        (version_id, after, batch_size),
                    ).fetchall()
            for row in rows:
                yield _stored_case(row)
            if len(rows) < batch_size:
                return
            after = rows[-1]["case_key"]

    def get_case(self, version_id: str, case_key: str) -> StoredCase | None:
        with self._lock:  # type: ignore[attr-defined]
            row = self._conn.execute(
                f"SELECT {_CASE_COLUMNS} FROM cases WHERE dataset_version_id = ? AND case_key = ?",
                (version_id, case_key),
            ).fetchone()
        return None if row is None else _stored_case(row)

    # -- integrity and lint ---------------------------------------------------------------

    def verify_version(self, version_id: str) -> VerifyReport:
        with self._lock:  # type: ignore[attr-defined]
            version = self._version_by_id(version_id)
        problems: list[str] = []
        pairs: list[tuple[str, str]] = []
        count = 0

        def problem(text: str) -> None:
            if len(problems) < 100:
                problems.append(text)

        for case in self.iter_cases(version_id):
            count += 1
            pairs.append((case.case_key, case.stored_hash))
            if case.content_hash != case.stored_hash:
                problem(f"case {case.case_key!r}: content does not match its recorded hash")
            if case.input_hash != case.stored_input_hash:
                problem(f"case {case.case_key!r}: input does not match its recorded input hash")
        if count != version.case_count:
            problem(f"case count is {count}, recorded {version.case_count}")
        if dataset_hash(pairs) != version.content_hash:
            problem("dataset hash does not match the recorded content hash")
        return VerifyReport(ok=not problems, case_count=count, problems=problems)

    def lint_version(self, version_id: str) -> list[LintFinding]:
        with self._lock:  # type: ignore[attr-defined]
            self._version_by_id(version_id)
            q = self._conn.execute

            def keys(sql: str, *args) -> list[str]:
                return [r[0] for r in q(sql + f" LIMIT {_EXAMPLES}", (version_id, *args))]

            def count(sql: str) -> int:
                return q(sql, (version_id,)).fetchone()[0]

            findings: list[LintFinding] = []

            def add(code: str, n: int, message: str, examples: list[str]) -> None:
                if n:
                    findings.append(LintFinding(code, n, message, examples))

            n = count(
                "SELECT COALESCE(SUM(c - 1), 0) FROM (SELECT COUNT(*) c FROM cases "
                "WHERE dataset_version_id = ? GROUP BY content_hash HAVING c > 1)"
            )
            add(
                "duplicate_content",
                n,
                "cases with identical content under different case_keys",
                keys(
                    "SELECT case_key FROM cases WHERE dataset_version_id = ? AND content_hash IN "
                    "(SELECT content_hash FROM cases WHERE dataset_version_id = ? "
                    "GROUP BY content_hash HAVING COUNT(*) > 1) ORDER BY case_key",
                    version_id,
                ),
            )
            same_prompt = (
                "FROM cases WHERE dataset_version_id = ? AND prompt IN "
                "(SELECT prompt FROM cases WHERE dataset_version_id = ? "
                "GROUP BY prompt HAVING COUNT(DISTINCT content_hash) > 1)"
            )
            add(
                "duplicate_prompt",
                q(f"SELECT COUNT(*) {same_prompt}", (version_id, version_id)).fetchone()[0],
                "cases sharing a prompt but differing elsewhere (fine if intentional)",
                keys(f"SELECT case_key {same_prompt} ORDER BY case_key", version_id),
            )
            for column in ("output", "reference"):
                add(
                    f"empty_{column}",
                    count(
                        f"SELECT COUNT(*) FROM cases WHERE dataset_version_id = ? AND {column} = ''"
                    ),
                    f"cases whose `{column}` is an empty string (an absent field is null)",
                    keys(
                        "SELECT case_key FROM cases WHERE dataset_version_id = ? "
                        f"AND {column} = '' ORDER BY case_key"
                    ),
                )
            with_out, total = q(
                "SELECT SUM(output IS NOT NULL), COUNT(*) FROM cases WHERE dataset_version_id = ?",
                (version_id,),
            ).fetchone()
            if 0 < with_out < total:
                add(
                    "mixed_output_presence",
                    total - with_out,
                    "some cases have a pre-generated `output` and some do not; scoring "
                    "pre-generated outputs needs all of them",
                    keys(
                        "SELECT case_key FROM cases WHERE dataset_version_id = ? "
                        "AND output IS NULL ORDER BY case_key"
                    ),
                )
            add(
                "retrieved_without_relevance",
                count(
                    "SELECT COUNT(*) FROM cases WHERE dataset_version_id = ? "
                    "AND retrieved_json IS NOT NULL AND relevance_json IS NULL"
                ),
                "cases with `retrieved` ids but no `relevance` labels "
                "(retrieval metrics need both)",
                keys(
                    "SELECT case_key FROM cases WHERE dataset_version_id = ? "
                    "AND retrieved_json IS NOT NULL AND relevance_json IS NULL ORDER BY case_key"
                ),
            )
            return findings
