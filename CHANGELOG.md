# Changelog

## Unreleased - P1 (in progress)

### Runs, results, attempts, failure taxonomy

- New: the persistent run domain (`EvalKit.runs`, `evalkit runs create|list|show|failures|verify`):
  `Run` (frozen `RunConfig`, one sealed dataset version, `identity_hash` vs `exec_hash`, lifecycle),
  `CaseResult`, `EvaluatorResult` with named metrics, and `RunAttempt` (every external call, with
  P0 evidence rules). Migration 4 (`runs_and_results`): tables, CHECKs and write-once triggers.
  This records results; it does not execute anything yet.
- New: failure taxonomy `FailureClass` (input / target / evaluator / infrastructure), `Failure`,
  `EvalFailure`. A target failure is a case-result failure and its evaluator results can only be
  `skipped`: it never becomes a score.
- `evalkit.evidence` now holds the attempt-evidence capture shared by the single-record `Evaluator`
  and run attempts (pure refactor, behaviour unchanged).

### Dataset foundation

- New: versioned, immutable, content-addressed datasets (`EvalKit.datasets`, `evalkit dataset ...`):
  atomic strict JSONL import with all-problems reporting, idempotent re-import, `name@version`
  refs, export round-trip, lint, and `verify`. Migration 3 (`datasets`, `dataset_versions`, `cases`)
  applies automatically on open; sealed data is immutable via database triggers.
- New: `evalkit.hashing` (canonical JSON, domain-separated stable hashes), `DatasetError`, and
  case/import limits (`Limits.max_case_bytes`, `max_import_cases`).
- The single-record `Evaluator` API, its tables and CLI commands are unchanged.

## 0.2.0 - P0 hardening of the single-record path

Behavior changes worth knowing about (details: `docs/ARCHITECTURE.md`, "0.2.0 (P0 hardening) changes"):

- **Judge prompt bytes changed** (lossless delimiter encoding replaces HTML escaping; context
  instruction reworded), so `judge_prompt_version` changed. Results from before and after 0.2.0
  are deliberately not comparable. The new wording has not been validated against a live judge.
- `Criterion.weight` must be within `[1e-6, 1e6]` and finite; scale bounds within +/-1,000,000;
  rubric text is length-limited and may not contain `<<<EVALKIT:` / `<<<END:`. Inputs over the
  documented limits are refused with `RubricError` before any judge call.
- A failed save now raises `StoreError` (still an `sqlite3.Error`) carrying the judged result, and
  spills it to `<db>.spill/`. New `evalkit recover` command.
- Databases are migrated on open (a `.bak-*` copy is written first); files are `0600` and use WAL.
  Databases from 0.1.0 and from before the `context` column are upgraded; unrecognized or newer
  databases are refused.
- Reusing a rubric `version` label for different content raises `RubricError`.
- `EvaluationResult.attempts` records every judge call; new error types `ScoringError`,
  `StoreError`, `MigrationError`; new provider error `kind`s for guardrail/content-filter/malformed/
  context-window stop reasons.
- `list(limit)` is validated (1..10,000); new `list_page()` and `evalkit list --cursor`.
- Error messages are scrubbed of AWS ARNs, keys and bearer tokens before being stored or printed.
- The CLI reports internal `TypeError`s as bugs (traceback) instead of "invalid input file".
