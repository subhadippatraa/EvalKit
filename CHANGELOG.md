# Changelog

## Unreleased - P1: the evaluation platform

Adds a single-process evaluation platform on the existing SQLite store. The single-record
`Evaluator` API, its tables and its CLI commands are unchanged; migrations 3-5 apply on open (with a
backup) and released migrations 1-2 are byte-identical.

### Execution, targets, evaluators (migration 5 adds summaries, tags, generalized reviews, judge checks)

- New: `EvalKit.controller` (`RunController`): preflight, `create`, `execute` / resume, cancellation
  (`CancelToken`, Ctrl-C in the CLI), a thread pool with a bounded window and a single batching
  writer, seeded processing order, retry with full-jitter backoff, per-call timeouts, a rate limiter,
  a consecutive-failure breaker, token / call / duration budgets, spill-and-replay when storage
  fails. Not included (P2): retrying failed results, leases / multi-process workers, response cache.
- New targets: `precomputed`, `reuse`, `callable`, `model` (no `http`). Targets never see the
  reference, labels, tags or metadata.
- New evaluators: `exact_match`, `regex`, `json_schema` (optional `evalkit[jsonschema]`),
  `retrieval`, `citation_check`, `llm_judge`; `Criterion.must_pass` / `min_normalized` (rubric hashes of
  existing rubrics are unchanged).
- **LLM client split**: `evalkit.llm` (`LLMClient`, `LLMRequest`, `LLMResponse`), `BedrockClient` /
  `BedrockOpenAIClient` own the request, usage, latency, request id and classified provider errors;
  `evalkit.judge_eval` owns judge semantics. `BedrockJudge` / `BedrockOpenAIJudge` remain, unchanged
  in behaviour, as adapters (their errors now also carry `.failure`). SDK retries stay off.
- Failure taxonomy gains `infrastructure.internal_error`; `input.bad_config` is systemic.

### Analysis

- New: aggregation with denominators, coverage and intervals (headline withheld below the required
  coverage), paired comparison with confounder detection, decision rule and declared gates
  (`evalkit compare`, exit codes 3 / 4), human review / calibration / review queues / evaluator
  disagreement, `judge-check` golden set with adversarial cases, static HTML report
  (`evalkit report`). `reviews` is rebuilt (append-only) to grade a run's case result as well as a
  legacy evaluation.
- New CLI: `runs create|execute|resume|tag|review|calibrate|queue|disagreements`, `compare`,
  `report`, `judge-check`. `runs create` now takes a spec file (preflight + plan).
- New: `examples/quickstart`, `scripts/benchmark.py`, `scripts/mutation.py`, `docs/BENCHMARK.md`.


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
