# Changelog

## Unreleased - P2.1: production operations

Retry, cache, cost and budgets, structured events, a reproducibility snapshot and run-level
observability. Migration 7 applies on open (with a backup); migrations 1-6 are byte-identical, the
single-record `Evaluator` API, the providers' constructors and every existing CLI command and
exit code are unchanged. Every new behaviour is opt-in: with no new flag or policy key a run behaves as
before (the response cache is **off** by default).

### Added

- **`--retry-failed`** (`runs execute|resume RUN --retry-failed`, `controller.execute(...,
  retry_failed=True)`). Failed units whose failure is retryable (`infrastructure.rate_limited`,
  `provider_unavailable`, `timeout`, `connection`, `target.timeout`, and units cut short by a deadline
  or a budget) get another try; `evaluator.invalid_output`, systemic failures and everything
  non-retryable never do. Only failed units are touched: no successful result and no successful
  provider call is repeated. The failed result is replaced **in place** (same row, so every earlier
  attempt stays attached, numbering continues) after its failure is archived in `result_history`; a
  unit is retried at most `policy.retry_failed_rounds` (3) times. A `succeeded` run with retryable
  failures is reopened for it (a database rule accepts that only for an execution that names the run's
  `finished_at`); with nothing to retry the run is left untouched. `runs failures RUN --history` lists
  the superseded failures; `runs show` shows retry history and executions.
- **Response cache** (`--cache readwrite|replay`, `policy.cache`, `evalkit cache stats|clear`). Content-
  addressed: the key covers every request field (system, user, tool schema, temperature, max_tokens,
  sample index, role), the provider, model, endpoint and the evaluator/target scope. Only validated
  successes are stored; only deterministic requests (`temperature <= cache.max_temperature`, default 0)
  are cached; target calls only with `--cache-targets`. `replay` never calls the provider: a miss stops
  the run `partial(cache_miss)`. Identical requests in flight together are paid for once. Every attempt
  records `cache_hit` and its `cache_key`; reports and `runs status` show hits, misses and hit rate.
- **Token and cost tracking.** Provider-reported tokens are stored on every attempt, including a paid
  response that failed validation (before, those were recorded as zero). Cost is an **estimate** from a
  versioned price table you supply (`--pricing FILE`, `policy.pricing`; frozen with the run; the
  version is stored on each attempt). An unpriced model or a provider that reported no usage is
  **unknown**, never zero; a cache hit is exactly 0.0.
- **Budgets enforced per call.** `--max-tokens`, `--max-cost-usd` (needs prices for every paid model;
  refused at preflight otherwise) and `--max-calls`. A call is made only if its worst case (a UTF-8
  upper bound on its input tokens + its output cap) fits what is left, atomically across workers, so
  concurrent workers cannot overspend; the response then replaces the reservation with actual usage
  (unreported usage is assumed to be the worst case). A refused call is not made and its unit stays
  pending; the run stops `partial(budget)` and resumes with a larger budget. Token and USD budgets are
  **per run** (earlier executions count); `max_calls` and `max_duration_s` stay per execution.
- **Structured events** (`--log-level`, `--log-format json|text`, `--log-file`, `EVALKIT_LOG_*`):
  `run.started/finished`, `provider.call`, `provider.retry/failed`, `cache.hit/miss`, `target/evaluator
  .finished/failed`, `retry_failed.reopened`, `budget.exhausted`, `unit.abandoned`, each with run_id,
  case_key, evaluator, attempt, duration, failure class/kind, request id and a per-stage correlation id.
  Only an allow-list of identifier and number fields can be logged: never a prompt, output, reference,
  rubric or provider message, and nothing to switch that on.
- **Reproducibility snapshot.** The frozen run environment now records EvalKit version and commit,
  Python and platform, SDK versions, the dataset (ref, version id, content hash, size), target and
  judge provider/model/generation settings, evaluator, prompt and scoring versions, retry and timeout
  settings, cache and budget configuration and the pricing table (version and hash). Each execution
  also records its runtime and the endpoints (region / host) it used. `compare` treats a different
  endpoint or a different `validation_retries` as a **confounder**; runtime, SDK, price-table and timeout
  differences are informational.
- **Observability.** `runs status RUN [--format text]`, an `operations` block in `runs execute` output
  and a summary on stderr, and an **Operations** section in the static HTML report (cases, evaluator
  coverage, provider calls, retries, cache, tokens, estimated cost, duration, budget state, executions,
  reproducibility). The report stays one self-contained, CSP-locked file.
- `evalkit cache stats|clear [--older-than-days N]`; `examples/pricing.example.toml`.

### Changed

- Attempts gain `cache_hit`, `cache_key`, `cost_usd`, `price_version`, `retry_round`; results gain
  `retry_round`; new tables `result_history`, `run_executions`, `llm_cache` (migration 7). The write-once
  triggers of case and evaluator results now allow exactly two transitions: reopening a *failed* result
  with its failure archived, and replacing a *failed* evaluator result the same way. A successful
  result can never change. `runs verify` also checks that retry rounds match the archived failures.
- A free target's (`precomputed`, `reuse`) case result is now stored together with its first evaluator
  result, so a unit whose evaluator call the budget refuses is left wholly pending.
- `RunSummary.usage` gains `cache`, `provider_calls`, `price_versions`, `cost_complete`,
  `unknown_usage_attempts`; its cost comes from the stored per-attempt costs (a caller-supplied `prices`
  mapping still works).
- LLM clients expose an optional `endpoint` (region / host only) used in the cache key and environment.

### Not done (see docs/ARCHITECTURE.md "Remaining work")

Multi-process workers and leases, adaptive concurrency, self-consistency sampling, streaming analysis,
cross-process cancellation, `evalkit gc`. Live-provider validation of cost estimates and of the cache
was not possible (no working Bedrock credentials): both are tested against fakes only.

## Unreleased - P1.1: hardening of the P1 platform

Fixes the findings of the P1 audit. Migration 6 applies on open (with a backup); migrations 1-5 are
byte-identical, the single-record `Evaluator` API, the providers' constructors and the existing CLI
commands are unchanged.

### Fixed - correctness

- **A regression could pass a gate (audit P0-1).** Comparison paired cases on their full content hash,
  which includes the system's output, so every case whose output changed dropped out and a 33-point loss
  was reported as "equivalent". It now pairs on `(case_key, input_hash)` (prompt, context, reference,
  relevance); unpaired cases are counted and warned about. New end-to-end tests run the real executor,
  evaluators, comparison and gate. **Compare results of runs over different dataset versions can differ
  from before: they were wrong.**
- **Not-applicable could hide a regression.** The not-applicable share is now reported, compared per side,
  warned about, and a declared gate fails on a shift over 5 points (`max_na_asymmetry`), with optional caps
  (`max_na_share`, `max_truncated_share`).
- **Truncated target outputs were scored as complete with no trace.** The target's stop reason and a
  `truncated` flag are stored with the case result, visible to evaluators (`EvalInput.target_meta`),
  counted, warned about, reported and gateable. They never reach a judge's prompt.
- **`succeeded` now means something.** Every case terminal, every evaluator result present (a database
  trigger), every evaluator scored something (`policy.min_coverage`). A run with no evaluator results, or in
  which every case failed, is no longer `succeeded`; `runs execute` exits 1.
- **Gates refuse unpinned callable targets** (`unpinned_target`), as documented; two unpinned runs are no
  longer called "run-to-run noise".
- **`judge-check --run RUN`** checks the run's own frozen evaluator so its result is shown in the report
  (before, it was stored under a hard-coded name that never matched a run). `--name` for standalone checks.

### Fixed - execution

- A failed target whose `skipped` rows were never written stranded the run in `partial(incomplete)`;
  such units are written atomically and repaired on resume.
- **One executor per run** (`RunBusyError`, an OS lock released when its holder dies): two executors
  used to make duplicate target calls. A crashed executor never blocks recovery.
- `execute()` is exception-safe: Ctrl-C or any exception flushes in-flight results, closes the writer and
  leaves a resumable run (`cancelled` / `partial`) instead of a run stuck in `running`.
- A writer-thread exception no longer hangs the run: it is recorded, the queue is spilled, the run stops
  (`infrastructure.storage` or `infrastructure.internal_error`).
- **A single systemic-looking provider error no longer ends a run.** Auth/quota/rejected-request failures
  must repeat (`systemic_threshold`, default 3, per provider and model). Cases caught in the stop stay
  pending, so a resume after the fix completes them. An input too long for the model is `input.oversize`
  (a documented, **unverified-against-a-live-provider** message heuristic).
- **The run's `retry.timeout_s` reaches the provider request** (it was ignored by both clients); it is
  recorded in the run environment and named by `compare` when runs differ.
- **Paging pending units is constant-cost** (was quadratic: 187 ms/page at 100K cases; now 1.1 ms/page).

### Added

- Documents: `docs/EVALUATION-METHODOLOGY.md`, `docs/SECURITY.md`, `docs/DECISIONS.md`; the P1.1 section of
  `docs/ARCHITECTURE.md` lists every deviation. `docs/BENCHMARK.md` gains a 100,000-case measurement.
- CI: a branch-coverage floor (97%); a scheduled/manual mutation workflow. No type checker (DR-12).
- 124 new tests (1,809 in total: end-to-end regression detection; executor seams; realistic provider
  stubs via botocore's `Stubber`; query-plan and instruction-count tests); 24 new mutants of the changed
  invariants (126 in total, all killed).

### Known limits (unchanged or newly documented)

Live validation of the judge prompt is still pending. A `regex` evaluator with a catastrophic pattern can
freeze the process. Compare/calibrate/report load every case of a run into memory. Throughput falls about
44% between 10K and 100K cases (cause not established). Failed results still cannot be retried.


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
