# Decision records

Decisions taken while building, hardening and operating the platform, each with what was rejected and what it
costs. The original design decisions (D1-D16) are in
[`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) §16; the list of deviations from that design is in
[`ARCHITECTURE.md`](ARCHITECTURE.md) ("P1 platform" and "P1.1 hardening"). Format: **Decision · Why ·
Rejected · Cost.**

## DR-1. The v1 "out of scope" list is superseded

- **Decision.** Datasets, runs, batch execution, comparison, retrieval and deterministic metrics, and a
  report are in scope. The original spec and the pre-P1 README listed all of
  them as out of scope for v1.
- **Why.** The product goal changed from "a single-call LLM-as-judge library" to "systematically
  evaluate AI systems and detect regressions".
- **Still out of scope** (`TARGET-ARCHITECTURE.md` §18): an HTTP API or server, Postgres, brokers,
  microservices, a dashboard, an `http` target, BLEU/ROUGE/embedding metrics, user-code evaluators.

## DR-2. Pair runs on the input side of a case

- **Decision.** `compare` pairs on `(case_key, input_hash)`, where `input_hash` covers prompt,
  context, reference and relevance labels and **not** output, retrieved documents, metadata or tags
  (migration 6 adds and backfills `cases.input_hash`).
- **Why.** Pairing on the whole case hash excluded every case whose output changed, so a regression in
  precomputed outputs disappeared (audit P0-1). The question a case asks is the identity to pair on;
  the answer is what is being compared.
- **Rejected.** Keeping the full hash and importing outputs as a separate table (a larger data-model
  change); computing the hash at compare time (needs every prompt streamed on each comparison).
- **Cost.** A stored column and a backfill migration. Two datasets that differ only in metadata or
  tags now pair (intended). `retrieved` is treated as output; a dataset whose `retrieved` documents are
  *inputs* to the system (a fixed candidate set) is not distinguished from one whose `retrieved` is its
  output.

## DR-3. One executor per run, by an OS lock

- **Decision.** `execute` takes an exclusive `flock` on `<db>.locks/<run>.lock` (an in-process
  registry covers in-memory databases and platforms without `fcntl`). A second executor raises
  `RunBusyError`.
- **Why.** Two executors called the target and the judge for the same cases (77 calls for 40 cases in
  the audit's probe). A kernel lock disappears with its holder, so a `kill -9` never leaves a stale
  lock and there is no timeout to tune.
- **Rejected.** A `heartbeat_at` column with a staleness window (stale-state and clock decisions, and it
  is the lease machinery that is P2); no protection (double billing); an advisory flag in `runs`
  (survives a crash, so it *would* go stale).
- **Cost.** Local filesystems only (`flock` is unreliable on network filesystems; SQLite is too).
  Windows gets in-process exclusion only. This is exclusion, not the multi-worker leases of P2.

## DR-4. A systemic failure must repeat before it stops a run

- **Decision.** `systemic_threshold` (default 3) consecutive failures of the same systemic kind from the
  same provider and model with no success of that provider and model in between. Failures of a
  systemic kind are held until a later success proves them case-specific; if the run stops, the held
  cases stay pending.
- **Why.** Design 8.4 says "N failures with the same systemic kind and zero successes". Stopping on the
  first one let a single over-long input end a run; recording the cases caught in a real stop as
  write-once failures made fixing the cause useless.
- **Rejected.** Retrying failed results on resume (`--retry-failed`, P2): a much larger feature. Leaving
  the first-failure stop: a single 400 ends everything.
- **Cost.** A few extra provider calls before a genuinely broken configuration stops (bounded by the
  threshold plus the in-flight window); one systemic-looking failure at the very end of a run is
  recorded as that case's failure.

## DR-5. What `succeeded` means

- **Decision.** Every case terminal, every evaluator result present (database trigger), and every
  evaluator scored at least one case and at least `policy.min_coverage` (default 0) of its applicable
  cases; otherwise `partial(insufficient_coverage)`. A run must have at least one evaluator.
- **Why.** A run whose every case failed, or with no evaluator results, was reported `succeeded` and
  `runs execute` exited 0 (audit P1-7). The design says success needs coverage at least `min_coverage`.
- **Rejected.** A default of 0.95 (the gate default): it would turn ordinary runs with a few unrepairable
  failures into permanent `partial`, since failed results cannot be retried yet. **Gates, not run
  status, remain where a real coverage bar is enforced.**
- **Cost.** A run with 60% coverage still succeeds. Documented in `EVALUATION-METHODOLOGY.md` §5.

## DR-6. Not-applicable is watched, not banned

- **Decision.** Report the not-applicable share; compare it across sides; a declared gate fails on a
  shift over 5 points by default (`max_na_asymmetry`) and can cap the share (`max_na_share`).
- **Why.** `not_applicable` depends on the output for `citation_check` (no citations), so a candidate
  could leave a metric and keep 100% coverage of the remainder.
- **Rejected.** Scoring "no citations" as 0 (that is a property of one evaluator and one policy, not of
  the platform); banning `not_applicable` (retrieval without labels is legitimately undefined).
- **Cost.** A legitimately different dataset version can trip the gate; the tolerance is configurable.

## DR-7. Truncation is data, not a failure

- **Decision.** A target output that hit its limit is scored as an answer and flagged
  (`case_results.meta_json`), visible to evaluators, summarized, reported, compared and gateable.
- **Why.** A truncated answer is the system's quality; failing it would hide the quality signal. But it
  must not be *silently* scored as complete.
- **Cost.** Existing runs have no metadata (NULL): unknown, not "not truncated".

## DR-8. The run's timeout reaches the provider

- **Decision.** `retry.timeout_s` is the request timeout: per request for the OpenAI SDK; a boto3 client
  per distinct timeout (boto3 fixes timeouts per client). It is recorded in the run environment and
  named by `compare`. A caller-supplied boto3 client is used as is.
- **Rejected.** A thread-based wall-clock wrapper around every call (abandoned threads, and the SDK
  timeout would still apply); leaving the env var as the only source (a policy field that did nothing).

## DR-9. Gates refuse unpinned callable targets

- **Decision.** With any declared gate, a callable target without a `fingerprint` (candidate or
  baseline) is a gate failure (`unpinned_target`), unless `allow_unpinned`. Two unpinned runs are never
  "the same identity" and get no noise floor.
- **Why.** EvalKit cannot hash code; the documentation said gates refuse it and they did not (audit
  P1-12).

## DR-10. `judge-check` checks a run's own evaluator

- **Decision.** `judge-check --run RUN [--evaluator NAME]` builds the check from the run's frozen spec
  and stores it under that evaluator key; the built-in golden set is refused for another rubric.
- **Why.** A check stored under a hard-coded name never matched a run's key, so reports always said
  "not checked" (audit P1-11).

## DR-11. Schema changes are new migrations

- **Decision.** Migration 6 adds columns, triggers and a backfill. Migrations 3-5 are untouched from
  here on, even though they never shipped in a release.
- **Why.** P1 edited migration 4 in place while it was unreleased (documented then). Any database
  created at a P1 commit would now be refused, so from this point the checksums are treated as
  released.

## DR-12. Type checking is deferred

- **Decision.** No type checker is added or enforced in CI.
- **Why.** The code is annotated but not written to a checker's standard (`Any` at the SDK and JSON
  seams, `# type: ignore` in places, pydantic-heavy models). Adding `mypy`/`pyright` now would mean
  either a wall of ignores that proves nothing or a large refactor unrelated to the audit. The
  strategy that does exist is behavioural: property tests, raw-SQL invariant tests, mutation tests.
- **Revisit** when a public typed surface is promised (`py.typed`) or a second contributor joins.

## DR-13. Mutation testing runs on a schedule, not on every push

- **Decision.** `scripts/mutation.py` (hand-picked mutants of the invariants that matter) runs in a
  separate scheduled/manual workflow; the per-push CI runs tests, lint, format and a coverage floor.
- **Why.** The full set takes tens of minutes. The suite still has to *kill* every mutant: a survivor is
  a defect in the tests.

## DR-14. Deferred on purpose (P1.1 did not touch)

- **Regex evaluator process freeze** (a catastrophic pattern on a crafted output blocks the whole
  process): the operator writes the pattern; a real fix needs a subprocess or `re2`. Documented in
  `SECURITY.md`.
- **Analysis loads every case outcome into memory** (compare, calibrate, report); fine at the measured
  sizes, the first ceiling beyond them.
- Everything listed under "Remaining P2 work" in `ARCHITECTURE.md` (superseded by the P2.1 list).

## DR-15. Retry replaces a failed result in place, with its failure archived

- **Decision.** `--retry-failed` gives a retryable failed result a new outcome *on the same row*, after
  copying its failure into the append-only `result_history`. The write-once triggers are replaced by
  ones that allow exactly that: `failed -> pending` (case) and `failed -> ok | not_applicable | failed`
  (evaluator), the round advanced by one, the failure archived. A successful result never changes.
- **Why.** Every earlier attempt (evidence, tokens, cost) stays attached to the row and numbering
  continues; every analysis query keeps reading "the current result" with no generation column and no
  join; `succeeded` and coverage stay defined as before.
- **Rejected.** A new row per try with a "current" flag or generation (every aggregate and index would
  change; the `UNIQUE(run, case)` constraints cannot be relaxed without rebuilding the biggest tables);
  deleting the failed row (destroys history, and attempts reference it); a separate retry table that
  analysis must merge.
- **Cost.** Two triggers are more permissive than before; each has raw-SQL tests including forged
  history rows, and `runs verify` checks that retry rounds match the archive. A retry changes a finished
  run's numbers (methodology §12).

## DR-16. A succeeded run is reopened only by a recorded execution

- **Decision.** Retrying failures of a `succeeded` run (the usual case: P1.1 made a run with a few
  failures `succeeded`) moves it `succeeded -> running -> succeeded`. The database allows that only when
  a `run_executions` row names the run's current `finished_at`; the service-level state machine
  (`RunService.transition`) still refuses it, and the existing table-driven tests of all 36 status pairs
  are unchanged.
- **Why.** Without it `--retry-failed` would not work on the runs that need it; with a plain trigger
  relaxation, any writer could reopen a finished run.
- **Rejected.** Making runs with retryable failures `partial` (changes P1.1's meaning of `succeeded`);
  requiring a new run (loses history; a rerun with the cache is still available for judges).
- **Cost.** A finished run can change after the fact; each change is a recorded execution and a new
  summary snapshot.

## DR-17. `evaluator.invalid_output` is never retried by `--retry-failed`

- **Decision.** Eligible: retryable failures except `evaluator.invalid_output`, systemic kinds, and the
  non-retryable rest; units cut short by a deadline or a budget are eligible although their kind is not
  "retryable" in the in-unit sense. One rule (`failures.retry_eligible`), mirrored in SQL and tested equal
  for every kind and flag.
- **Why.** The in-unit retry already fed the validation error back; at temperature 0 a later identical
  try reproduces the answer and spends money for nothing (design 8.5 says the same).
- **Cost.** A judge that was merely unlucky twice stays failed; that is what coverage reports.

## DR-18. Token and USD budgets are per run and enforced per call; `max_calls` stays per execution

- **Decision.** A call is admitted only if its worst case fits every remaining limit, checked
  atomically under the guard's lock, with what earlier executions spent counted. `max_calls` and
  `max_duration_s` keep their P1 per-execution meaning.
- **Why.** "Cannot overspend" needs the check *at the call*, not at unit dispatch; a per-run budget must
  survive a resume. An existing test pins `max_calls` as per execution (a resume with the same budget
  makes progress), and P1's documented semantics are preserved rather than silently changed.
- **Rejected.** Checking at dispatch (overshoots by the window, as P1 did); a soft budget with a stated
  overshoot; making every budget per run (would change P1 behaviour).
- **Cost.** Conservative: the worst case counts input <= UTF-8 bytes + 64 and output = `max_tokens`, so a
  budget below one call's worst case runs nothing and the tail of a budget goes unused. The bound is an
  assumption about tokenizers, not a proof.

## DR-19. A refused call leaves its unit pending, not failed

- **Decision.** Budget refusal and a replay miss raise `UnitAbandoned`: nothing is recorded, the run
  stops `partial(budget)` / `partial(cache_miss)`, and the resume finishes the unit. A free target's case
  result is written together with its first evaluator result so the unit is wholly pending. A refusal
  after a real attempt in the same call is recorded as `infrastructure.budget_exceeded` (evidence kept),
  which `--retry-failed` retries.
- **Why.** Results are write-once: a failure recorded because the budget ran out would make raising the
  budget useless.

## DR-20. The response cache is off by default and scoped

- **Decision.** `cache.mode` defaults to `off`; the key covers every request field (minus `timeout_s`),
  provider, model, endpoint and the evaluator/target scope; only validated, deterministic responses are
  stored; targets only with `--cache-targets`; `replay` never calls the provider.
- **Why.** A cache changes what a rerun measures (it repeats a judge's answers), so it must be a choice;
  the design's judge-by-default would also have changed the call counts P1 tests pin. The scope is in
  the key because "cache identity must include evaluator configuration": the request text usually
  already differs, and when it does not, not sharing is the safe error.
- **Rejected.** Caching final results; caching failures; caching targets by default; an LRU with a size
  cap (a `cache clear` exists; `gc` is not built); a global lock around calls (a per-key lock only).
- **Cost.** A rerun of an unchanged judge does not share entries between two evaluators that differ
  only in scoring parameters; a judge whose first answer is invalid re-pays it on a rerun.

## DR-21. No bundled price table; unknown is unknown

- **Decision.** Prices are supplied (`--pricing`, `policy.pricing`), versioned, frozen into the run's
  policy and stamped on each attempt. Unpriced and no-usage are `NULL`; cost totals say when they are
  partial; a USD budget is refused unless every paid model is priced.
- **Why.** EvalKit cannot verify a provider's prices from here; a packaged table would be a claim it
  cannot back, and a stale one silently wrong. Zero-by-default would understate spend.
- **Cost.** The operator maintains a small file.

## DR-22. Paid responses that fail validation record their usage

- **Decision.** The tokens and cost of a response that arrived but failed validation are recorded on the
  failed attempt (and count against budgets).
- **Why.** They were billed. Before, an invalid judge answer counted as zero tokens, so both reports and
  budgets undercounted exactly the wasteful calls.

## DR-23. Events carry an allow-list of fields and no content

- **Decision.** `events.emit` drops any field not in `FIELDS` and any non-scalar value, scrubs and clips
  strings, and identifies failures by class and kind, not message. There is no option to log content.
  Nothing is emitted until a handler is configured (a `NullHandler` keeps a library quiet).
- **Why.** Prompts, outputs and provider messages are sensitive and can contain secrets; the safest
  interface is one that cannot accept them.
- **Cost.** Debugging a specific failure needs the stored attempt evidence, not the log.

## DR-24. What counts as an environment confounder

- **Decision.** A different endpoint (region / host) or a different `validation_retries` blocks a
  comparison (unless allowed); EvalKit / Python / SDK / SQLite versions, the commit, price tables,
  transport attempts, deadlines and timeouts are informational; a run without a snapshot is unknown.
- **Why.** Confounders are differences that can change scores or which cases are scored. Blocking on
  every version bump would make every cross-release comparison need `--allow-confounders`; scoring
  changes are already a confounder through `scoring_version`.
- **Cost.** A bug fix that changes scores without a `scoring_version` bump is visible only as an
  informational note.

## DR-25. Executions are recorded; the run's snapshot is split in two

- **Decision.** The frozen environment holds what defines the measurement; each `execute` writes a
  `run_executions` row (environment incl. endpoints, cache mode, budget in force, spend, outcome,
  retry counts). Duration, budget state and "which machine resumed this" come from there.
- **Why.** Endpoints and runtime are known only when the clients exist (at execute), and a resume can
  happen elsewhere; a frozen record cannot hold that.
- **Cost.** One more table; an interrupted execution's row stays unfinished (shown as such).

