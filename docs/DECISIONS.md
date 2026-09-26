# Decision records

Decisions taken while building and hardening the platform, each with what was rejected and what it
costs. The original design decisions (D1-D16) are in
[`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) §16; the list of deviations from that design is in
[`ARCHITECTURE.md`](ARCHITECTURE.md) ("P1 platform" and "P1.1 hardening"). Format: **Decision · Why ·
Rejected · Cost.**

## DR-1. The v1 "out of scope" list is superseded

- **Decision.** Datasets, runs, batch execution, comparison, retrieval and deterministic metrics, and a
  report are in scope. The original spec (`evalkit-prompt.md`) and the pre-P1 README listed all of
  them as out of scope for v1.
- **Why.** The product goal changed from "a single-call LLM-as-judge library" to "systematically
  evaluate AI systems and detect regressions" (`PRODUCTION-AUDIT.md` §1, §25).
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
- Everything listed under "Remaining P2 work" in the P1.1 section of `ARCHITECTURE.md`.
