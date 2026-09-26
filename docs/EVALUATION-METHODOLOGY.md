# Evaluation methodology

What an EvalKit number means, what it does not, and which choices were made on purpose. It is the
companion of [`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) (§7, §9) and states the behaviour
of the code as of the P1.1 hardening. Where something is a heuristic or unverified it says so.

## 1. What is measured

A **run** applies a frozen set of **evaluators** to the outputs of one **target** over one exact
**dataset version**. Every case ends in exactly one *case result* (the target stage) and, for each
evaluator, one *evaluator result*. Nothing is an average of averages: every reported number is a
function of stored rows and states how many cases it is over.

| Question | Where the answer comes from |
|---|---|
| What was measured? | `identity_hash`: dataset content, target identity, evaluator keys, scoring version |
| How was it executed? | `exec_hash`: the execution policy (concurrency, retry, timeouts, budgets) |
| Which cases count? | coverage and the per-evaluator accounting (below) |

`identity_hash` and `exec_hash` are different on purpose: two runs with the same identity measure the
same thing; a different execution policy changes speed, cost and which cases fail, not what a score
means. (Two policy settings can change *outcomes*: `retry.validation_retries` gives a judge a second
try, and `on_missing` decides whether a missing field is a failure or "not applicable". `compare`
reports the policy difference, but does not treat it as a confounder; see §6.)

## 2. Denominators and coverage

For each evaluator, every case is exactly one of: **scored** (`ok`), **not applicable**, **skipped**
(the target failed, so nothing was scored), **evaluator-failed**, **infrastructure-failed**,
**input-failed**, or still **missing** (no result yet).

```
coverage = scored / (cases - not_applicable)
```

- A **target failure is never a score.** It is a case-result failure; its evaluators are `skipped`.
  It counts against coverage and against `pass_rate_strict` (passes / *all* cases), and is reported as
  `target_failure_rate`. A crash is not turned into a `0` on a 1-5 rubric: that would invent a judge
  opinion nobody gave.
- **`not_applicable` is a first-class outcome**, for a metric that is *undefined* for a case (recall
  with no labelled relevant document; a citation check on an output with no citations). It is
  excluded from means and from the coverage denominator, never counted as a zero.
- **Below the required coverage the headline value is withheld.** The observed mean stays visible,
  flagged. Missing data is not evidence of health.

### 2.1 Not-applicable is a way to leave a metric (and is therefore watched)

`not_applicable` is safe when it depends on the *case* (the same cases are excluded for both systems).
It is not safe when it depends on the *output*: a system that stops emitting citations moves its cases
out of `citation_check`, its coverage stays at 100% of what remains, and the metric looks unchanged.
EvalKit therefore:

- reports each evaluator's **not-applicable share** and warns when it is 50% or more;
- in `compare`, reports the share **per side over the paired cases** and warns when they differ by
  more than 5 points;
- by default **fails a declared gate** whose evaluator's share moved by more than 5 points
  (`max_na_asymmetry`, `null`/absent to disable) and can cap the share outright (`max_na_share`).

### 2.2 Truncated outputs

A target answer that hit its output limit (`stop_reason = max_tokens`) is *kept as an answer* and
flagged (`case_results.meta_json`: `stop_reason`, `truncated`). It is scored as if complete: a
truncated answer is the system's quality, not a failure of measurement. The flag is stored, visible to
evaluators (`EvalInput.target_meta`), counted in the summary and report, warned about, compared, and
gateable (`max_truncated_share`). It is **never** put into a judge's prompt (provenance must not reach
the judge).

## 3. Comparing two runs

`compare(baseline, candidate)` pairs results **by `case_key` and the case's input identity**
(`input_hash`: prompt, context, reference, relevance labels). It deliberately does **not** include the
system's output, retrieved documents, metadata or tags: those describe an execution, not the question.
(Pairing on the whole case content dropped every case whose output had changed, that is, the
regressions, and let a 33-point loss come back as "equivalent". That was audit finding P0-1, and
`tests/test_regression_e2e.py` guards it end to end.)

- A case whose *input* differs between the runs (or that exists in one run only) is **not paired**; it
  is counted (`Comparison.unpaired`) and warned about.
- **Confounders**: a different evaluator key (judge model, rubric, prompt fingerprint, parameters) or
  scoring version makes the difference unattributable, so the comparison is refused unless
  `--allow-confounders`. Target differences are the point of a comparison and are not confounders.
  A different dataset version is informational: the pairing above is what makes it safe.
- **Survivorship**: cases unscored on either side are excluded and counted per side and reason; a
  coverage gap above 5 points is warned about (a candidate that "improves" by crashing on hard cases).
- **Statistics** (stdlib only, seeded, so a report is reproducible): the paired difference with a 95%
  interval (percentile bootstrap, B = 2,000, below 5,000 pairs; normal approximation above), Wilson
  intervals for proportions, exact McNemar for the discordant pairs of a binary metric, and the
  **minimum detectable difference** `2.8 x SE` of the paired differences (about 80% power, 5%
  alpha). A dataset that cannot detect a change of size *x* says so.

## 4. The decision rule and gates

For a declared gate (`direction`, tolerance `delta` or relative `rel_delta`), the paired interval
`[lo, hi]` is compared with the tolerance band:

| Condition (direction `higher`) | Decision |
|---|---|
| fewer than `min_n` (30) paired cases | INCONCLUSIVE (`insufficient_data`) |
| `hi < -delta` | REGRESSION |
| `lo > +delta` | IMPROVEMENT |
| `-delta < lo` and `hi < +delta` (and `delta > 0`) | EQUIVALENT |
| otherwise | INCONCLUSIVE, with the minimum detectable difference |

- **Only declared gates can fail a build.** Other metrics are informational; that is the guard against
  fishing across many metrics (the report footer states that multiple comparisons inflate false
  positives; no correction is applied).
- A gate **fails** on: a regression; a candidate below an absolute **floor**; a **must-pass** criterion
  failing more often than allowed; **coverage below the minimum** for any evaluator a gate refers to;
  a not-applicable or truncation share beyond its cap; and an **unpinned callable target**.
- **Unpinned targets.** EvalKit cannot hash code. A callable target must declare a `fingerprint`
  (a git SHA, say) or a gate refuses it (`allow_unpinned` opts out explicitly). Two unpinned runs are
  never called "the same measurement" or given a noise floor, since a rerun and a code change look
  identical.
- Exit codes: `0` pass, `1` error, `2` usage, `3` a gate failed, `4` inconclusive under `--strict`.
  Without `--strict` an inconclusive gate exits `0`: **declare `--strict` in CI if "cannot tell" must
  block a merge.**
- The statistical self-test (`tests/test_compare.py`) simulates the comparator: under no difference
  the false-regression rate stays at or below 8% over 1,000 simulations, and an effect of 1.5 x the
  minimum detectable difference is detected at least 80% of the time.

## 5. What "succeeded" means for a run

A run is `succeeded` only when **every case has a terminal result, every terminal case has a result
from every evaluator (the database enforces this), and every evaluator scored something** (at least
one `ok` result, and at least `min_coverage` of its applicable cases; the default `min_coverage` is
0, i.e. "scored anything"). A run in which every case failed at the target, or in which an evaluator
scored nothing, ends `partial(insufficient_coverage)` and `runs execute` exits `1`.

This is a floor, not a quality bar. **A run can succeed with 60% coverage; the gates are what require
95%.** Set `policy.min_coverage` to make the run itself stricter.

## 6. Reproducibility

- **Dataset versions** are immutable and content-addressed; re-importing identical content returns the
  same version. Runs pin one sealed version.
- **Evaluator keys** (`kind:name:hash`) cover every parameter that can change a score, the rubric's
  content hash, the judge model and the judge **prompt fingerprint** (a hash of the static text *and*
  of a fixed fixture rendered through the real prompt code, so a change in how the prompt is built
  changes the key). Results with different keys are never silently compared.
- **Seeded order**: cases are processed in `sha256(seed || case_key)` order, so a run that stops early
  (budget, breaker, cancel) has processed an unbiased sample.
- **Seeded statistics**: every interval is seeded; a report is byte-identical for the same rows and
  seed (tested across `PYTHONHASHSEED` values).
- **What is not pinned**: a hosted LLM at temperature 0 is not deterministic, and the provider can
  change a model behind an id. EvalKit records the environment (§13) but cannot pin the model's
  behaviour; only the response cache (§10, `replay`) reproduces a judge's answers exactly. A rerun of
  the same identity gives a **noise floor** (`compare` reports it when both runs are pinned); without
  one, a tolerance is a guess.
- **Policy settings that affect outcomes**: `retry.validation_retries` (a judge answer rejected by
  validation is retried once with the error fed back, so the *accepted* answer can be a second
  sample) and `on_missing` are execution policy (in `exec_hash`), not identity. Keep them equal across
  runs you compare, or read the policy line of the comparison.
- **Timeouts**: the run's `retry.timeout_s` is the timeout of every provider request (per request for
  the OpenAI-compatible SDK; boto3 fixes timeouts per client, so a client is built per distinct
  timeout). It is frozen in the run and recorded in the run environment; `compare` names a difference.

## 7. Failures and who is responsible

| Class | Meaning | Examples |
|---|---|---|
| `input` | the case or configuration is unusable | missing field, an input too long for a model (`oversize`), a target's rejected configuration |
| `target` | the system under test failed | exception, timeout, contract violation, blocked output |
| `evaluator` | a healthy provider returned something the evaluator cannot use | invalid or truncated judge output, refusal, a rejected judge request |
| `infrastructure` | we did not get a complete response at all | throttling, 5xx, connection, auth, quota, storage, budget, cancellation |

**Systemic failures** (auth, quota, a rejected request) fail every case the same way, so the run
should stop. But one rejected request among thousands is that case's problem. The run therefore stops
(`failed`) only when the **same systemic kind repeats `systemic_threshold` times (default 3) from the
same provider and model with no success of that provider and model in between**. A working callable
target does not mask a broken judge. Cases caught in that stop are left **pending**, not recorded as
write-once failures, so resuming after the fix does the remaining work.

**An over-long input** is recognised by a narrow message/code heuristic (`context_length_exceeded`;
"input is too long", "too many tokens", ...) and recorded as `input.oversize`, non-systemic. **This
heuristic is unverified against a live provider**: Bedrock is understood to report such inputs as a
`ValidationException`, but no live call could be made while developing (see §9). A phrasing it does
not recognise is a rejected request, stopped only on repetition.

## 8. Judge trust

A judge is an instrument with unknown error, not ground truth.

- **Deterministic evaluators are first-class** so claims do not rest on a judge alone.
- **Must-pass criteria** stop a weighted mean from hiding a failed critical criterion.
- **Calibration** (`runs calibrate`) compares an evaluator's verdicts with human consensus from
  **random-sample reviews only** (accuracy, FAIL precision/recall, Cohen's kappa, reviewer
  agreement); below 30 paired cases the evaluator is `UNCALIBRATED`. Targeted (triage) reviews are
  stored but never enter the statistics. `sample=random` is the reviewer's own declaration: nothing
  proves a case came from the random queue.
- **`judge-check`** runs a golden set with adversarial cases (hidden instructions, verbosity padding,
  forged delimiters, injected context) through a judge and stores the accuracy under the
  **evaluator key**. Use `judge-check --run RUN` to check the run's own frozen evaluator so the
  report shows it; the built-in golden set is written for the built-in rubric only and is refused for
  any other (supply `--cases`). The set is small and hand-labelled: it detects gross failure,
  injection susceptibility and drift, not subtle bias.
- **Diagnostics**: cross-evaluator disagreement, a length-bias correlation, and a self-preference
  warning (a heuristic on model-family names).
- **Semantic prompt injection cannot be prevented by encoding.** Prompt content is delimited
  injectively (a per-request marker derived from the content), which removes *structural* break-out;
  a judge can still be talked into a score. It is made detectable, not impossible.

## 9. What has not been validated

- **No live-provider validation.** The Bedrock account available during development answered every
  model invocation with `ValidationException: Operation not allowed`. Provider clients are tested
  against botocore's own `Stubber` (which validates requests and responses against the AWS service
  model) and against the OpenAI SDK's real error types, not against a live model. The **judge prompt,
  including the wording that context is the "source of truth", has never been checked against a real
  model**, and the design's own condition for shipping that wording (no loss of golden-set accuracy) is
  therefore unmet. Treat judge scores as uncalibrated until `judge-check` and `calibrate` have been run
  against your judge.
- The oversize-input heuristic (§7) and every provider's real throttling behaviour.
- **Cost estimates and the cache have not met a live provider.** Token counts are whatever the
  provider reports and the price table is yours; the estimate has never been reconciled with an
  invoice. The budget's input-size bound (§11) is an assumption about tokenizers, checked only
  against fakes.
- Anything beyond one process, one local SQLite file, and the measured dataset sizes (see
  [`BENCHMARK.md`](BENCHMARK.md)).

## 10. The response cache

- **What a hit means.** The provider was not called; the *stored* validated response was used and is
  re-validated by today's code. A hit is recorded as an attempt (`cache_hit = 1`, cost exactly 0.0, no
  tokens). A run's tokens and cost are therefore what *this run* spent, and `usage.cache` says how much
  of it was served from the cache.
- **Identity.** The key is a hash of every request field (system, user, tool name / description /
  schema, temperature, max_tokens, sample index, role), the provider, model, endpoint (region or host)
  and a scope: the evaluator key (rubric, prompt version, scoring version) or the target's identity.
  Any difference is a different entry, including an evaluator configuration change that leaves the
  request text identical (deliberately conservative: it forgoes sharing to rule out crossing).
- **What is cached.** Validated successes of *deterministic* requests only. A response that failed
  validation is never cached, so a judge whose first answer is invalid re-pays that first call on a
  rerun (the corrected retry hits). Target calls are cached only with `--cache-targets`: a target's
  variability is what an evaluation measures, and caching it would flatter reruns and hide
  regressions. Raw provider payloads kept as failure evidence are not cached.
- **Replay.** `--cache replay` answers from the cache alone and never calls the provider; a miss stops
  the run `partial(cache_miss)`. It is the honest reproduction of a judged run: same responses, same
  scoring code.
- **Cache and comparison.** A judge that is non-deterministic at temperature 0 gives different scores
  on different live runs; a cached rerun repeats the first run's answers, so a cached rerun **understates
  run-to-run noise**. Do not use one to measure the noise floor.
- **Limits.** Entries are never evicted automatically (`evalkit cache clear`); the endpoint, not other
  client settings, is in the key; a provider that changes a model behind a stable id is invisible to it.

## 11. Cost and budgets

- **Tokens are what the provider reported**, per attempt, including a paid response that then failed
  validation. A provider that reports none leaves the attempt's usage `NULL` (*unknown*), counted in
  `unknown_usage_attempts`.
- **Cost is an estimate**: reported tokens x the run's price table (`policy.pricing`, versioned, frozen
  with the run; the version is stored on every attempt). EvalKit ships no prices. An unpriced model is
  *unknown*, not zero, and the run's cost is then labelled **partial** with the number of unpriced
  attempts. A cache hit is exactly zero. A callable target has no price: its spend is not in the cost.
- **Budgets are enforced before each call.** A call is admitted only if
  `spent + reserved by calls in flight + this call's worst case` is within every limit, checked
  atomically, so concurrent workers cannot overspend. The worst case is an *upper bound*: input tokens
  <= the request's UTF-8 bytes + 64 (a token is at least a byte for byte-level tokenizers), output
  tokens <= `max_tokens`. Consequences: the bound is conservative (the last few thousand tokens of a
  budget go unused; a budget below one call's worst case runs nothing), so **set a judge's
  `max_tokens` close to what its answers need** (the default is 4096); and a provider whose tokenizer
  violates the assumption could still overshoot by the difference. Usage the provider did not report is
  assumed to have been the worst case *within the execution*; across a resume only reported usage
  carries over.
- **Per run, not per execution** for tokens and USD: a resume counts what earlier executions spent, so
  restarting cannot spend a budget twice. `max_calls` and `max_duration_s` are per execution (their P1
  behaviour). A USD budget needs a price for every model the run pays for, else it is refused up front.
- **Failed calls are counted at what the provider reported** (usually nothing): a request that timed out
  may still have been billed, and EvalKit cannot know.
- **Stopping.** A budget stop is `partial(budget)`: nothing is recorded as a failure, the refused units
  stay pending, and a resume with a larger budget finishes them.

## 12. Retrying failed results

- **Who is retried.** A failed result is retried when its failure is retryable *and* another try can
  differ: transport failures (`rate_limited`, `provider_unavailable`, `timeout`, `connection`), a
  target timeout, and units cut short by a deadline or a budget. **Never**: `evaluator.invalid_output`
  (already retried once with the validation error fed back; at temperature 0 a repeat reproduces it),
  systemic failures (auth, bad configuration, quota, rejected request: fix, don't retry), input
  failures, and target exceptions or contract violations. A `reuse` target's inherited failures are
  retried at the source run, not the copy.
- **What is repeated.** Only failed units. A successful case result, a successful evaluator result and
  every successful provider call are never repeated (tested by counting provider calls per case across
  interruptions and retries). A unit is retried at most `retry_failed_rounds` (3) times over the life
  of the run, so repeated invocations cannot spend without bound.
- **History.** The failed result is archived (`result_history`) before it is replaced; its attempts
  stay on the same row with numbering continuing, each stamped with its `retry_round`. Scores and
  coverage always describe the *current* results; `runs failures --history` and the report show what
  was superseded.
- **Effect on a finished run.** Retrying can move a `succeeded` run's scores (coverage rises, a metric's
  mean moves). It is reopened, finished again, snapshotted again, and the change is recorded as an
  execution. A comparison made before the retry described the earlier state.
- **What it does not do.** It does not retry within one execution (a unit first failing in this
  execution is retried by the next `--retry-failed`), and it cannot make a non-deterministic judge
  agree with itself.

## 13. The environment snapshot and comparison

A run records what defines its measurement (dataset identity, target and judge provider / model /
generation settings, evaluator, prompt and scoring versions, retry and timeout settings, cache and
budget configuration, the price table) and, per execution, the runtime that made the calls (EvalKit
version and commit, Python, platform, SDK versions, endpoints). No credentials, no prompts, no URLs
beyond a host.

`compare` treats a difference as a **confounder** (refused unless `--allow-confounders`, and a gate
failure) only when it can change what is measured or how invalid answers are handled: a different
**endpoint** (a different deployment can answer differently under the same model id) or a different
**`validation_retries`**. It reports as information, not confounders: EvalKit / Python / SDK / SQLite
versions and the commit (scoring changes are tracked by `scoring_version`, which is a confounder),
price tables (cost estimates are not comparable), transport attempts, deadlines and timeouts (they show
up as coverage). A run created before this snapshot existed, or a client that reports no endpoint, is
*unknown*, never different.

