# evalkit

A small, reusable **LLM-as-judge** evaluation library and CLI.

Every AI project ends up re-implementing the same eval plumbing: a judge prompt, score
parsing, a pass/fail rule, somewhere to store results, and a way for humans to disagree.
`evalkit` does that once, application-agnostic (chatbots, agents, RAG answers,
summarization, extraction, code generation), so projects just install it.

- The **LLM judge** only produces per-criterion `{reasoning, score}` via forced tool use.
- **Python** validates every score, normalizes, weights, and decides PASS/FAIL.
- **SQLite** stores every judge-stage run (successful *and* failed) plus any number of human
  reviews per evaluation.
- The **CLI** is a thin wrapper over the library.

Beyond scoring one record, evalkit is a small **single-process evaluation platform** backed by
SQLite: versioned datasets, runs over them, pluggable targets and evaluators (deterministic checks,
retrieval metrics, an LLM judge), honest aggregation, paired run-vs-run comparison with regression
gates, human calibration, and a self-contained HTML report. See
[Evaluation platform](#evaluation-platform). It is *not* a distributed system, a dashboard or an
HTTP service (see [Limitations](#limitations)). Providers: **AWS Bedrock** (Converse) and Bedrock's
OpenAI-compatible endpoint. Design details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/TARGET-ARCHITECTURE.md`](docs/TARGET-ARCHITECTURE.md).

## Install

Requires Python 3.12+.

```bash
# from another project
uv add "evalkit @ git+https://github.com/subhadippatraa/EvalKit.git"
# or
pip install "git+https://github.com/subhadippatraa/EvalKit.git"

# for development
git clone https://github.com/subhadippatraa/EvalKit.git && cd EvalKit
uv sync            # creates .venv with runtime + dev dependencies
```

Runtime dependencies: `pydantic`, `boto3`. `openai` is an optional extra
(`evalkit[bedrock-openai]`), needed only for the `bedrock-openai` provider.

## Configuring the judge

`Evaluator.from_env()` reads these variables (see [`.env.example`](.env.example)):

| Variable | Default | |
|---|---|---|
| `EVALKIT_JUDGE_PROVIDER` | `bedrock` | `bedrock` (boto3 Converse) or `bedrock-openai` (Bedrock's OpenAI-compatible gateway) |
| `EVALKIT_JUDGE_MODEL` | — | **Required.** Model ID/name for the chosen provider |
| `EVALKIT_JUDGE_TEMPERATURE` | `0` | |
| `EVALKIT_JUDGE_TIMEOUT` | `60` | Seconds per request |
| `EVALKIT_DB_PATH` | `./evalkit.db` | SQLite file |
| `EVALKIT_JUDGE_API_KEY` | — | **Required** for `bedrock-openai` |
| `EVALKIT_JUDGE_BASE_URL` | Bedrock's Mantle gateway | `bedrock-openai` only |

evalkit does not load `.env` files; export the variables yourself, e.g.
`set -a; . ./.env; set +a`.

### Provider: `bedrock` (default)

Uses `boto3`'s Bedrock Converse API with a forced tool call. AWS credentials and region
come from the standard AWS chain (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`, instance roles, …); evalkit never reads AWS secrets itself. Use
`AWS_DEFAULT_REGION`, not `AWS_REGION` — boto3's client-construction chain only reads the
latter inside a Lambda environment, so setting only `AWS_REGION` elsewhere fails with
`NoRegionError` unless a region is also set in `~/.aws/config`.

**Model requirement:** the judge forces a tool call (`toolChoice: {"tool": ...}`). Use a
model family that supports forced tool choice (e.g. Anthropic Claude models on Bedrock) and
that your AWS account has access to. Other models, or an account without Converse access,
fail with a `JudgeError` (`ValidationException`).

### Provider: `bedrock-openai`

Uses Bedrock's OpenAI-compatible endpoint (chat completions + forced function calling) via
the `openai` SDK, for models reachable through that gateway (e.g. `openai.gpt-oss-120b`) but
not necessarily through Converse. Install the extra:

```bash
uv sync --extra bedrock-openai   # or: pip install "evalkit[bedrock-openai]"
```

```bash
export EVALKIT_JUDGE_PROVIDER=bedrock-openai
export EVALKIT_JUDGE_MODEL=openai.gpt-oss-120b
export EVALKIT_JUDGE_API_KEY=<bedrock-api-key>
# export EVALKIT_JUDGE_BASE_URL=https://bedrock-mantle.<region>.api.aws/v1  # defaults to us-east-1
```

The `openai` package is an optional dependency; importing `evalkit` never loads it unless
this provider is selected.

### Retries (both providers)

The provider SDK's own automatic retries are disabled. evalkit retries **once**, and only
when the judge output is malformed/invalid. Provider errors, throttling and timeouts are not
retried; they are stored as `status=error` and raised.

## Python usage

```python
from evalkit import Evaluator, JudgeError

evaluator = Evaluator.from_env()  # BedrockJudge + SQLiteStore

try:
    result = evaluator.evaluate(
        prompt="Explain dependency injection in .NET.",
        model_output="DI in .NET is built into Microsoft.Extensions.DependencyInjection ...",
        reference_output="Constructor injection via IServiceCollection ...",  # optional
        context=None,  # optional: supporting material the model output may rely on
        criteria={"correctness": "Is it factually correct?", "clarity": "Is it clear?"},
        metadata={"app": "ragforge", "app_version": "1.4.2", "model": "gpt-x"},
        tags=["regression-suite"],
    )
    print(result.verdict, result.overall_score)          # PASS 0.875
    print(result.scores["clarity"].reasoning)
except JudgeError as e:  # includes JudgeTimeoutError and JudgeOutputError
    print("judge failed; stored as", e.evaluation_id)
```

Explicit construction (no env vars):

```python
from evalkit import Evaluator
from evalkit.bedrock import BedrockJudge
from evalkit.store import SQLiteStore

evaluator = Evaluator(
    judge=BedrockJudge(model="<bedrock-model-id>", temperature=0, timeout=30, region="us-east-1"),
    store=SQLiteStore("evals.db"),  # store=None -> no persistence
)
```

Other calls: `evaluator.get(id)` (includes reviews), `evaluator.list(tag=None, limit=20)`,
`evaluator.review(...)`.

### Errors

| Error | When | Persisted? |
|---|---|---|
| `RubricError` | invalid inputs or rubric (checked before the judge is called) | no |
| `JudgeOutputError` | missing/malformed/out-of-range judge output, after one retry | yes, `status=error` |
| `JudgeTimeoutError` | provider timeout (not retried) | yes, `status=error` |
| `JudgeError` | any other provider failure (not retried) | yes, `status=error` |
| `ScoringError` | internal scoring inconsistency (a `JudgeError`; never retried) | yes, `status=error` |
| `StoreError` | judging succeeded but saving failed; `.result` holds the result, also spilled to `<db>.spill/` (`evalkit recover`) | no (spilled) |
| `MigrationError` | database schema unrecognized, newer than this evalkit, or a migration failed | no |
| `ConfigError` | bad/missing env config, or no judge/store configured | no |
| `EvalKitError` | base class; also "not found" and invalid reviews | — |

Judge errors carry `.evaluation_id` pointing at the stored error row.

## Rubrics

A plain `criteria={"name": "question"}` dict becomes the default rubric: each criterion
scored on an integer **1–5** scale, weight **1.0**, pass threshold **0.75**.

For scales, weights or a different threshold, pass a `Rubric`:

```python
from evalkit import Criterion, Rubric

rubric = Rubric(
    criteria=[
        Criterion(name="faithfulness", description="Is every claim supported by the context?",
                  scale=(1, 5), weight=3),
        Criterion(name="cites_sources", description="Does it cite a source?", scale=(0, 1)),
        Criterion(name="tone", description="Is the tone professional?", scale=(1, 10)),
        # a categorical judgment instead of a Likert scale: exactly one of `scale`/`labels`
        Criterion(name="support", description="Is the claim supported by the source?",
                  labels=("CONTRADICTED", "NOT_ADDRESSED", "SUPPORTED")),
    ],
    threshold=0.8,
    version="rag-v2",   # optional; defaults to a 12-char content hash
)
result = evaluator.evaluate(prompt=..., model_output=..., rubric=rubric)
```

Scoring, done entirely in Python:

1. Output is validated **strictly**. Every criterion must be present, with no extra ones, and
   non-empty `reasoning`. A `scale` criterion needs an **integer** `score` within its scale
   (`"4"`, `4.0` and `true` are rejected); a `labels` criterion needs a `label` that's exactly
   one of its declared values.
2. Each score is normalized to 0–1: a `scale` criterion as `(score − min) / (max − min)`; a
   `labels` criterion by the label's position in the declared (worst → best) order, the same
   way: `index / (len(labels) − 1)`.
3. `overall_score = Σ weightᵢ·normᵢ / Σ weightᵢ`.
4. `verdict = "PASS" if overall_score >= threshold else "FAIL"`.

Every result stores the rubric, its `rubric_version`, and the `judge_prompt_version`, a hash
of the judge prompt template. Scores from different rubric or prompt versions shouldn't be
compared directly.

## Integrity guarantees (0.2.0)

- **Scores are always finite and coherent.** Weights are bounded (`1e-6`..`1e6`); an `ok` result
  always has a score in `[0, 1]` and a verdict that follows from it, enforced in Python and by a
  database trigger.
- **Evaluated content reaches the judge byte-for-byte**, between marker-delimited blocks that
  the content cannot forge (no HTML escaping). Rubric text is kept in a separate trusted section.
- **Every judge call is recorded** in `result.attempts`: a result that succeeded on its retry shows
  the rejected first attempt and its (truncated) raw output. Provider errors are scrubbed of ARNs/keys.
- **A rubric `version` label always means one rubric.** Reusing a label for different content raises
  `RubricError` before the judge is called. Omit `version` to use the content hash.
- **Inputs are bounded** (256 KiB per text field, 16 KiB metadata, 32 tags/criteria; see
  `evalkit.Limits`) and validated before any paid call.
- **Databases migrate themselves.** Opening an older database upgrades it (a `.bak-*` copy is
  written first); unrecognized or newer databases are refused untouched. New files are `0600`, WAL mode.
- **A paid result is not lost when saving fails**: `StoreError.result` plus a spill file; run
  `evalkit recover` (or `evaluator.recover_spilled()`) once the database is healthy.

## Datasets

Versioned, immutable collections of evaluation cases (the foundation for comparing runs; runs and
targets come next). One JSON object per line:

```json
{"case_key": "q1", "prompt": "What is 2+2?", "output": "4", "reference": "4", "tags": ["math"]}
{"case_key": "q2", "prompt": "Capital of France?", "output": "Paris", "retrieved": ["d1", "d2"], "relevance": {"d1": 2, "d2": 0}, "metadata": {"source": "faq"}}
```

Only `case_key` (stable id, `[A-Za-z0-9._:/-]`, up to 128) and `prompt` are required; `output`,
`reference`, `context`, `retrieved` (ranked doc ids), `relevance` (doc id to grade), `metadata`,
`tags` are optional. Unknown fields are refused.

```bash
evalkit dataset import support-qa cases.jsonl     # creates support-qa@1
evalkit dataset import support-qa cases.jsonl     # identical content: same version, "created": false
evalkit dataset list
evalkit dataset show support-qa@latest --cases 3  # also: support-qa@2, support-qa@<hash prefix>
evalkit dataset lint support-qa                   # duplicates, empty fields, missing labels
evalkit dataset verify support-qa                 # recompute hashes from stored rows
evalkit dataset export support-qa cases-copy.jsonl
```

```python
from evalkit import EvalKit

kit = EvalKit.open("evalkit.db")                  # or EvalKit.from_env()
result = kit.datasets.import_jsonl("support-qa", "cases.jsonl")   # or import_cases([...])
print(result.version.ref, result.version.content_hash, result.created)
for case in kit.datasets.cases("support-qa@latest"):
    ...
```

- **Import is all-or-nothing** and reports every problem with its line number; a bad file stores
  nothing (not even the dataset).
- **A version is immutable and content-addressed.** Changing any case creates the next version;
  older versions never change. Case order in the file does not matter, nor does the order of `tags`.
- **Content is stored losslessly** (no escaping or normalization) and hashed canonically; the
  database itself refuses to update or delete sealed data, and `dataset verify` detects tampering.

## Evaluation platform

```text
Dataset ─► DatasetVersion (immutable) ─► Run (frozen config) ─► per case: Target ─► CaseResult
                                                                        └─► Evaluator(s) ─► EvaluatorResult (+ metrics, attempts)
Run ─► summary (coverage, CIs, failures) ─► compare (paired, gated) ─► static HTML report
```

A quick tour lives in [`examples/quickstart`](examples/quickstart) (`bash run.sh`, also under test).

**Run.** `evalkit runs create DATASET --config spec.toml` runs *preflight* (every case has what each
evaluator needs; bad config; unpinned targets; dataset lint; contamination), freezes the config
against one exact dataset version, and plans one pending result per case. `evalkit runs execute RUN`
executes it: a bounded window of cases on a thread pool, one writer committing batches, cases in a
seeded hash order (a run that stops early has processed an unbiased sample, not a prefix).

```toml
# spec.toml
[target]                       # precomputed (default) | reuse | callable | model
kind = "callable"
callable = "mypkg.rag:answer"  # trusted code, imported from this file/flag only
fingerprint = "git:3f2a9c1"    # you declare what you test; without it the run is "unpinned"

[[evaluators]]
kind = "exact_match"           # exact_match | regex | json_schema | retrieval | citation_check | llm_judge
name = "answer"
params = { normalize = ["strip", "casefold"] }

[[evaluators]]
kind = "llm_judge"
name = "quality"
params = { rubric = { criteria = [{ name = "correctness", description = "Is it correct?", must_pass = true }] } }

[policy]                       # execution only: never part of a run's identity
concurrency = 4
retry = { max_attempts = 4, base_s = 1.0, cap_s = 30.0, timeout_s = 60.0 }
budget = { max_tokens = 2000000, max_calls = 50000, max_duration_s = 3600 }
```

**Targets** produce the output being scored. `precomputed`: the dataset's own `output`.
`reuse`: outputs copied from an earlier run (re-score with new evaluators at zero target cost).
`callable`: a Python function `(TargetInput) -> str | TargetOutput`. `model`: a prompt template over
an LLM client. A target sees only `prompt`/`context`: **never the reference, relevance labels, tags or
metadata**. Every external call is an *attempt* (latency, tokens, request id, error, kept evidence).

**Evaluators.** `exact_match` (declared normalization), `regex`, `json_schema` (optional extra:
`pip install 'evalkit[jsonschema]'`; remote `$ref` is refused), `retrieval` (recall/precision/hit/MRR/nDCG
exactly as specified in the design), `citation_check` (structure only: it does not judge whether a
cited document supports a claim), `llm_judge` (rubric + must-pass criteria). There is deliberately no
BLEU/ROUGE/embedding-similarity evaluator. An evaluator's identity (`kind:name:hash`) covers every
parameter, its version and, for judges, the rubric hash, model and prompt fingerprint.

**Failures are attributed, never scored.** Every failure has a class: `input` (case/config),
`target` (the system under test), `evaluator` (rubric/judge), `infrastructure` (provider, storage,
EvalKit). A target failure is a case-result failure and its evaluators are `skipped`; nothing becomes a
zero. `evalkit runs failures RUN --class target` is the AI system's problem list.

**Aggregation states its denominators.** Per evaluator: scored / not applicable / skipped /
evaluator-failed / infrastructure-failed / input-excluded / missing, `coverage = scored / (cases −
not_applicable)`, means with intervals (Wilson for proportions, seeded bootstrap otherwise), pass rate
with bounds for UNCERTAIN verdicts, latency, tokens, slices by tag. **Below the required coverage the
headline value is withheld** (the observed mean stays visible and flagged).

**Stopping and resuming.** Ctrl-C (or `CancelToken`) stops dispatch and lets in-flight cases finish;
budgets, a tripped provider breaker and systemic failures (auth, bad model id) stop the run as
`cancelled` / `partial` / `failed`, keeping every result. `evalkit runs resume RUN` finishes what is
left, including cases whose evaluators were interrupted by a crash. If storage itself fails,
unwritten results are spilled to `<db>.spill/<run>.jsonl` and replayed on the next execute.

**Comparing runs fairly.** `evalkit compare CAND --baseline RUN|tag:main [--gates gates.toml]`
pairs results by case. It refuses a *confounded* comparison (different judge model, rubric, prompt
fingerprint, evaluator parameters, scoring version) unless `--allow-confounders`; reports excluded
cases and a survivorship warning; and gives per metric n, the paired difference with a 95% interval,
the minimum detectable difference, higher/lower/tied counts and (for proportions) an exact McNemar
p-value. Each declared gate decides REGRESSION / IMPROVEMENT / EQUIVALENT / INCONCLUSIVE from the
interval against your tolerance; fewer than 30 pairs is always INCONCLUSIVE, and a candidate below
the minimum coverage cannot pass. Exit codes: `0` pass, `1` error, `2` usage, `3` gate failed, `4`
inconclusive under `--strict`.

```toml
# gates.toml: only declared gates can fail a build
min_coverage = 0.95
[gates]
"exact_match:answer.match" = { direction = "higher", delta = 0.02 }
"run.target_failure_rate"  = { direction = "lower",  delta = 0.005 }
"run.latency_p95_ms"       = { direction = "lower",  rel_delta = 0.25 }
[gates.floor]
"exact_match:answer.match" = 0.90
```

**A judge is an instrument, not truth.** `evalkit runs review ... --sample random` records human
verdicts; `runs calibrate` reports accuracy, FAIL-precision/recall, Cohen's kappa and reviewer
agreement from random-sample reviews only (`UNCALIBRATED` below 30). `runs queue` offers unbiased
(`random`, `stratified`) or triage (`uncertain`, `disagreement`) review queues, and
`runs disagreements` compares evaluators per case. `evalkit judge-check` runs a golden set with
adversarial cases (hidden instructions, verbosity padding, forged delimiters, injected context)
through the configured judge and stores its accuracy per evaluator key.

**Report.** `evalkit report RUN --out report.html [--compare tag:main --gates gates.toml]` writes one
static file (inline CSS, no JavaScript, no network, CSP `default-src 'none'`); all case text, model
output and error messages are escaped. It shows the run header and frozen config, the summary with
denominators and intervals, the failure breakdown by class, judge trust, the comparison and gate with
its reasoning, and per-case evidence (input, target output, evaluator results, attempts, failures).

```python
from evalkit import EvalKit, EvaluatorSpec
from evalkit.targets import CallableTarget
from evalkit.compare import compare, Gates, evaluate_gates

kit = EvalKit.open("evalkit.db")
kit.datasets.import_jsonl("support-qa", "cases.jsonl")
target = CallableTarget(my_pipeline, name="rag", fingerprint="git:3f2a9c1")
run = kit.controller.create("support-qa", target=target,
                            evaluators=[EvaluatorSpec(kind="exact_match", name="answer")])
report = kit.controller.execute(run.id, target=target)     # report.status, .counts, .spend
summary = kit.summarize(run.id)                             # coverage, metrics with CIs, failures
cmp = compare(kit, baseline_run_id, run.id)
gate = evaluate_gates(kit, run.id, Gates.from_file("gates.toml"), cmp)   # gate.exit_code
```

### Limitations

- **One process, one SQLite file** on a local filesystem. Threads, not workers: no multi-process or
  multi-host execution, no leases. Measured numbers are in [`docs/BENCHMARK.md`](docs/BENCHMARK.md);
  nothing here claims distributed or 1M-case production scale.
- **Not in P1** (designed for P2): retrying *failed* results on resume (`--retry-failed`; terminal
  results are write-once), adaptive concurrency, per-provider limiter tables, the LLM response
  cache, USD-priced budgets (budgets are tokens / calls / duration; cost is an estimate from
  user-supplied prices), `evalkit gc`, samples-per-case self-consistency, run cancellation from another
  process, structured logging/metrics export.
- **Callable targets are trusted code** with full process privileges, are timed out but not killed
  (a timed-out function keeps running in its thread), and may be called twice for one case (a timeout
  is retried once): make them safe to call twice. There is no `http` target.
- **LLM judges can be talked into a score**; injection is made detectable (golden set, calibration,
  deterministic evaluators), not preventable. The built-in golden set is small and hand-labelled.
  **Live validation of the judge prompt against a real model is still pending**: the credentials
  available while developing could not invoke Bedrock models (`ValidationException: Operation not
  allowed`), so the provider clients and prompts are tested against recorded-shape fakes only.
- Statistics are stdlib-only and stated with their limits: a 60-case dataset cannot show a change
  smaller than about 10 points (comparisons print the minimum detectable difference).

## Human review

Any evaluation, including failed ones, can have any number of reviews. A review has a
`reviewer`, a `verdict` (PASS/FAIL), and an optional comment. It can also carry an optional
`score` normalized to **0–1**, so it is directly comparable with the judge's `overall_score`.

```python
evaluator.review(result.id, reviewer="alice", verdict="PASS", score=0.9, comment="agree")
evaluator.review(result.id, reviewer="bob", verdict="FAIL", comment="misses scoped lifetimes")
[r.reviewer for r in evaluator.get(result.id).reviews]   # ['alice', 'bob']
```

## CLI

```bash
evalkit run --input input.json           # evaluate; prints the result as JSON
evalkit get <id>                         # one evaluation + its reviews
evalkit list [--tag X] [--limit N] [--cursor C]   # newest first; next_cursor on stderr
evalkit recover                          # save results a failed database write left behind
evalkit review <id> --reviewer alice --verdict PASS [--score 0.9] [--comment "..."]

evalkit dataset import|list|show|export|lint|verify ...        # versioned datasets
evalkit runs create|execute|resume|list|show|failures|verify ... # runs (platform)
evalkit runs tag|review|calibrate|queue|disagreements ...       # baselines, human calibration
evalkit compare CAND --baseline RUN|tag:NAME [--gates FILE] [--strict] [--allow-confounders]
evalkit report RUN --out report.html [--compare BASELINE] [--gates FILE] [--max-cases N] [--force]
evalkit judge-check [--cases FILE] [--rubric FILE] [--min-accuracy X]
```

`input.json` uses the same keys as `evaluate()`; `rubric` may be given as a JSON object:

```json
{
  "prompt": "Explain dependency injection in .NET.",
  "model_output": "DI in .NET is built into Microsoft.Extensions.DependencyInjection ...",
  "criteria": {"correctness": "Is it factually correct?", "clarity": "Is it clear?"},
  "metadata": {"app": "ragforge", "app_version": "1.4.2"},
  "tags": ["regression-suite"]
}
```

Exit codes:

- `0`: success, including a `FAIL` verdict.
- `1`: evaluation or library error. The message and any stored `evaluation_id` go to stderr.
- `2`: usage error or unreadable input file.
- `3`: a declared gate failed (`compare`, `judge-check --min-accuracy`). `4`: gate inconclusive under `--strict`.
  A run that did not complete (`runs execute` on a partial/failed/cancelled run) exits `1`.

`run` needs judge configuration. `get`, `list` and `review` only need `EVALKIT_DB_PATH`.

## Full example: input → stored result

```bash
export EVALKIT_JUDGE_MODEL=<bedrock-model-id> AWS_DEFAULT_REGION=us-east-1 EVALKIT_DB_PATH=./evals.db
evalkit run --input input.json > result.json
evalkit review "$(jq -r .id result.json)" --reviewer alice --verdict pass --score 0.9
evalkit get "$(jq -r .id result.json)"
```

The output looks like this. The values are illustrative.

```json
{
  "id": "825dc7c6-4c1f-4422-8e67-4f09164e16aa",
  "created_at": "2026-09-19T07:05:08.177691Z",
  "status": "ok",
  "error": null,
  "prompt": "Explain dependency injection in .NET.",
  "model_output": "DI in .NET is built into Microsoft.Extensions.DependencyInjection ...",
  "reference_output": null,
  "context": null,
  "rubric": {
    "criteria": [
      {"name": "correctness", "description": "Is it factually correct?", "scale": [1, 5], "weight": 1.0},
      {"name": "clarity", "description": "Is it clear?", "scale": [1, 5], "weight": 1.0}
    ],
    "threshold": 0.75,
    "version": "8c9f61c100c8"
  },
  "rubric_version": "8c9f61c100c8",
  "judge_provider": "bedrock",
  "judge_model": "<bedrock-model-id>",
  "judge_temperature": 0.0,
  "judge_prompt_version": "e5d44d771d14",
  "scores": {
    "correctness": {"reasoning": "Accurately describes the built-in container ...", "score": 5},
    "clarity": {"reasoning": "Clear, but skips service lifetimes ...", "score": 4}
  },
  "overall_score": 0.875,
  "verdict": "PASS",
  "latency_ms": 2140,
  "metadata": {"app": "ragforge", "app_version": "1.4.2"},
  "tags": ["regression-suite"],
  "reviews": [
    {
      "id": "55b609a0-01e8-4566-a53d-f3bc6e568b9d",
      "evaluation_id": "825dc7c6-4c1f-4422-8e67-4f09164e16aa",
      "reviewer": "alice",
      "verdict": "PASS",
      "score": 0.9,
      "comment": null,
      "created_at": "2026-09-19T07:05:09.032397Z"
    }
  ]
}
```

Storage is two SQLite tables:

- `evaluations`: one row per run, `status` `ok` or `error`.
- `reviews`: many rows per evaluation.

Per-criterion scores, metadata and tags are stored as JSON columns. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the schema.

## Adding a provider or a store

The core depends only on two Protocols, so an extension is one new module. No registration
is needed.

**Judge** (`evalkit.judge.Judge`, also exported as `evalkit.Judge`): expose `provider`,
`model`, `temperature`, `prompt_version`, and implement

```python
def judge(self, prompt, model_output, reference_output, context, rubric) -> dict: ...
```

Rules:

- Make exactly one provider request per call.
- Force a structured/tool-call response using `render_prompt()`, `output_schema(rubric)` and
  `TOOL_NAME` from `evalkit.judge`, and use `PROMPT_VERSION` for `prompt_version`.
- Return the raw payload unmodified. Do **not** validate, score, or retry: `Rubric.score`
  and `Evaluator` do that.
- Raise `JudgeTimeoutError` on timeouts and `JudgeError` on other provider failures. Raise
  `JudgeOutputError` only when the response has no structured payload at all.

Use [`src/evalkit/bedrock.py`](src/evalkit/bedrock.py) or
[`src/evalkit/bedrock_openai.py`](src/evalkit/bedrock_openai.py) as the template, then pass your judge
in with `Evaluator(judge=MyJudge(...), store=...)`.

**Store** (`evalkit.store.Store`): implement `save(result)`, `get(id) -> EvaluationResult |
None` (with `.reviews` populated), `list(tag=None, limit=20)` and `add_review(review)`.

## Running tests

```bash
uv run pytest            # no network or AWS calls: fake judges + stubbed Bedrock client
uv run ruff check . && uv run ruff format --check .
uv run --with coverage coverage run --branch --source=evalkit -m pytest -q && uv run --with coverage coverage report
uv run python scripts/mutation.py     # ~100 targeted mutants of the invariants; every one must be killed
uv run python scripts/benchmark.py --cases 10000    # local throughput; see docs/BENCHMARK.md
```

A real Bedrock smoke test is manual. With AWS credentials and model access configured, run
`evalkit run --input input.json` and confirm you get `status: ok` and a verdict.
