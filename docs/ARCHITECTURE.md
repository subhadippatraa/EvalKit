# evalkit — Architecture (v1)

Status: **approved and implemented (v0.1.0).** Source of
truth for requirements: `evalkit-prompt.md`. All open decisions are resolved
(see [Resolved Decisions](#resolved-decisions)); references appear as **[OD-n]**.

## Goals and scope

A small, reusable Python 3.12 library + CLI for LLM-as-judge evaluation of LLM
outputs, installable into any project (chatbots, agents, RAG, summarization,
extraction, codegen).

- **In scope:** rubric-based judging, code-side scoring and PASS/FAIL, SQLite
  persistence of every judge-stage run (ok and error), multiple human reviews
  per evaluation, thin CLI.
- **Out of scope (v1):** retrieval metrics, auth, multi-tenancy, web UI, HTTP
  API, queues/workers, batch/dataset management, experiment orchestration,
  dashboards, providers other than Bedrock.

Responsibility split:

| Layer | Responsibility |
|---|---|
| LLM judge | Criterion-level `{reasoning, score}` only |
| Python (`Rubric.score`, `Evaluator`) | Validation, normalization, weighted score, verdict, retry |
| Store | Persistence |
| CLI | Argument parsing + calls into the library; no logic |

## Module / file structure

Flat `src/` layout.

```
pyproject.toml
.env.example
README.md
docs/ARCHITECTURE.md
src/evalkit/
  __init__.py       # public exports
  models.py         # Criterion, Rubric (+ .score()), CriterionScore, EvaluationResult, Review
  errors.py         # error hierarchy
  judge.py          # Judge Protocol, PROMPT_TEMPLATE, render_prompt(), output_schema(), PROMPT_VERSION  (no SDK imports)
  bedrock.py        # BedrockJudge — the only module that imports boto3/botocore
  bedrock_openai.py # BedrockOpenAIJudge — the only module that imports openai (added post-v1, see below)
  store.py          # Store Protocol, SQLiteStore
  evaluator.py      # Evaluator (+ from_env): orchestration, retry, persistence
  cli.py            # argparse entry point `evalkit`
tests/
  test_models.py test_evaluator.py test_bedrock.py test_bedrock_openai.py test_store.py
  test_cli.py test_e2e.py
```

- Only two Protocols (`Judge`, `Store`). No registry, factory, provider
  manager, or plugin system.
- `evaluator.py` depends only on the `Judge` Protocol. `Evaluator.from_env()`
  is the single place that names a concrete judge class, chosen by
  `EVALKIT_JUDGE_PROVIDER`; each provider module is imported lazily, only when
  selected, so its SDK is never a hard dependency of the core.
- A future provider (e.g. `AnthropicJudge`) is one more module like
  `bedrock.py`/`bedrock_openai.py`, reusing `judge.py`.

### Addition beyond original v1 scope: `bedrock-openai` provider

The original v1 scope (and [OD-6](#resolved-decisions)) said "v1 implements
only `BedrockJudge`". A second judge, `BedrockOpenAIJudge`, was added on
explicit request to reach Bedrock through its OpenAI-compatible gateway
(chat completions + forced function calling) instead of the boto3 Converse
API — needed because this account's boto3 `Converse` calls were blocked
(`ValidationException: Operation not allowed`) while the gateway path works.
It follows the same isolation rule as `bedrock.py`: it is the only module
that imports `openai`, it reuses the shared prompt/schema/version from
`judge.py`, and `openai` is an optional install extra (`evalkit[bedrock-openai]`),
not a core dependency. No registry was added — `Evaluator.from_env()` just
branches on `EVALKIT_JUDGE_PROVIDER` between the two.

## Public API

Exports from `evalkit`: `Evaluator`, `Rubric`, `Criterion`, `EvaluationResult`,
`Review` **[OD-3]**, and the error classes.

```python
class Evaluator:
    def __init__(self, judge: Judge | None, store: Store | None = None): ...
    @classmethod
    def from_env(cls, *, with_judge: bool = True) -> "Evaluator": ...  # BedrockJudge + SQLiteStore
    def evaluate(self, prompt: str, model_output: str,
                 reference_output: str | None = None,
                 context: str | None = None,
                 criteria: dict[str, str] | None = None,
                 rubric: Rubric | dict | None = None,
                 metadata: dict | None = None,
                 tags: list[str] | None = None) -> EvaluationResult: ...
    def review(self, evaluation_id: str, reviewer: str, verdict: str,
               score: float | None = None, comment: str | None = None) -> Review: ...
    def get(self, evaluation_id: str) -> EvaluationResult: ...
    def list(self, tag: str | None = None, limit: int = 20) -> list[EvaluationResult]: ...
```

- Exactly one of `criteria` / `rubric` is required; otherwise `RubricError`.
- `store=None` disables persistence. `review`/`get`/`list` then raise
  `ConfigError`. `from_env()` always attaches `SQLiteStore`.
- `rubric` may be a `Rubric` or its dict form (as used by the CLI input file).
- `judge=None` gives a store-only evaluator (`get`/`list`/`review`); `evaluate`
  then raises `ConfigError`. `from_env(with_judge=False)` builds one without
  requiring `EVALKIT_JUDGE_MODEL` or AWS configuration. The CLI uses it for
  `get`/`list`/`review`, so those commands need only `EVALKIT_DB_PATH`.
  *(Added during implementation.)*
- `BedrockJudge` is importable as `evalkit.bedrock.BedrockJudge` for explicit
  construction; it is not a top-level export.
- `context`: optional supporting material the model output may be checked
  against (e.g. source documents, a spec, background text) -- distinct from
  `reference_output` (a gold *answer*, not grounding material). Rendered into
  its own delimited `context` block, encoded like `model_output`/
  `reference_output` (see [0.2.0 changes](#020-p0-hardening-changes)); omitted
  (`None`) is stated outside any block as `context: (not provided)`. Persisted on `EvaluationResult` for
  the same auditability reason `reference_output` is. *(Added during
  implementation, generic scope -- not domain-specific.)*

## Pydantic / domain models (`models.py`)

| Model | Fields |
|---|---|
| `Criterion` | `name: str`, `description: str`, `scale: tuple[int, int] \| None`, `labels: tuple[str, ...] \| None` (exactly one of `scale`/`labels` ends up set -- both omitted defaults to `scale=(1, 5)`, the pre-`labels` behavior), `weight: float = 1.0` |
| `Rubric` | `criteria: list[Criterion]`, `threshold: float = 0.75`, `version: str \| None` |
| `CriterionScore` | `reasoning: str`, `score: int \| None`, `label: str \| None` (exactly one of `score`/`label` set; field order: reasoning first) |
| `EvaluationResult` | `id`, `created_at`, `status: "ok" \| "error"`, `error: str \| None`, `prompt`, `model_output`, `reference_output`, `context`, `rubric`, `rubric_version`, `judge_provider`, `judge_model`, `judge_temperature`, `judge_prompt_version`, `scores: dict[str, CriterionScore]`, `overall_score: float \| None` (0–1), `verdict: "PASS" \| "FAIL" \| None`, `latency_ms`, `metadata: dict`, `tags: list[str]`, `reviews: list[Review]` (populated on `get`) |
| `Review` | `id`, `evaluation_id`, `reviewer: str`, `verdict: "PASS" \| "FAIL"`, `score: float \| None` in `[0, 1]` **[OD-2]**, `comment: str \| None`, `created_at` |

IDs: uuid4 strings. Timestamps: ISO-8601 UTC.

## Rubric and Criterion design

- `Criterion.scale` is an integer range with `min < max`; scores are integers.
- `Criterion.weight > 0`.
- Criterion names: unique, non-empty (used as JSON keys in the judge schema).
- `Rubric.threshold` in `[0, 1]`, applied to the normalized overall score.
  Single rubric-level threshold; no per-criterion thresholds.
- `Rubric.version`: explicit string, or `sha256(canonical JSON of rubric)[:12]`
  when omitted.
- `Rubric.from_dict({name: description})` builds the default rubric
  (scale 1–5, weight 1.0, threshold 0.75) for plain `dict[str, str]` criteria.
- `Rubric.score(raw: dict) -> (scores, overall_score, verdict)` is the **single
  place** judge output is validated and scored (see below).

## Judge Protocol (`judge.py`)

```python
class Judge(Protocol):
    provider: str          # e.g. "bedrock"
    model: str             # provider-specific model id, opaque to the core
    temperature: float
    prompt_version: str
    def judge(self, prompt: str, model_output: str, reference_output: str | None,
              context: str | None, rubric: Rubric) -> dict[str, Any]: ...
```

Contract for any implementation:

- Returns the **raw** structured payload (the tool input object) unmodified.
  It does not coerce, validate ranges, score, or retry.
- Raises `JudgeTimeoutError` on timeout, `JudgeError` on any other provider
  failure, `JudgeOutputError` only when the response contains no structured
  payload at all (e.g. no tool call, truncated response).
- Makes exactly one provider request per call (no SDK-level retries) **[OD-1]**.

Provider-agnostic helpers, shared by all current and future judges:

- `PROMPT_TEMPLATE`: instructions + prompt, model output, optional context,
  optional reference output, criterion descriptions with scales.
- `render_prompt(prompt, model_output, reference_output, context, rubric) -> str`.
- `output_schema(rubric) -> dict`: the JSON Schema below.
- `PROMPT_VERSION = compute_prompt_version()`: a hash of the static prompt text **and** of a fixed
  fixture rendered through the real `render_prompt`/`output_schema` code, so a change to rendering
  behavior changes the version (0.2.0; before, only string constants were hashed).

## Bedrock judge integration (`bedrock.py`)

- Uses `boto3` `bedrock-runtime` **Converse API**. The only module importing
  `boto3`/`botocore`.
- `BedrockJudge(model: str, temperature: float = 0.0, timeout: float = 60,
  region: str | None = None, client=None)`. `client` is injectable for tests.
- `provider = "bedrock"`; `model` is a Bedrock model ID or inference profile ID.
- Client config: `botocore.config.Config(retries={"total_max_attempts": 1},
  read_timeout=timeout, connect_timeout=timeout)` → no SDK retries **[OD-1]**.
- Request: `system` + one user message from `render_prompt`;
  `toolConfig = {"tools": [{"toolSpec": {"name": TOOL_NAME, "description": ...,
  "inputSchema": {"json": output_schema(rubric)}}}],
  "toolChoice": {"tool": {"name": TOOL_NAME}}}`;
  `inferenceConfig = {"temperature": temperature, "maxTokens": MAX_TOKENS}`.
- Response: return the `toolUse.input` of the `submit_evaluation` block. No
  such block, or `stopReason == "max_tokens"` → `JudgeOutputError`.
- Error mapping: `ReadTimeoutError`, `ConnectTimeoutError`, and `ClientError`
  code `ModelTimeoutException` → `JudgeTimeoutError`; any other
  `ClientError` / `BotoCoreError` → `JudgeError` (message includes AWS error
  code).
- Credentials and region come from the standard AWS chain (`AWS_PROFILE`,
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, instance
  role…). evalkit reads no AWS secrets itself. Use `AWS_DEFAULT_REGION`, not
  `AWS_REGION` — this botocore version's client-construction chain only reads
  the latter inside a Lambda environment; found via a CI failure (`NoRegionError`)
  that a local `~/.aws/config` default region had been masking.

## Structured judge-output contract

`output_schema(rubric)`:

```json
{
  "type": "object",
  "properties": {
    "<criterion_name>": {
      "type": "object",
      "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "integer", "minimum": <scale_min>, "maximum": <scale_max>}
      },
      "required": ["reasoning", "score"],
      "additionalProperties": false
    }
  },
  "required": ["<every criterion name>"],
  "additionalProperties": false
}
```

`reasoning` precedes `score` in property order. The schema guides the model;
Python re-validates everything and never trusts schema conformance.

## Code-based scoring and normalization

In `Rubric.score(raw)`:

1. Validate `raw` **strictly** (no coercion: `"4"`, `4.0`, `true` are
   rejected). Reject missing criteria, extra criteria, missing/empty
   `reasoning`, non-integer scores, and scores outside the criterion's scale
   → `JudgeOutputError`.
2. Normalize per criterion: `n_i = (score_i − min_i) / (max_i − min_i)`.
3. Overall: `sum(w_i * n_i) / sum(w_i)` → `[0, 1]`.
4. Verdict: `PASS` if `overall ≥ threshold`, else `FAIL`.

The LLM never produces the overall score or verdict.

## Error and failure semantics

```
EvalKitError
├── ConfigError          # missing/invalid env config, no store configured
├── RubricError          # invalid rubric or evaluate() inputs
└── JudgeError           # provider/API/connection failure
    ├── JudgeTimeoutError
    └── JudgeOutputError # missing/malformed/out-of-range judge output
```

| Failure | evalkit retry | Persisted | Raised |
|---|---|---|---|
| Invalid inputs / rubric | no | **no** [OD-4] | `RubricError` |
| Malformed judge output | **once**, fresh identical request from `Evaluator` [OD-5] | yes, `status=error` | `JudgeOutputError` (after the retry also fails) |
| Provider error | **no** [OD-1] | yes, `status=error` | `JudgeError` |
| Timeout | **no** [OD-1] | yes, `status=error` | `JudgeTimeoutError` |
| Provider error/timeout during the malformed-output retry | no | yes, `status=error` | that error |
| Storage failure after judging | no | result spilled to `<db>.spill/` | `StoreError` (an `sqlite3.Error`) carrying `.result` |

- A malformed result is never accepted or stored as `ok`.
- Raised judge errors carry `.evaluation_id` of the persisted error row.
- Error rows store inputs, rubric/version, judge config, `latency_ms`, the
  error message; score fields are NULL.

## SQLite schema and relationships

stdlib `sqlite3`. Tables created on first connect. `PRAGMA foreign_keys=ON`.

```sql
CREATE TABLE evaluations (
  id                   TEXT PRIMARY KEY,
  created_at           TEXT NOT NULL,
  status               TEXT NOT NULL CHECK (status IN ('ok','error')),
  error                TEXT,
  prompt               TEXT NOT NULL,
  model_output         TEXT NOT NULL,
  reference_output     TEXT,
  context              TEXT,
  rubric_json          TEXT NOT NULL,
  rubric_version       TEXT NOT NULL,
  judge_provider       TEXT NOT NULL,
  judge_model          TEXT NOT NULL,
  judge_temperature    REAL NOT NULL,
  judge_prompt_version TEXT NOT NULL,
  scores_json          TEXT,
  overall_score        REAL,
  verdict              TEXT CHECK (verdict IN ('PASS','FAIL')),
  latency_ms           INTEGER,
  metadata_json        TEXT NOT NULL DEFAULT '{}',
  tags_json            TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX idx_evaluations_created_at ON evaluations(created_at);
```

- NOT NULL on rubric/judge columns is valid because invalid inputs are never
  persisted **[OD-4]**; every stored row reached the judge stage.
- Per-criterion scores + reasoning live in `scores_json`.
- Relationship: `evaluations 1 ─── * reviews`.

## Human review schema

```sql
CREATE TABLE reviews (
  id            TEXT PRIMARY KEY,
  evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
  reviewer      TEXT NOT NULL,
  score         REAL CHECK (score BETWEEN 0 AND 1),
  verdict       TEXT NOT NULL CHECK (verdict IN ('PASS','FAIL')),
  comment       TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX idx_reviews_evaluation_id ON reviews(evaluation_id);
```

- `score` is an optional overall human score normalized to `[0, 1]`, directly
  comparable with `evaluations.overall_score` **[OD-2]**. Validated in the
  `Review` model and by the CHECK constraint.
- Multiple reviews per evaluation; append-only.
- Reviewing a non-existent evaluation raises `EvalKitError` (explicit check;
  FK as backstop).
- Reviews may be added to `status=error` evaluations (e.g. to annotate a
  failure).
- `Evaluator.get()` returns the evaluation with its reviews.

## Metadata and tags

- `metadata`: JSON-serializable dict, stored as `metadata_json`. Not
  interpreted by evalkit. Non-serializable values → `RubricError` at input
  validation.
- `tags`: list of strings, de-duplicated (order preserved), stored as
  `tags_json`.
- `list --tag X` filters with
  `EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)`. No tags table.

## CLI design

`argparse`, entry point `evalkit = evalkit.cli:main`. Each command builds an
`Evaluator` via `from_env()` (`with_judge=False` for `get`/`list`/`review`) and
makes one library call; output is JSON on stdout.

| Command | Library call |
|---|---|
| `evalkit run --input file.json` | `evaluate(**json)` (same keys as `evaluate()`; `rubric` as a dict) |
| `evalkit get <id>` | `get(id)` |
| `evalkit list [--tag X] [--limit N]` | `list(tag, limit)` |
| `evalkit review <id> --reviewer R --verdict PASS\|FAIL [--score S] [--comment C]` | `review(...)` |

Exit codes: `0` success (including verdict `FAIL`), `1` evaluation/library
error (error row id printed when one was persisted), `2` usage error.

## Configuration / environment variables

| Variable | Default | Required by `from_env()` | Purpose |
|---|---|---|---|
| `EVALKIT_JUDGE_PROVIDER` | `bedrock` | no | `bedrock` or `bedrock-openai` (§ below); anything else → `ConfigError` |
| `EVALKIT_JUDGE_MODEL` | — | **yes** [OD-6] | Model ID/name for the chosen provider |
| `EVALKIT_JUDGE_TEMPERATURE` | `0` | no | Judge temperature |
| `EVALKIT_JUDGE_TIMEOUT` | `60` | no | Per-request timeout (seconds) |
| `EVALKIT_DB_PATH` | `./evalkit.db` | no | SQLite path |
| `AWS_DEFAULT_REGION`, `AWS_PROFILE` / AWS credentials | AWS chain | via AWS chain | Read by boto3, `bedrock` provider only; **not** `AWS_REGION` (unread outside Lambda) |
| `EVALKIT_JUDGE_API_KEY` | — | **yes**, `bedrock-openai` only | API key for Bedrock's OpenAI-compatible gateway |
| `EVALKIT_JUDGE_BASE_URL` | `bedrock_openai.DEFAULT_BASE_URL` | no | Override the gateway URL, `bedrock-openai` only |

- Missing `EVALKIT_JUDGE_MODEL` or unparsable numbers → `ConfigError`.
- `.env.example` documents these with placeholder values; no secrets
  committed. No `python-dotenv`.
- `from_env()` is the only code that reads environment variables.

## Dependencies

- Runtime: `pydantic>=2`, `boto3`.
- Optional extra `bedrock-openai`: `openai>=1.50`, only for that provider.
- Dev: `pytest`, `ruff`, `openai` (so `bedrock-openai` tests can run against a
  fake client without a real account).
- Stdlib for everything else: `argparse`, `sqlite3`, `hashlib`, `uuid`, `json`.

## Testing strategy

pytest, no network or AWS calls.

- `FakeJudge` (test-local class implementing `Judge`) returns scripted raw
  payloads or raises scripted errors. Used by evaluator, CLI, and e2e tests.
- `BedrockJudge` tested with an injected fake client whose `converse()`
  returns canned Converse responses or raises `ReadTimeoutError` /
  `ClientError`. Asserts request shape (forced `toolChoice`, schema,
  temperature) and error mapping.
- `SQLiteStore` tested with `tmp_path` databases.

Coverage:

- Model validation: criterion scale/weight, unique names, threshold,
  criteria-vs-rubric exclusivity, rubric version hashing, review score range.
- Scoring: normalization, weights, mixed scales, threshold boundary.
- Malformed output: missing/extra criterion, out-of-range, non-integer,
  coercible strings; retry once then `JudgeOutputError`; success on retry;
  exactly two judge calls max.
- Provider error and timeout: **no** retry (one judge call), typed error,
  persisted error row with `.evaluation_id`.
- Invalid inputs: `RubricError`, judge not called, nothing persisted.
- `from_env()`: missing `EVALKIT_JUDGE_MODEL` / unknown provider → `ConfigError`.
- Store: save/get/list (tag filter, limit, ordering)/review.
- CLI commands via `main([...])`.
- End-to-end: input → judge → result → persisted → retrieved → human review.

A real Bedrock run is a manual smoke test (documented in README), not part of
the pytest suite.

## Evaluation flow

1. `evaluate()` validates inputs and builds the `Rubric`. Failure →
   `RubricError`; judge not called; nothing persisted **[OD-4]**.
2. Start timer; `raw = judge.judge(...)`.
3. `rubric.score(raw)` validates and scores.
4. On `JudgeOutputError` (from step 2 or 3): repeat steps 2–3 once with an
   identical request **[OD-5]**. `JudgeError`/`JudgeTimeoutError` are never
   retried **[OD-1]**.
5. Stop timer → `latency_ms` (total, including the retry).
6. Build `EvaluationResult` (`status=ok`, or `status=error` + message).
7. `store.save(result)` if a store is configured.
8. Return the result, or raise the typed error (with `.evaluation_id`).

## Key design decisions

- Core (`evaluator`, `models`, `store`, `judge`) is provider-agnostic and
  SDK-free; Bedrock specifics live only in `bedrock.py`.
- Bedrock Converse with forced tool use for structured output.
- Judge returns raw payloads; `Rubric.score` is the single validation point,
  with strict (non-coercing) checks.
- Scoring and verdict are computed in Python, never by the LLM.
- Only malformed output is retried, once, by `Evaluator`.
- Prompt version = fingerprint of the static text plus a golden render of a fixture.
- Rubric version = explicit or content hash.
- Two tables; per-criterion scores, metadata and tags stored as JSON columns.
- Judge-stage failures are persisted, then raised.
- CLI is argparse and contains no logic.

## Resolved Decisions

| ID | Decision | Status | Resolution |
|---|---|---|---|
| OD-1 | Provider retry behavior | **RESOLVED** | SDK automatic retries disabled (`total_max_attempts=1`). evalkit retries at most once and only for malformed/invalid judge output. Provider errors and timeouts are never retried. |
| OD-2 | Human review score scale | **RESOLVED** | Optional overall human score normalized to `[0, 1]`, directly comparable with `overall_score`. |
| OD-3 | `Review` public export | **RESOLVED** | Exported from `evalkit`, as the return type of `Evaluator.review()`. |
| OD-4 | Persisting invalid input | **RESOLVED** | Invalid input/rubric errors are not persisted; validation happens before the judge. Judge-stage failures (provider error, timeout, malformed output) are persisted. |
| OD-5 | Malformed-output retry | **RESOLVED** | One retry from `Evaluator` with a fresh identical request. Provider-agnostic; no validation-feedback logic in v1. |
| OD-6 | Default judge model | **RESOLVED** | No hardcoded default. `EVALKIT_JUDGE_MODEL` is required by `from_env()`. v1 implements only `BedrockJudge`; the `Judge` Protocol stays provider/model-agnostic so `AnthropicJudge`/`OpenAIJudge` can be added later (not created in v1). |

## Known limitations (v1)

- Forced `toolChoice` is only supported by some Bedrock model families
  (e.g. Anthropic Claude). Unsupported models fail with a `ValidationException`
  → `JudgeError`. README will state the supported-model requirement.
- JSON Schema keywords (`minimum`/`maximum`, `additionalProperties`) are
  advisory to the model; enforcement is in Python.
- `MAX_TOKENS` is a fixed constant; very large rubrics could truncate →
  `JudgeOutputError`.
- Retrying an identical request at temperature 0 may reproduce the same
  malformed output; accepted for v1 [OD-5].
- Truncated judge output (`stopReason=max_tokens`) is still retried once, like
  any other malformed output, even though the identical retry will usually
  truncate again too. Left as-is to keep the retry rule uniform per [OD-5]
  rather than carve out a silent exception; a very large rubric is the only
  way to hit this.
- `judge_prompt_version` hashes the prompt template and tool definition only,
  not provider-specific request parameters (e.g. Bedrock's `MAX_TOKENS`).
  Schema *shape* doesn't need to be included: it's a deterministic function of
  the rubric, which is already versioned separately via `rubric_version`.

### Post-review hardening (implementation)

Found during a repository review after initial implementation, fixed without
changing approved design:

- **`SQLiteStore` is now safe to share across threads.** The connection is
  opened with `check_same_thread=False` and every call is serialized through
  one `threading.Lock`. A bare `sqlite3.connect()` raised `ProgrammingError`
  the moment a second thread called it, which would lose an already-paid-for
  judge result.
- **`Criterion.name` is restricted to `^[A-Za-z0-9_-]{1,64}$`.** A name is
  used verbatim as a tool-schema property name; anything else (spaces,
  slashes, >64 chars) is rejected as `RubricError` before the judge is called,
  instead of surfacing later as a Bedrock `ValidationException`.
- **`render_prompt` escapes `<`/`>` in `prompt`, `model_output` and
  `reference_output`.** Evaluated content containing a literal closing tag
  (e.g. `</model_output>` followed by injected instructions) can no longer
  break out of its section and add fake instructions to the judge prompt.
- **`SQLiteStore.list()` now loads each row's reviews**, matching `get()`,
  instead of always returning `reviews: []`.
- **A storage failure while persisting a judge-stage error no longer hides
  that error.** `Evaluator.evaluate()` chains the storage exception's `__cause__`
  to the original judge error before letting it propagate (still a plain
  exception per the architecture's failure table, just no longer silently
  losing context).
- **CLI:** `run` checks the input file's keys against `evaluate()`'s
  parameters up front (clearer error than a bare `TypeError`), and
  `list --limit` rejects non-positive values instead of silently returning
  every row (SQLite treats `LIMIT -1` as "no limit").

## 0.2.0 (P0 hardening) changes

Implements P0 of [`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) for the single-record path;
findings refer to [`PRODUCTION-AUDIT.md`](PRODUCTION-AUDIT.md). These **supersede** any earlier
statement in this document.

- **Numeric safety (F-1).** `Criterion.weight` is finite and within `[1e-6, 1e6]`; scale bounds are
  within +/-1,000,000; all float fields reject NaN/Infinity. `Rubric.score` raises `ScoringError`
  (a `JudgeError`, persisted, never retried) if the result is not a finite value in `[0, 1]`.
  `EvaluationResult` enforces coherence: `ok` <=> score + verdict (following the rubric threshold) +
  one score per criterion + no error; `error` <=> message and none of those. The `evaluations`
  table has an INSERT trigger enforcing the same (SQLite stores NaN as NULL, which is how the
  original bug produced `ok`/NULL rows).
- **Migrations (F-3).** `evalkit.migrations`: `schema_migrations` table with checksums; each
  migration in its own `BEGIN IMMEDIATE` transaction; legacy databases (v0: before `context`;
  v1: 0.1.0) are detected by column set and adopted; unrecognized, newer, or tampered databases
  are refused; a consistent backup `<db>.bak-v<N>-<timestamp>` is written before changing a
  non-empty file. All INSERTs name their columns. New files are `0600` and use WAL with a 10 s busy
  timeout (the WAL switch is retried: it needs a lock the busy timeout does not cover).
- **Lossless prompt encoding (F-4).** `_escape` is gone. Each of `prompt`, `model_output`,
  `reference_output`, `context` is placed verbatim between
  `<<<EVALKIT:<marker> NAME>>>` / `<<<END:<marker> NAME>>>` lines. `marker` is 16 hex characters of
  a SHA-256 over the content, re-derived with a counter if it occurs in any content, so a delimiter
  line cannot be forged from inside evaluated content. Rubric descriptions, labels and versions are
  validated (length, printable, no delimiter text) and rendered in a separate trusted section after
  the data. Prompt bytes changed, so `judge_prompt_version` changed: pre-0.2.0 and 0.2.0
  results are deliberately not comparable. The `context` instruction now says context is the
  source of truth for support; that wording is **not yet validated against a live judge**.
- **Attempt evidence (F-2).** `EvaluationResult.attempts` (`evaluations.attempts_json`): one
  `Attempt` per judge call with outcome (`ok`, `invalid_output`, `timeout`, `provider_error`,
  `scoring_error`, `unexpected_error`), timing, scrubbed error, provider `kind`
  (`truncated`, `refused`, `malformed_output`, `context_window`, `no_tool_call`, `invalid_json`),
  and, for failed attempts, the rejected/partial output (truncated to 64 KB, with the SHA-256 of the
  full text). Rows written before 0.2.0 read as `attempts=[]` (unknown). Retry policy is unchanged.
- **Rubric identity (F-7).** `Rubric.content_hash` (full SHA-256 of content excluding `version`;
  the auto version is its first 12 hex characters, unchanged). `rubric_versions` binds each version
  label to one content hash; `Evaluator` checks it **before** the judge call and `save()` re-checks.
  Existing databases are backfilled (earliest row wins; conflicts are logged, rows untouched).
- **Limits (F-10).** `Limits` (per `Evaluator`): 256 KiB per text field (UTF-8 bytes), 16 KiB
  metadata, 32 tags (128 chars each); rubric: 32 criteria, 2,000-char descriptions, 32 labels of
  <= 64 printable chars. Unpaired surrogates are refused up front. CLI input: 8 MiB cap, UTF-8,
  strict JSON (no NaN/Infinity/`1e999`, no duplicate keys).
- **Pagination (F-12).** `limit` is validated (1..10,000); `list_page()` returns
  `(results, next_cursor)` using keyset pagination on `(created_at, rowid)` (a cursor should not
  outlive a `VACUUM`); reviews are loaded in chunks of 500. CLI `list --cursor`.
- **Storage failure (F-13).** A failed `save()` raises `StoreError` (also an `sqlite3.Error`,
  preserving the old contract) carrying the complete `.result`, and writes it to `<db>.spill/<id>.json`
  (atomic, `0600`). `recover_spilled()` / `evalkit recover` replays it; opening a store with waiting
  spill files logs a warning.
- **Errors and CLI.** New `ScoringError`, `StoreError`, `MigrationError`. Provider stop reasons
  `guardrail_intervened`, `content_filtered`, `malformed_*`, `model_context_window_exceeded` are
  distinct kinds (previously "no tool call"). Error text is scrubbed of ARNs, keys and bearer
  tokens (`evalkit.redact`). The CLI no longer converts an internal `TypeError` into "invalid input
  file", and reports database errors cleanly.

## Dataset foundation (P1, part 1)

The first slice of P1 from [`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) (see its
implementation notes under §3.3): versioned, immutable datasets. No runs, targets or
evaluator orchestration yet; the single-record `Evaluator` API is unchanged and shares the database.

| Module | Role |
|---|---|
| `hashing.py` | `canonical_json` (sorted, compact, ASCII, no NaN, string keys only), `stable_hash(domain, obj)`, `dataset_hash` |
| `datasets.py` | pure: `EvaluationCase`, `StoredCase`, `Dataset`, `DatasetVersion`; validation (`validate_cases`, `check_case_limits`), streaming `read_jsonl`, `IssueCollector`, `DatasetService` (import/read/export/lint/verify) |
| `dataset_store.py` | `DatasetStoreMixin` on `SQLiteStore`: atomic import transaction, resolve, keyset-paged `iter_cases`, `verify_version`, `lint_version` |
| `kit.py` | `EvalKit.open(path)` / `.from_env()` with `.datasets` |
| `migrations.py` | migration 3 `dataset_foundation`: `datasets`, `dataset_versions`, `cases`, unique/CHECK constraints, immutability triggers |

- **Import** streams JSONL (UTF-8, no NaN/Infinity/duplicate keys, 4 MiB line cap, oversized lines
  skipped unread), validates every case strictly (no type coercion, unknown fields refused, size and
  structure limits, no lone surrogates), and reports **all** problems (capped at 100) with line
  numbers; nothing is stored unless the whole input is valid. It inserts under `BEGIN IMMEDIATE` into an
  unsealed version, computes the hash in SQL order, and either returns the existing identical
  version (rolled back) or seals the new one. The store lock is held for the whole import.
- **Immutability** is enforced by the database (triggers + CHECKs) and detectable after the fact:
  `verify` recomputes every case hash and the dataset hash from the stored rows; `export`
  re-checks each case while writing.
- **Export** writes JSONL in `case_key` order, atomically, owner-only; re-importing it yields the
  same content hash (the same version).
- **Content is lossless**: no escaping or normalization anywhere; `None`, `""` and `[]` stay distinct.

CLI: `evalkit dataset import NAME FILE | list | show REF [--cases N] | export REF FILE [--force] |
lint REF | verify REF`.
