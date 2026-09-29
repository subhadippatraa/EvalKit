# evalkit — Architecture (v1)

Status: **approved and implemented (v0.1.0).** All open decisions are resolved
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
`F-n` labels refer to findings of an internal production audit. These **supersede** any earlier
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


## P1 platform: runs, execution, analysis

The remaining P1 slices of [`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md), built on the dataset
foundation. Still one process and one SQLite file; the single-record `Evaluator` API is unchanged.

| Module | Role |
|---|---|
| `failures.py` | `FailureClass` (input / target / evaluator / infrastructure), the kind table, `Failure`, `EvalFailure` |
| `runs.py` | pure domain: `RunConfig` (identity vs exec hash), `Run`, `CaseOutcome`/`CaseResult`, `EvaluatorOutcome`/`EvaluatorResult`, `RunAttempt`, `RunService` |
| `run_store.py`, `analysis_store.py` | SQL for runs/results and for analysis reads, tags, reviews, summaries, judge checks (mixins on `SQLiteStore`) |
| `llm.py`, `judge_eval.py` | transport contract (`LLMClient`) and judge semantics over it; `bedrock.py` / `bedrock_openai.py` are the only SDK importers |
| `calls.py` | retry policy, backoff, timeouts, rate limiter, breaker / budget guard, `UnitCalls` (an attempt per call) |
| `targets.py`, `evaluators.py` | the four targets; six evaluators and the registry (`resolve()` makes a spec canonical so its key names its behaviour) |
| `engine.py` | `ExecPolicy`, preflight, `ResultWriter` (single writer, batches, spill), `RunController` |
| `stats.py`, `analysis.py` | stdlib statistics; aggregation with denominators / coverage |
| `compare.py` | paired comparison, confounders, decision rule, `Gates`, baselines by tag |
| `calibration.py`, `judgecheck.py` | reviews, calibration, queues, disagreement; golden-set judge check |
| `report.py`, `cli_platform.py`, `env.py` | static HTML report; CLI handlers; clients from environment variables |

Invariants the database enforces (migration 4), independent of the Python code:

- A run pins one *sealed* dataset version; its config, identity, evaluator set and creation time
  are frozen. Status changes follow the lifecycle (`created → running → succeeded | partial |
  cancelled | failed`, and resume) and a run cannot succeed until every case has a terminal result.
- Results are write-once and only accepted while the run is `running`; a case result must belong
  to the run's dataset version; an evaluator result must carry its case result's run and one of the
  run's registered evaluators.
- **A target failure is never scored**: a failed case result accepts only `skipped` evaluator
  results, and `failure_class` sets are disjoint by level (case: input | target | infrastructure;
  evaluator: input | evaluator | infrastructure).
- Metrics are finite and only attach to `ok` results; attempts have exactly one owner and a failure
  class that fits what was called. Nothing is ever updated or deleted.

CHECK constraints are written so a NULL cannot slip through (a CHECK that evaluates to NULL passes);
each has a raw-SQL test including the NULL cases. `verify` recomputes hashes and relationships to
detect changes made by a writer that bypassed the triggers.

**Execution model.** `create` (preflight, freeze, plan with a seeded order) → `execute`: a producer
pages pending units by `(ord, id)` keeping at most `window` in flight; each unit runs the target then
every evaluator independently; results go through one writer thread that commits batches. Cancellation
and budgets stop dispatch (in-flight units finish); systemic failures stop the run as `failed`, a
tripped breaker as `partial`. Resume repairs half-finished units and finishes pending ones; terminal
results are never redone. On a persistent storage failure the unwritten results are appended to
`<db>.spill/<run>.jsonl` and replayed first on the next execute.

**Deviations from `TARGET-ARCHITECTURE.md`** (each a decision, not an accident):

1. *Flat module layout* instead of the sub-package restructure; the import-direction rules of §2.1 are
   enforced by `tests/test_architecture.py` on the flat layout. No compatibility shims were needed.
2. `CaseResult.status` is `pending | complete | failed` plus `(failure_class, kind, retryable)` columns,
   not `claimed | target_failed | retryable`; `claimed`, `worker_id` and `lease_expires_at` are P2 (leases).
   `ord` (seeded order) was added. No `runs.source_run_id`: a `reuse` target keeps it in its identity.
   No `cancel_requested` column (cancellation is in-process).
3. `succeeded` means every case reached a terminal result; **coverage is enforced by gates**
   (`Gates.min_coverage`), not by the run status. *(Superseded by P1.1: success also needs every
   evaluator result and at least one scored case per evaluator; see "P1.1 hardening".)*
4. *Pulled forward from P2 at the request of the P1 brief, in minimal form:* a rate limiter
   (fixed spacing), a consecutive-failure breaker, token / call / duration budgets, `citation_check`.
   *Not* pulled forward: adaptive concurrency, the response cache, USD price tables (prices are a
   caller-supplied mapping used only for an estimate), attempt columns for queue-wait / backoff / cost /
   cache-hit, leases, workers, `--retry-failed`.
5. Provider mapping: a *target* whose provider rejects the request is `input.bad_config` (the design's
   `target.bad_config` is not in its own kind table); `infrastructure.internal_error` was added for bugs
   in EvalKit itself; `input.bad_config` is systemic. A truncated *target* answer is kept as an answer
   (flagged), not a failure.
6. `json_schema` is an optional extra (`evalkit[jsonschema]`), as §20 recommends.
7. Run summaries are appended as history (one snapshot per execute), not a single row.
8. Migration 4 was still unreleased when `ord` and `evaluator_count` were added, so it was edited
   in place (released migrations 1-2 are untouched; a test pins every checksum).


## P1.1 hardening

Fixes from the audit of the P1 platform (finding ids `P0-1`, `P1-n` refer to that audit). No new
capability; it makes the P1 guarantees hold. It **supersedes** the P1 text above where they differ
(notably deviation 3 and the "Execution model" paragraph). Rationale for each choice:
[`DECISIONS.md`](DECISIONS.md); what the numbers mean: [`EVALUATION-METHODOLOGY.md`](EVALUATION-METHODOLOGY.md).

| Module / object | Role |
|---|---|
| `runlock.py` | `RunLock`: one executor per run (`flock` on `<db>.locks/<run>.lock`, in-process registry for `:memory:`); `RunBusyError` |
| migration 6 `p1_1_hardening` | `cases.input_hash` (backfilled), `case_results.meta_json`, the stricter `succeeded` trigger, pending rows must carry `ord` |
| `hashing.case_input_hash` | the input-side identity comparison pairs on |

**Schema (migration 6).** Added by a new migration; migrations 1-5 are byte-identical.
`runs_succeeded_needs_every_case` is replaced: success needs every case terminal **and** exactly
`terminal cases x evaluator_count` evaluator results. `case_results_pending_is_ordered` refuses a pending
result without `ord`; `case_results_meta_*` allow metadata only on complete results. `input_hash` is
backfilled by a migration hook that lifts and restores the `cases_no_update` trigger inside the
migration's own transaction (a failure rolls both back; tested).

**Execution model changes.**

- **Exclusive execution.** `execute` holds a `RunLock` for its whole duration. A second executor gets
  `RunBusyError` before any call; a dead executor's lock is released by the kernel.
- **Exception safety.** The loop is wrapped: on Ctrl-C or any exception the token is cancelled, queued
  units are dropped, running ones get a bounded grace period (`retry.timeout_s + 5`), the writer is
  flushed and closed, the run is transitioned (`cancelled(interrupted)` or
  `partial(infrastructure.internal_error)`) and the exception is re-raised. A result produced after the
  writer closed is spilled, not lost.
- **The writer cannot die silently.** Groups (`put(a, b, c)`) are written in one transaction and never
  split. Any exception in the thread is recorded in `writer.failed`, the batch in hand and the queue
  are spilled, and later `put`s refuse; the executor stops with `infrastructure.storage` (a storage
  error) or `infrastructure.internal_error` (a bug) and reports `ExecutionReport.writer_error`.
- **Failed targets** are written together with their `skipped` evaluator rows (one group). Recovery
  (`pending_units`) now also repairs a failed case result missing them.
- **Systemic failures.** `RunGuard` counts consecutive failures of one systemic kind per
  `(provider, model)`, reset by a success of that provider and model; `systemic_threshold` (default 3)
  stops the run `failed`. A systemic-looking failure is held by the executor until a later success proves
  it case-specific (then recorded) and dropped if the run stops (its cases stay pending). At the end of
  a run that did not stop for it, held failures are recorded.
- **What `succeeded` means** (`RunController._complete`): terminal and complete (above), every evaluator
  with at least one `ok` result and at least `policy.min_coverage` of its applicable cases (default
  `0.0`), at least one evaluator. Otherwise `partial(insufficient_coverage)`.
- **Timeouts.** `retry.timeout_s` reaches the request: OpenAI-compatible `timeout=` per call; boto3 gets a
  client per distinct timeout (a caller-supplied client is used as is). Recorded in the run environment.
- **Pagination.** `next_pending` orders and seeks by exactly `(ord, id)`, the columns of
  `idx_case_results_claim`: an index range scan, no sort, constant cost per page.

**Analysis and gates.** Comparison pairs on `(case_key, input_hash)`; `Comparison.unpaired`,
`.not_applicable`, `.truncated`, `.pairs` are new. `Gates` gains `max_na_asymmetry` (default 0.05),
`max_na_share`, `max_truncated_share`, `allow_unpinned`. Gate failure codes added: `unpinned_target`,
`not_applicable_asymmetry`, `not_applicable_share`, `truncated_share`. `EvaluatorSummary` gains
`not_applicable_share`; `RunSummary` gains `truncated_outputs`. `CaseResult.meta` / `CaseOutcome.meta` /
`EvalInput.target_meta` carry what a target reported about its output.

**CLI.** `judge-check --run RUN [--evaluator NAME]` checks the run's own frozen evaluator;
`--name` names a standalone check; the built-in golden set is refused for another rubric. `runs execute`
exits `1` for anything but `succeeded`, including `partial(insufficient_coverage)`.

### Deviations from `TARGET-ARCHITECTURE.md` added or changed by P1.1

Numbering continues the P1 list above.

9. **Success requires scoring something, not the design's `min_coverage` default.** The design says
   "coverage at least `min_coverage`" (0.95 for gates). The run-level default is `0.0` with a hard
   requirement of at least one scored result, because failed results cannot be retried yet (`--retry-failed`
   is P2) and a 0.95 default would make ordinary runs permanently `partial`. Gates keep the real bar.
10. **Run exclusivity by an OS lock**, not by the design's leases and heartbeat (P2). Exclusion only; no
    multi-worker claim of units.
11. **Systemic failures need repetition** and are held until proven case-specific; the design's "zero
    successes" is refined to "no success of the same provider and model in between".
12. **Pairing on the input side.** The design pairs on `case_key + case content_hash`; the content hash
    includes the system's output, which made precomputed comparisons drop changed cases.
13. **Truncation is stored on the case result** (`meta_json`) instead of on the design's attempt columns
    (which do not exist yet), so evaluators and reports can see it without joining attempts.
14. **Analysis is not "pure".** The design (§2.1) says analysis is pure (rows in, numbers out). Only
    `stats.py` is; `summarize`, `compare`, `calibrate` and `report` read the store through the `EvalKit`
    object. The import-direction test only sees top-level imports, so it cannot see this; it is stated here
    instead of being hidden.
15. **No type checker** (see `DECISIONS.md`, DR-12). Mutation testing runs on a schedule, not per push.

## P2.1 production operations

Retry, cache, cost and budgets, events, a reproducibility snapshot and observability, on the same
one-process, one-SQLite-file architecture. Rationale: [`DECISIONS.md`](DECISIONS.md) DR-15 to DR-25;
what the numbers mean: [`EVALUATION-METHODOLOGY.md`](EVALUATION-METHODOLOGY.md) §10-§13.

| Module / object | Role |
|---|---|
| `pricing.py` | `PriceTable`: versioned, user-supplied prices; unpriced and no-usage are `None`, never zero |
| `cache.py` | cache key (every request field + provider, model, endpoint, scope), codec, `CachePolicy`, `ResponseCache` with a per-key lock |
| `events.py` | structured events: an allow-list of fields, JSON / text formatters, `configure_logging` |
| `calls.py` | `RunGuard.reserve` (per-call budget reservation), `UnitCalls.run` (cache lookup, cost, events), `UnitAbandoned` |
| `envsnapshot.py` | the frozen environment, the per-execution runtime record, `environment_differences` |
| `ops_store.py` | cache table, `run_executions`, `spend_totals` (mixin on `SQLiteStore`) |
| `run_store.py` | retry: `count_retryable`, `reopen_failed_cases`, `retry_units`, `_replace_evaluator_result`, `failure_history` |
| `operations.py` | `run_status` / `format_status`: the observability dictionary the CLI and report share |

**Call path.** Every provider call of a unit (target or evaluator) goes through `UnitCalls.run`:
limiter, then (for an LLM request) cache key and lookup under the key's lock; on a hit the response is
re-validated and recorded as a `cache_hit` attempt (zero cost, no tokens); on a miss the call's worst
case is *reserved* against the budget under the guard's lock (or the call is refused: `UnitAbandoned`),
sent, validated, written to the cache, and recorded with its tokens and cost. A response that arrived
but failed validation records its tokens too. `RunGuard.record` replaces the reservation by what was
really used.

**Abandoned units.** A refused call (budget) or a replay miss raises `UnitAbandoned`. The executor
records nothing for the unit: it stays pending, the guard is `blocked`, dispatch stops and the run ends
`partial(budget)` / `partial(cache_miss)`. If the refusal comes after a real attempt in the same call
(a retry the budget will not pay for) the call ends as `infrastructure.budget_exceeded` with the
evidence kept, which `--retry-failed` retries.

**Retry-failed.** `execute(run_id, retry_failed=True)` (under the run lock): count eligible failures
(`failures.retry_eligible`, mirrored in SQL and tested equal); write an `execution` row (for a
`succeeded` run it is the ticket the database requires to reopen it); reopen the run; reopen failed
*case* results in chunks (failure archived in `result_history`, `skipped` rows removed, attempts kept);
enumerate failed *evaluator* results as retry units; then run the ordinary pipeline. A retried
evaluator's outcome replaces its failed row in place (`evaluator_retry` write item: archive, update,
metrics, attempts continuing at `n+1`). Crash safety: every step is one transaction; a reopened case is
simply pending on the next execute; a retried result already stored is a duplicate, never overwritten.

**Schema (migration 7).** `attempts` +`cache_hit, cache_key, cost_usd, price_version, retry_round`;
`case_results` / `evaluator_results` +`retry_round`; tables `result_history` (append-only),
`run_executions` (finished once), `llm_cache`. Replaced triggers: `runs_legal_transition` (a
`succeeded` run reopens only for an execution naming its `finished_at`), `case_results_write_once`
(`failed -> pending` only with the failure archived and the round advanced by one),
`evaluator_results_no_update` (`failed -> ok|not_applicable|failed` on the same terms),
`evaluator_results_no_delete` (only `skipped` rows of a reopened case). Each has raw-SQL tests including
the forged-history cases (`tests/test_p2_schema.py`).

### Deviations from `TARGET-ARCHITECTURE.md` added by P2.1

16. **Retry replaces in place**, with `result_history`, instead of the design's
    `INSERT ... ON CONFLICT DO UPDATE WHERE status != 'ok'` upsert: the write-once triggers stay, and
    every earlier attempt stays attached to the same row.
17. **The cache is off by default** (the design caches judge calls by default) and **its key includes the
    evaluator/target scope**; `--cache-only` is `--cache replay`, and a replay miss stops the run rather
    than being a stored `infra.cache_miss` failure.
18. **Budgets reserve per call** and count earlier executions (tokens, USD); the design's guarantee
    `spend <= budget + window x max_unit_cost` is replaced by "a call is made only if its worst case
    fits". `max_calls` is per execution (P1 behaviour kept).
19. **No packaged price table** (the design ships one): EvalKit cannot verify provider prices, so the
    operator supplies a versioned table; unpriced is unknown.
20. A per-execution table (`run_executions`) instead of a `run cancel` / heartbeat column; there is still
    no cross-process cancellation.

### Remaining P2 work

Superseding the P1.1 list. Not built, and not required by any workflow above: multi-process workers and leases; adaptive
concurrency and per-provider limiter tables; `samples=k` self-consistency and UNCERTAIN verdicts;
cross-process cancellation; `evalkit gc` and cache eviction (there is `cache clear`); streaming
analysis (compare / calibrate / report hold every case in memory); a subprocess or `re2` for the regex
evaluator; benchmarks at 1M cases; and live validation of the judge prompt, of the over-long-input
heuristic and of cost estimates once provider credentials work.

