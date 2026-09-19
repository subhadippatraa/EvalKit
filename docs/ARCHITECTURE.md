# evalkit — Architecture (v1)

Status: **architecture final; implementation not yet approved.** Source of
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
  __init__.py    # public exports
  models.py      # Criterion, Rubric (+ .score()), CriterionScore, EvaluationResult, Review
  errors.py      # error hierarchy
  judge.py       # Judge Protocol, PROMPT_TEMPLATE, render_prompt(), output_schema(), PROMPT_VERSION  (no SDK imports)
  bedrock.py     # BedrockJudge — the only module that imports boto3/botocore
  store.py       # Store Protocol, SQLiteStore
  evaluator.py   # Evaluator (+ from_env): orchestration, retry, persistence
  cli.py         # argparse entry point `evalkit`
tests/
  test_models.py test_evaluator.py test_bedrock.py test_store.py test_cli.py test_e2e.py
```

- Only two Protocols (`Judge`, `Store`). No registry, factory, provider
  manager, or plugin system.
- `evaluator.py` depends only on the `Judge` Protocol. The single place that
  names `BedrockJudge` is `Evaluator.from_env()`, which imports `bedrock`
  lazily inside the function.
- Future providers (`AnthropicJudge`, `OpenAIJudge`) are **not** created in v1;
  each would be one new module like `bedrock.py`, reusing `judge.py`.

## Public API

Exports from `evalkit`: `Evaluator`, `Rubric`, `Criterion`, `EvaluationResult`,
`Review` **[OD-3]**, and the error classes.

```python
class Evaluator:
    def __init__(self, judge: Judge, store: Store | None = None): ...
    @classmethod
    def from_env(cls) -> "Evaluator": ...          # BedrockJudge + SQLiteStore
    def evaluate(self, prompt: str, model_output: str,
                 reference_output: str | None = None,
                 criteria: dict[str, str] | None = None,
                 rubric: Rubric | None = None,
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
- `BedrockJudge` is importable as `evalkit.bedrock.BedrockJudge` for explicit
  construction; it is not a top-level export.

## Pydantic / domain models (`models.py`)

| Model | Fields |
|---|---|
| `Criterion` | `name: str`, `description: str`, `scale: tuple[int, int] = (1, 5)`, `weight: float = 1.0` |
| `Rubric` | `criteria: list[Criterion]`, `threshold: float = 0.75`, `version: str \| None` |
| `CriterionScore` | `reasoning: str`, `score: int` (field order: reasoning first) |
| `EvaluationResult` | `id`, `created_at`, `status: "ok" \| "error"`, `error: str \| None`, `prompt`, `model_output`, `reference_output`, `rubric`, `rubric_version`, `judge_provider`, `judge_model`, `judge_temperature`, `judge_prompt_version`, `scores: dict[str, CriterionScore]`, `overall_score: float \| None` (0–1), `verdict: "PASS" \| "FAIL" \| None`, `latency_ms`, `metadata: dict`, `tags: list[str]`, `reviews: list[Review]` (populated on `get`) |
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
    def judge(self, prompt: str, model_output: str,
              reference_output: str | None, rubric: Rubric) -> dict[str, Any]: ...
```

Contract for any implementation:

- Returns the **raw** structured payload (the tool input object) unmodified.
  It does not coerce, validate ranges, score, or retry.
- Raises `JudgeTimeoutError` on timeout, `JudgeError` on any other provider
  failure, `JudgeOutputError` only when the response contains no structured
  payload at all (e.g. no tool call, truncated response).
- Makes exactly one provider request per call (no SDK-level retries) **[OD-1]**.

Provider-agnostic helpers, shared by all current and future judges:

- `PROMPT_TEMPLATE`: instructions + prompt, model output, optional reference
  output, criterion descriptions with scales.
- `render_prompt(prompt, model_output, reference_output, rubric) -> str`.
- `output_schema(rubric) -> dict`: the JSON Schema below.
- `PROMPT_VERSION = sha256(PROMPT_TEMPLATE + TOOL_NAME + TOOL_DESCRIPTION)[:12]`.

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
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, instance role…).
  evalkit reads no AWS secrets itself.

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
| SQLite failure | no | n/a | `sqlite3.Error` propagates |

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
`Evaluator` via `from_env()` and makes one library call; output is JSON on
stdout.

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
| `EVALKIT_JUDGE_PROVIDER` | `bedrock` | no | Only `bedrock` accepted in v1; anything else → `ConfigError` |
| `EVALKIT_JUDGE_MODEL` | — | **yes** [OD-6] | Bedrock model ID / inference profile ID |
| `EVALKIT_JUDGE_TEMPERATURE` | `0` | no | Judge temperature |
| `EVALKIT_JUDGE_TIMEOUT` | `60` | no | Per-request timeout (seconds) |
| `EVALKIT_DB_PATH` | `./evalkit.db` | no | SQLite path |
| `AWS_REGION`, `AWS_PROFILE` / AWS credentials | AWS chain | via AWS chain | Read by boto3, not by evalkit |

- Missing `EVALKIT_JUDGE_MODEL` or unparsable numbers → `ConfigError`.
- `.env.example` documents these with placeholder values; no secrets
  committed. No `python-dotenv`.
- `from_env()` is the only code that reads environment variables.

## Dependencies

- Runtime: `pydantic>=2`, `boto3`.
- Dev: `pytest`, `ruff`.
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
- Prompt version = hash of the shared template and tool definition text.
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
