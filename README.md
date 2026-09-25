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

Out of scope: retrieval metrics (recall/MRR/nDCG), batch/dataset management, dashboards,
HTTP APIs. v1 ships one judge provider: **AWS Bedrock**. Design details:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

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
```

A real Bedrock smoke test is manual. With AWS credentials and model access configured, run
`evalkit run --input input.json` and confirm you get `status: ok` and a verdict.
