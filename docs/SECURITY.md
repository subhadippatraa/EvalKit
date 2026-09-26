# Security

Threat model and controls of EvalKit as implemented (see also
[`TARGET-ARCHITECTURE.md`](TARGET-ARCHITECTURE.md) §14). EvalKit is a library and CLI that reads
files, calls LLM providers and writes one SQLite file. It has no server, no authentication, and no
multi-user model. Everything below assumes **one operator, one host, one local database**.

## Trust boundaries

| Data / actor | Trust | Notes |
|---|---|---|
| Dataset files and case fields | **untrusted** | may be adversarial: huge, malformed, or written to manipulate a judge |
| Target outputs | **untrusted** | a system may (even accidentally) address the judge |
| Retrieved documents, references, context | **untrusted** | an indirect-injection channel |
| Provider responses | **untrusted** | validated to numbers in Python; never executed |
| Rubrics, run specs, evaluator params | operator-authored, validated | size-, delimiter- and shape-checked |
| **`callable` target code** (`--target pkg.mod:fn`, `callable = "..."` in a spec file) | **trusted code** | runs with the operator's full privileges |
| Environment variables, the AWS credential chain, API keys | secrets | read at use; never persisted or logged |

**Treat a run spec file as code.** A spec's `[target] callable = "pkg.mod:fn"` imports and calls that
function, exactly like `--target`. Never execute a spec file (or import a module on `sys.path`) from a
source you would not run a script from. Datasets are data only: no case field is ever imported,
evaluated or used as a path.

## Controls

| Threat | Control | Residual risk |
|---|---|---|
| Structural prompt injection (breaking out of a data block) | each evaluated field is placed **byte-for-byte** between `<<<EVALKIT:<marker> NAME>>>` lines whose marker is derived from the content itself; rubric text is validated and rendered in a separate trusted section | none structural |
| **Semantic** injection ("score this 5") | the judge has no tools except a forced `submit_evaluation`; its output is reduced to validated numbers in Python; `judge-check` has adversarial cases; deterministic evaluators, calibration and disagreement reports exist | **real and unpreventable**: a judge can be talked into a score. It is made detectable, not impossible. |
| Arbitrary code execution | no `eval`/`exec`/`pickle`; no user-code evaluators; only `callable` targets run operator code | `callable` is trusted code by design |
| Reference / label leakage to the system under test | `TargetInput` has no field that could carry the reference, relevance, tags or metadata (a test asserts the field set); a model target's template may only use `{prompt}` and `{context}`; preflight warns when a template contains evaluation-set text | a *callable* is trusted and could read the dataset itself |
| Truncation or run provenance reaching the judge | target metadata (stop reason, truncation) is data for evaluators, never rendered into a judge prompt | |
| XSS from stored text in the HTML report | every interpolation is `html.escape`d; content sits in text nodes or `<pre>`; no attribute carries data; no JavaScript at all; CSP `default-src 'none'; style-src 'unsafe-inline'; img-src 'none'` | a browser bug |
| Regular-expression denial of service | patterns are operator-authored and length-capped; the subject is capped at 256 KB | **not mitigated.** Python's `re` cannot be interrupted and holds the interpreter lock: a catastrophic pattern such as `^(\w+\s?)*$` on a crafted target output freezes the *whole process* (all threads, Ctrl-C, the writer) until `kill -9`. The size cap does **not** bound the time. Review `regex` patterns for nested quantifiers; deferred (see `DECISIONS.md`). |
| SSRF | EvalKit never dereferences a URL from a dataset or an output; `json_schema` refuses remote `$ref` and never fetches; provider base URLs are operator config (`EVALKIT_JUDGE_BASE_URL`) | an operator-supplied base URL receives the API key |
| Path traversal | only operator-supplied paths are opened (`--input`, `--out`, `EVALKIT_DB_PATH`); existing output files are not overwritten without `--force` | operator error |
| Resource exhaustion | per-field, per-case, metadata, tag, criteria and import limits; bounded in-flight window and writer queue; token/call/duration budgets; a breaker | one valid 256 KB field is allowed |
| Malicious dataset causing cost | preflight states the number of calls; token, call, duration and **USD budgets are enforced before each call** (worst-case reservation, per run); a USD budget is refused unless every paid model is priced | the input bound assumes byte-level tokenizers; unreported usage is assumed worst-case only within an execution; there is no `--confirm-above` |
| Secrets or content in logs | events accept only an allow-list of identifier and number fields (no prompt, output, reference, rubric or provider message field exists); strings are scrubbed and clipped; failures are logged by class and kind, never message; a test drives a full run and greps the log | `case_key` and run ids are logged (identifiers you chose); an application that attaches its own handler to `evalkit.events` sees the same restricted records |
| Subprocess in environment capture | the only subprocess is `git rev-parse` (no shell, fixed arguments, 2 s timeout) run in EvalKit's own package directory, to record its commit; it reports nothing unless that directory is the `src/evalkit` of a git checkout | a hostile `git` on `PATH` (the operator's environment) |
| A stale or poisoned cache entry | only validated successes are stored; a hit is re-validated by current code, and an entry that fails is deleted; a corrupt entry is a miss; the key includes provider, model, endpoint, every request field and the evaluator/target scope | the cache holds **model text in plaintext** in the database (same sensitivity as results); anyone who can write the file can plant an entry, like any other row |
| Duplicate execution (double billing) | one executor per run: an OS advisory lock (`flock`) on `<db>.locks/<run>.lock`, released by the kernel when its holder dies | local filesystems only; not for network filesystems |

## Secrets and sensitive data

- Secrets exist only in the process environment or the AWS credential chain. The run config is
  persisted through an allow-listed serializer that has no credential or header fields; the run
  environment records versions (EvalKit, git commit, Python, SDKs, SQLite), the dataset identity, the
  evaluator / generation / retry / cache / budget / pricing configuration and the **host** of a client's
  base URL or its region, never keys, tokens, full URLs or credentials in a URL (a test sets secret
  environment variables and asserts none reaches the database).
- **Error text** is scrubbed before it is stored or printed: AWS ARNs and account ids, access-key ids,
  `sk-...` style keys, bearer tokens and `token=`/`password=`/`api-key` style values.
  **Judge reasoning, judged content and failed-attempt evidence are not scrubbed** (they are
  evidence). Failed attempts keep up to 64 KB of the rejected or partial provider output, which can
  contain the text that was sent. The SQLite file, the spill files and any HTML report therefore
  contain your prompts, outputs, references and judge reasoning **in plaintext**.
- **The price table** is a plain file you author (no secrets); it is stored in the run's policy. The
  response cache is part of the database and is covered by the same file permissions; `evalkit cache
  clear` deletes it. Structured logs (`--log-file`) are created owner-only.
- The database, backups (`<db>.bak-*`), spill files and reports are created owner-only (`0600`), lock
  and spill directories `0700`. Encryption at rest is delegated to the disk or volume. A backup file
  is briefly created with the process umask before it is restricted.
- Do not attach a report or database to a public ticket or CI artifact unless its content is
  shareable.

## Integrity, not authenticity

- Datasets, run configs, results, attempts and reviews are append-only, and this is enforced by
  database triggers and CHECKs as well as Python (a failed result can be superseded only with its
  failure archived in an append-only history, and a succeeded run can be reopened only through a
  recorded execution; a successful result can never change); `evalkit dataset verify` and `evalkit runs verify`
  recompute hashes and relationships to detect a writer that bypassed them. That detects accidental
  and naive tampering. **It is not a defence against a malicious local user**: anyone who can write the
  file can rewrite it consistently, drop the triggers, or delete it.
- Reviewer identity is an unauthenticated label. Run **tags are mutable** (a baseline is `tag:main`)
  and their history is not kept: whoever can run the CLI can repoint a baseline.
- `sample=random` on a review is the reviewer's declaration, not proof.

## Reporting a vulnerability

EvalKit has no security contact of its own. Open a private report with the repository owner rather
than a public issue, including the smallest input that reproduces the problem.
