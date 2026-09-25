# Changelog

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
