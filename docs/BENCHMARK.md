# Local benchmark

> Re-measured after the P1.1 hardening (10K in this file's first table, 100K in the section at the end).

`scripts/benchmark.py` measures the platform in one process against one local SQLite file with a
fake zero-latency judge and target: it measures **EvalKit's own overhead** (validation, hashing,
persistence, aggregation), not any provider. Raw output: [`benchmark-10k.json`](benchmark-10k.json).

**Machine:** macOS-26.6.2-arm64-arm-64bit, 8 logical CPUs (arm), Python 3.12.14, SQLite 3.53.1; local
SSD; WAL mode; 4 worker threads; writer batch of 200 results.

**Workload:** 10,000 cases; three deterministic evaluators plus a scripted `llm_judge` (each case: 1 case
result + 4 evaluator results + 4 metric rows, 5 result rows per case, 50,000 rows in total).

| Step | Time | Throughput |
|---|---|---|
| Dataset import (10,000 cases) | 0.301 s | 33,210 cases/s |
| Create run + plan 10,000 pending results | 0.486 s | 20,573 cases/s |
| Execute, precomputed target, 4 evaluators | 7.71 s | **1,298 cases/s, 6,488 result rows/s** |
| Execute, callable target, 3 evaluators | 7.476 s | 1,338 cases/s |
| Aggregate a run (`summarize`) | 0.574 s | |
| Compare a run with itself (10,000 pairs) | 1.418 s | |
| Compare two different runs | 1.720 s | |
| Report, 200 cases of evidence | 0.818 s | 0.56 MB file |
| `runs verify` | 0.312 s | |

Database size after both runs: 74.4 MB. Peak resident memory of the whole process:
87.5 MB.

## 100,000 cases (P1.1)

Same machine, same workload shape (four evaluators including a scripted judge; 100,000 cases, 500,000
result rows, 500,000 metric rows), one run each. Raw output: [`benchmark-100k.json`](benchmark-100k.json).

| Step | 10K | 100K |
|---|---|---|
| Dataset import | 0.30 s (33,210 cases/s) | 3.8 s (26,306 cases/s) |
| Create run + plan pending results | 0.49 s | 5.7 s |
| **Page through all pending units** (256 per page) | 0.11 s | **0.42 s** (1.08 ms/page first third, 1.08 last third, 1.74 max) |
| Execute, precomputed target, 4 evaluators | 7.7 s (**1,298 cases/s**) | 136.9 s (**731 cases/s**, 3,654 rows/s) |
| Execute, callable target, 3 evaluators | 7.5 s (1,338 cases/s) | 115.1 s (869 cases/s) |
| Recovery scan of a finished run (`pending_units`) | 0.011 s | 0.19 s |
| Aggregate (`summarize`) | 0.57 s | 7.7 s |
| Compare a run with itself | 1.4 s | 21.1 s |
| Compare two different runs | 1.7 s | 25.1 s |
| Report, 200 cases of evidence | 0.82 s | 10.9 s |
| `runs verify` | 0.31 s | 4.9 s |
| Database size | 74 MB | 745 MB |
| Peak resident memory (whole process) | 88 MB | 450 MB |

**Pagination (audit P1-14).** Before the fix a page sorted every remaining pending row: 18 ms per page at
20K cases and 187 ms per page at 100K (about 73 s of store-lock time for one pass, measured in the
audit). It is now an index range scan: the cost of a page is the same at the start and the end of the
run (1.08 vs 1.08 ms) and a full pass over 100K units takes 0.42 s. `tests/test_pending_paging.py`
asserts it by the query plan and by SQLite's own instruction counter, not by wall-clock time.

**What the 100K numbers show, and do not.**

- Execution throughput fell by about 44% between 10K and 100K (1,298 to 731 cases/s). A diagnostic
  (three deterministic evaluators, 10K vs 60K cases) put the cost in the single result writer: its time
  per case grew from 323 to 587 microseconds, while every per-row statement is index-backed (checked with
  `EXPLAIN QUERY PLAN`). A 256 MB page cache changed the 60K figure by only 7%. The likely cause is
  ordinary B-tree growth with random UUID keys, but that is a hypothesis, not a finding, and no
  optimisation was made because the evidence does not identify a defect with a cheap fix. At 731
  cases/s a 100K-case run is still dominated by provider latency in any real use.
- Aggregation, comparison, the report and `verify` are **linear and read everything into memory**:
  compare holds a record per case for both runs (peak 450 MB for the whole 100K process). That is the
  first ceiling as datasets grow, and it is untested beyond 100K.
- One run each, no warm-up, one laptop: an order of magnitude, not a benchmark suite.

## What this does and does not say

- The design's baseline was ~186 rows/s for one commit per row (audit); batching gives
  6,488 rows/s here, and the acceptance check "all rows persisted exactly once" is asserted
  by `tests/test_engine.py` (1,000 cases x 3 evaluators: every case, evaluator result and metric
  present exactly once).
- Real runs are dominated by provider latency, not by these costs: at 200 ms per judge call and four
  threads the ceiling is about 20 judge calls/s regardless of the database.
- Numbers are from one laptop, one run each, no warm-up or repetition: treat them as an order of
  magnitude. Nothing was measured at 100K or 1M cases; nothing here supports a claim of
  distributed or production scale. Aggregation holds each metric's values in memory (about 32 bytes
  per value), which is the first ceiling as datasets grow.
- Interval computation is a seeded bootstrap below 5,000 values (a couple of seconds for a few
  metrics at that size) and a normal approximation above.
