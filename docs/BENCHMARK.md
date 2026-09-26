# Local benchmark

> The first two tables are the P1.1 measurements. **P2.1 re-measured both sizes** (section "P2.1: production operations" below; raw output in [`benchmark-10k.json`](benchmark-10k.json) and [`benchmark-100k.json`](benchmark-100k.json), which now hold the P2.1 run).

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

## P2.1: production operations

Same machine and workload shape, one run each, re-measured after P2.1 (`scripts/benchmark.py`, which now
also measures the P2.1 paths). The judge is a zero-latency fake that reports 200 input / 20 output
tokens per call, so these numbers are EvalKit's own overhead.

**Did P2.1 slow the P1 paths?** Slightly. Every call now reserves against the budget guard, records
tokens / cost / cache columns, and checks whether an event is enabled:

| Step | P1.1, 10K | P2.1, 10K | P1.1, 100K | P2.1, 100K |
|---|---|---|---|---|
| Execute, precomputed target, 4 evaluators | 1,298 cases/s | 1,179 cases/s | 731 cases/s | 675 cases/s |
| Execute, callable target, 3 evaluators | 1,338 cases/s | 1,268 cases/s | 869 cases/s | 839 cases/s |
| Aggregate (`summarize`) | 0.57 s | 0.6 s | 7.7 s | 8.212 s |
| Report, 200 cases (now with the Operations section) | 0.82 s | 0.915 s | 10.9 s | 12.789 s |
| `runs verify` | 0.31 s | 0.332 s | 4.9 s | 5.217 s |
| Page through pending units | 0.11 s | 0.033 s | 0.42 s | 0.519 s |

The execute figures are 5-9% below the P1.1 ones. One run each on a laptop cannot separate that from
run-to-run variation, so read it as "no large regression", not as a measured cost of P2.1. Database size
is larger (10K: 161.5 MB, 100K: 1,615 MB) mostly because this script now
creates four more judge runs (with per-attempt cache and cost columns and cached responses) in the same file.

**The new paths** (a single `llm_judge`, `pricing` and, where stated, `cache` on):

| Step | 10K | 100K |
|---|---|---|
| Judge run, priced, cache **cold** (100% misses, every response written) | 1,445 cases/s | 1,100 cases/s |
| Same run again, cache **warm** (100% hits, **0 provider calls**) | 1,650 cases/s (0 calls, 10,000 hits) | 1,225 cases/s (0 calls, 100,000 hits) |
| Judge run, priced, no cache | 1,720 cases/s | 1,302 cases/s |
| Injected 2% outage: first pass | 5.936 s, 200 failed | 77.487 s, 2,000 failed |
| `--retry-failed` of those | 0.434 s, 200 calls, 0 failed left, verify ok | 6.583 s, 2,000 calls, 0 failed left, verify ok |
| `run_status` (all operational numbers) | 0.042 s | 0.638 s |
| Report with Operations section (1 evaluator) | 0.43 s | 6.171 s |

What these do and do not show:

- **The cache saves provider calls, not local time.** A warm rerun makes zero provider calls, but its
  local cost (reading the entry, re-validating, writing the attempt and result rows) is close to a
  cold run's: the saving is the provider's latency and bill, which a zero-latency fake does not have.
- **Retry cost is proportional to what failed**, not to the run: 2,000 retried units of 100,000 took
  6.583 s and exactly 2,000 provider calls (asserted equal to the failures).
- **Cost tracking is exact arithmetic on the fake's reported tokens**: 90.0 USD
  for 100,000 calls at 200 in / 20 out under the benchmark's example prices. It says nothing about a real
  provider's bill.
- Memory: peak resident memory of the whole 100K process is 453.6 MB (450 MB before), still
  dominated by compare / report holding every case. Nothing was measured at 1M cases.

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
