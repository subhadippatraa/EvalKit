# Local benchmark

`scripts/benchmark.py` measures the platform in one process against one local SQLite file with a
fake zero-latency judge and target: it measures **EvalKit's own overhead** (validation, hashing,
persistence, aggregation), not any provider. Raw output: [`benchmark-10k.json`](benchmark-10k.json).

**Machine:** macOS-26.6.2-arm64-arm-64bit, 8 logical CPUs (arm), Python 3.12.14, SQLite 3.53.1; local
SSD; WAL mode; 4 worker threads; writer batch of 200 results.

**Workload:** 10,000 cases; three deterministic evaluators plus a scripted `llm_judge` (each case: 1 case
result + 4 evaluator results + 4 metric rows, 5 result rows per case, 50,000 rows in total).

| Step | Time | Throughput |
|---|---|---|
| Dataset import (10,000 cases) | 0.252 s | 39,639 cases/s |
| Create run + plan 10,000 pending results | 0.478 s | 20,904 cases/s |
| Execute, precomputed target, 4 evaluators | 7.79 s | **1,284 cases/s, 6,418 result rows/s** |
| Execute, callable target, 3 evaluators | 7.536 s | 1,327 cases/s |
| Aggregate a run (`summarize`) | 0.556 s | |
| Compare a run with itself (10,000 pairs) | 1.354 s | |
| Compare two different runs | 1.653 s | |
| Report, 200 cases of evidence | 0.794 s | 0.56 MB file |
| `runs verify` | 0.303 s | |

Database size after both runs: 73.8 MB. Peak resident memory of the whole process:
86.5 MB.

## What this does and does not say

- The design's baseline was ~186 rows/s for one commit per row (audit); batching gives
  6,418 rows/s here, and the acceptance check "all rows persisted exactly once" is asserted
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
