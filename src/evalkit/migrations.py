"""Versioned, forward-only SQLite migrations.

Every database -- new, pre-migration ("legacy"), or already migrated -- goes through `migrate()`:

* Applied versions are recorded in `schema_migrations` with a checksum of the migration text.
* Each migration runs in its own `BEGIN IMMEDIATE` transaction (SQLite DDL is transactional), so a
  failure leaves the database exactly as it was, and concurrent openers serialize on the write
  lock and re-check what is already applied (idempotent).
* A database with an `evaluations` table but no `schema_migrations` is a legacy database. Its
  shape is verified (v0 = before `context` existed, v1 = the 0.1.0 shape) and adopted; anything
  else is refused untouched, never guessed at.
* A database newer than this code, or with a modified applied migration, is refused.
* Before changing a non-empty file database, a consistent copy is written next to it.
"""

import hashlib
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evalkit.errors import MigrationError

log = logging.getLogger("evalkit.migrations")

BASELINE_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
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
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);

CREATE TABLE IF NOT EXISTS reviews (
  id            TEXT PRIMARY KEY,
  evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
  reviewer      TEXT NOT NULL,
  score         REAL CHECK (score BETWEEN 0 AND 1),
  verdict       TEXT NOT NULL CHECK (verdict IN ('PASS','FAIL')),
  comment       TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_evaluation_id ON reviews(evaluation_id);
"""

# P0 hardening. The trigger is the DB-level half of the score/verdict coherence invariant: SQLite
# stores NaN as NULL (which is how a NaN score once became `status=ok, overall_score=NULL`), so
# an "ok" row must carry a real score in [0, 1] and a verdict, and an "error" row neither.
# A trigger rather than a CHECK because a CHECK cannot be added to an existing table without a
# rebuild. INSERT only: rows are immutable and old rows are never re-validated.
P0_SQL = """
ALTER TABLE evaluations ADD COLUMN attempts_json TEXT;

CREATE TABLE rubric_versions (
  version       TEXT PRIMARY KEY,
  content_hash  TEXT NOT NULL,
  first_seen_at TEXT NOT NULL
);

CREATE TRIGGER evaluations_coherent_insert BEFORE INSERT ON evaluations
WHEN NOT (
  (NEW.status = 'ok'
     AND NEW.error IS NULL
     AND NEW.overall_score IS NOT NULL AND NEW.overall_score >= 0 AND NEW.overall_score <= 1
     AND NEW.verdict IS NOT NULL
     AND NEW.scores_json IS NOT NULL)
  OR
  (NEW.status = 'error'
     AND NEW.error IS NOT NULL
     AND NEW.overall_score IS NULL
     AND NEW.verdict IS NULL)
)
BEGIN
  SELECT RAISE(ABORT, 'evaluations: status/score/verdict are incoherent');
END;
"""


# Dataset foundation. A version row is created `sealed=0` while its cases are inserted and sealed
# (with its hash and count) in the same transaction, so a committed version is always sealed.
# Triggers make sealed data immutable at the database level, independent of the Python code.
DATASETS_SQL = """
CREATE TABLE datasets (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at  TEXT NOT NULL
);

CREATE TABLE dataset_versions (
  id           TEXT PRIMARY KEY,
  dataset_id   TEXT NOT NULL REFERENCES datasets(id),
  version_no   INTEGER NOT NULL CHECK (version_no >= 1),
  content_hash TEXT,
  case_count   INTEGER,
  source       TEXT,
  created_at   TEXT NOT NULL,
  sealed       INTEGER NOT NULL DEFAULT 0 CHECK (sealed IN (0, 1)),
  UNIQUE (dataset_id, version_no),
  -- explicit IS NOT NULL: a CHECK that evaluates to NULL (unknown) passes in SQL
  CHECK (sealed = 0 OR (content_hash IS NOT NULL AND length(content_hash) = 64
                        AND case_count IS NOT NULL AND case_count >= 1))
);
CREATE UNIQUE INDEX ux_dataset_versions_hash
  ON dataset_versions(dataset_id, content_hash) WHERE content_hash IS NOT NULL;

CREATE TABLE cases (
  id                 TEXT PRIMARY KEY,
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
  case_key           TEXT NOT NULL,
  content_hash       TEXT NOT NULL CHECK (length(content_hash) = 64),
  prompt             TEXT NOT NULL,
  output             TEXT,
  reference          TEXT,
  context            TEXT,
  retrieved_json     TEXT,
  relevance_json     TEXT,
  metadata_json      TEXT NOT NULL DEFAULT '{}',
  tags_json          TEXT NOT NULL DEFAULT '[]',
  UNIQUE (dataset_version_id, case_key)
);

CREATE TRIGGER cases_no_update BEFORE UPDATE ON cases
BEGIN SELECT RAISE(ABORT, 'cases are immutable'); END;

CREATE TRIGGER cases_no_delete BEFORE DELETE ON cases
BEGIN SELECT RAISE(ABORT, 'cases are immutable'); END;

CREATE TRIGGER cases_insert_only_unsealed BEFORE INSERT ON cases
WHEN (SELECT sealed FROM dataset_versions WHERE id = NEW.dataset_version_id) = 1
BEGIN SELECT RAISE(ABORT, 'dataset version is sealed'); END;

CREATE TRIGGER dataset_versions_sealed_no_update BEFORE UPDATE ON dataset_versions
WHEN OLD.sealed = 1
BEGIN SELECT RAISE(ABORT, 'dataset versions are immutable once sealed'); END;

CREATE TRIGGER dataset_versions_sealed_no_delete BEFORE DELETE ON dataset_versions
WHEN OLD.sealed = 1
BEGIN SELECT RAISE(ABORT, 'dataset versions are immutable once sealed'); END;
"""


# Runs and results. A run pins exactly one sealed dataset version and a frozen config; its results
# are write-once evidence. As with datasets, the rules live in the database (CHECKs and triggers)
# as well as in Python, so a bug or a raw connection cannot store an impossible state.
#
# NULL pitfall (see DATASETS_SQL): a CHECK that evaluates to NULL passes. Every disjunct below
# starts with a non-NULL test on `status`, and every `IN (...)` is guarded by `IS NOT NULL`.
# Triggers use COALESCE around lookups for the same reason: NOT(NULL) is NULL, not true.
#
# Where a fact spans tables (a case belongs to the run's dataset version; an evaluator result's
# run is its case result's run) a trigger checks it rather than denormalizing a column onto the
# biggest table. `run_id` on evaluator_results/metrics is kept because aggregation needs it.
RUNS_SQL = """
CREATE TABLE runs (
  id                 TEXT PRIMARY KEY,
  dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
  name               TEXT,
  status             TEXT NOT NULL CHECK (
    status IN ('created','running','succeeded','partial','failed','cancelled')),
  stop_reason        TEXT,
  config_json        TEXT NOT NULL,
  identity_hash      TEXT NOT NULL CHECK (length(identity_hash) = 64),
  exec_hash          TEXT NOT NULL CHECK (length(exec_hash) = 64),
  evaluator_count    INTEGER NOT NULL CHECK (evaluator_count >= 0),
  environment_json   TEXT NOT NULL DEFAULT '{}',
  idempotency_key    TEXT UNIQUE,
  created_at         TEXT NOT NULL,
  started_at         TEXT,
  finished_at        TEXT,
  CHECK (
    (status = 'created' AND started_at IS NULL AND finished_at IS NULL AND stop_reason IS NULL)
    OR (status = 'running' AND started_at IS NOT NULL AND finished_at IS NULL
        AND stop_reason IS NULL)
    OR (status = 'succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL
        AND stop_reason IS NULL)
    OR (status IN ('partial','failed','cancelled') AND started_at IS NOT NULL
        AND finished_at IS NOT NULL AND stop_reason IS NOT NULL)
  )
);
CREATE INDEX idx_runs_dataset_version ON runs(dataset_version_id, created_at);
CREATE INDEX idx_runs_identity ON runs(identity_hash);

CREATE TABLE run_evaluators (
  run_id        TEXT NOT NULL REFERENCES runs(id),
  evaluator_key TEXT NOT NULL,
  kind          TEXT NOT NULL,
  name          TEXT NOT NULL,
  version       INTEGER NOT NULL CHECK (version >= 1),
  spec_json     TEXT NOT NULL,
  PRIMARY KEY (run_id, evaluator_key)
) WITHOUT ROWID;

CREATE TABLE case_results (
  id              TEXT PRIMARY KEY,
  run_id          TEXT NOT NULL REFERENCES runs(id),
  case_id         TEXT NOT NULL REFERENCES cases(id),
  status          TEXT NOT NULL CHECK (status IN ('pending','complete','failed')),
  output          TEXT,
  retrieved_json  TEXT,
  failure_class   TEXT,
  failure_kind    TEXT,
  failure_message TEXT,
  retryable       INTEGER CHECK (retryable IS NULL OR retryable IN (0, 1)),
  created_at      TEXT NOT NULL,
  started_at      TEXT,
  finished_at     TEXT,
  duration_ms     INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
  -- processing order: a seeded hash of (seed, case_key), so a partial run is an unbiased sample
  ord             INTEGER,
  UNIQUE (run_id, case_id),
  CHECK (
    (status = 'pending' AND output IS NULL AND retrieved_json IS NULL
        AND failure_class IS NULL AND failure_kind IS NULL AND failure_message IS NULL
        AND retryable IS NULL AND started_at IS NULL AND finished_at IS NULL
        AND duration_ms IS NULL)
    OR (status = 'complete' AND output IS NOT NULL
        AND failure_class IS NULL AND failure_kind IS NULL AND failure_message IS NULL
        AND retryable IS NULL AND finished_at IS NOT NULL)
    OR (status = 'failed' AND output IS NULL AND retrieved_json IS NULL
        AND failure_class IS NOT NULL AND failure_class IN ('input','target','infrastructure')
        AND failure_kind IS NOT NULL AND failure_message IS NOT NULL
        AND retryable IS NOT NULL AND finished_at IS NOT NULL)
  )
);
CREATE INDEX idx_case_results_claim ON case_results(run_id, status, ord, id);
CREATE INDEX idx_case_results_failures ON case_results(run_id, failure_class, failure_kind)
  WHERE status = 'failed';

CREATE TABLE evaluator_results (
  id              TEXT PRIMARY KEY,
  case_result_id  TEXT NOT NULL REFERENCES case_results(id),
  run_id          TEXT NOT NULL,
  evaluator_key   TEXT NOT NULL,
  status          TEXT NOT NULL CHECK (status IN ('ok','not_applicable','failed','skipped')),
  verdict         TEXT CHECK (verdict IS NULL OR verdict IN ('PASS','FAIL','UNCERTAIN')),
  detail_json     TEXT NOT NULL DEFAULT '{}',
  failure_class   TEXT,
  failure_kind    TEXT,
  failure_message TEXT,
  retryable       INTEGER CHECK (retryable IS NULL OR retryable IN (0, 1)),
  created_at      TEXT NOT NULL,
  duration_ms     INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
  UNIQUE (case_result_id, evaluator_key),
  FOREIGN KEY (run_id, evaluator_key) REFERENCES run_evaluators(run_id, evaluator_key),
  CHECK (
    (status = 'ok' AND failure_class IS NULL AND failure_kind IS NULL
        AND failure_message IS NULL AND retryable IS NULL)
    OR (status IN ('not_applicable','skipped') AND verdict IS NULL AND failure_class IS NULL
        AND failure_kind IS NULL AND failure_message IS NULL AND retryable IS NULL)
    OR (status = 'failed' AND verdict IS NULL
        AND failure_class IS NOT NULL AND failure_class IN ('input','evaluator','infrastructure')
        AND failure_kind IS NOT NULL AND failure_message IS NOT NULL AND retryable IS NOT NULL)
  )
);
CREATE INDEX idx_evaluator_results_key ON evaluator_results(run_id, evaluator_key, status);
CREATE INDEX idx_evaluator_results_failures
  ON evaluator_results(run_id, failure_class, failure_kind) WHERE status = 'failed';

CREATE TABLE metrics (
  evaluator_result_id TEXT NOT NULL REFERENCES evaluator_results(id),
  run_id              TEXT NOT NULL,
  evaluator_key       TEXT NOT NULL,
  name                TEXT NOT NULL,
  -- NaN is stored as NULL by SQLite, which NOT NULL refuses; the range refuses +/-Infinity
  value               REAL NOT NULL CHECK (value BETWEEN -1.7976931348623157e308
                                                     AND 1.7976931348623157e308),
  PRIMARY KEY (evaluator_result_id, name)
) WITHOUT ROWID;
CREATE INDEX idx_metrics_aggregate ON metrics(run_id, evaluator_key, name);

CREATE TABLE attempts (
  id                  TEXT PRIMARY KEY,
  case_result_id      TEXT REFERENCES case_results(id),
  evaluator_result_id TEXT REFERENCES evaluator_results(id),
  n                   INTEGER NOT NULL CHECK (n >= 1),
  provider            TEXT,
  model               TEXT,
  started_at          TEXT NOT NULL,
  duration_ms         INTEGER NOT NULL CHECK (duration_ms >= 0),
  outcome             TEXT NOT NULL CHECK (outcome IN ('ok','failed')),
  error_class         TEXT,
  error_kind          TEXT,
  error_type          TEXT,
  error               TEXT,
  http_status         INTEGER CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599),
  request_id          TEXT,
  input_tokens        INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
  output_tokens       INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
  raw_payload         TEXT,
  raw_sha256          TEXT,
  raw_truncated       INTEGER NOT NULL DEFAULT 0 CHECK (raw_truncated IN (0, 1)),
  -- exactly one owner: a target call (case result) or an evaluator call (evaluator result)
  CHECK ((case_result_id IS NULL) + (evaluator_result_id IS NULL) = 1),
  CHECK (
    (outcome = 'ok' AND error_class IS NULL AND error_kind IS NULL AND raw_payload IS NULL)
    OR (outcome = 'failed' AND error_class IS NOT NULL
        AND error_class IN ('input','target','evaluator','infrastructure')
        AND error_kind IS NOT NULL)
  ),
  CHECK (raw_payload IS NULL OR raw_sha256 IS NOT NULL)
);
CREATE UNIQUE INDEX ux_attempts_case_result ON attempts(case_result_id, n)
  WHERE case_result_id IS NOT NULL;
CREATE UNIQUE INDEX ux_attempts_evaluator_result ON attempts(evaluator_result_id, n)
  WHERE evaluator_result_id IS NOT NULL;

-- runs -----------------------------------------------------------------------------------
CREATE TRIGGER runs_insert BEFORE INSERT ON runs
WHEN NEW.status <> 'created'
  OR (SELECT sealed FROM dataset_versions WHERE id = NEW.dataset_version_id) IS NOT 1
BEGIN SELECT RAISE(ABORT, 'a run starts as created, on a sealed dataset version'); END;

CREATE TRIGGER runs_config_immutable BEFORE UPDATE ON runs
WHEN NEW.id IS NOT OLD.id OR NEW.dataset_version_id IS NOT OLD.dataset_version_id
  OR NEW.name IS NOT OLD.name OR NEW.config_json IS NOT OLD.config_json
  OR NEW.identity_hash IS NOT OLD.identity_hash OR NEW.exec_hash IS NOT OLD.exec_hash
  OR NEW.evaluator_count IS NOT OLD.evaluator_count
  OR NEW.environment_json IS NOT OLD.environment_json
  OR NEW.idempotency_key IS NOT OLD.idempotency_key OR NEW.created_at IS NOT OLD.created_at
  OR (OLD.started_at IS NOT NULL AND NEW.started_at IS NOT OLD.started_at)
BEGIN SELECT RAISE(ABORT, 'a run''s configuration is frozen'); END;

CREATE TRIGGER runs_legal_transition BEFORE UPDATE ON runs
WHEN NOT (
  (OLD.status = 'created' AND NEW.status = 'running')
  OR (OLD.status = 'running' AND NEW.status IN ('succeeded','partial','cancelled','failed'))
  OR (OLD.status IN ('partial','cancelled','failed') AND NEW.status = 'running')
)
BEGIN SELECT RAISE(ABORT, 'illegal run status change'); END;

CREATE TRIGGER runs_succeeded_needs_every_case BEFORE UPDATE ON runs
WHEN NEW.status = 'succeeded'
  AND (SELECT COUNT(*) FROM case_results WHERE run_id = OLD.id AND status <> 'pending')
      <> (SELECT case_count FROM dataset_versions WHERE id = OLD.dataset_version_id)
BEGIN SELECT RAISE(ABORT, 'a run succeeds only when every case has a terminal result'); END;

CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'runs are permanent records'); END;

-- the run records how many evaluators it has, so no more can be added afterwards (deleting is
-- refused too), even while it is still `created`
CREATE TRIGGER run_evaluators_insert BEFORE INSERT ON run_evaluators
WHEN COALESCE((SELECT status FROM runs WHERE id = NEW.run_id), '') <> 'created'
  OR (SELECT COUNT(*) FROM run_evaluators WHERE run_id = NEW.run_id)
     >= COALESCE((SELECT evaluator_count FROM runs WHERE id = NEW.run_id), 0)
BEGIN SELECT RAISE(ABORT, 'evaluators are fixed when the run is created'); END;

CREATE TRIGGER run_evaluators_no_update BEFORE UPDATE ON run_evaluators
BEGIN SELECT RAISE(ABORT, 'run evaluators are immutable'); END;

CREATE TRIGGER run_evaluators_no_delete BEFORE DELETE ON run_evaluators
BEGIN SELECT RAISE(ABORT, 'run evaluators are immutable'); END;

-- case results ---------------------------------------------------------------------------
CREATE TRIGGER case_results_same_dataset_version BEFORE INSERT ON case_results
WHEN NOT EXISTS (
  SELECT 1 FROM runs r JOIN cases c ON c.dataset_version_id = r.dataset_version_id
  WHERE r.id = NEW.run_id AND c.id = NEW.case_id
)
BEGIN SELECT RAISE(ABORT, 'case is not part of the run''s dataset version'); END;

CREATE TRIGGER case_results_run_accepts_insert BEFORE INSERT ON case_results
WHEN NOT (
  (NEW.status = 'pending'
     AND COALESCE((SELECT status FROM runs WHERE id = NEW.run_id), '') IN ('created','running'))
  OR (NEW.status <> 'pending'
     AND COALESCE((SELECT status FROM runs WHERE id = NEW.run_id), '') = 'running')
)
BEGIN SELECT RAISE(ABORT, 'the run is not accepting this result (results need a running run)'); END;

CREATE TRIGGER case_results_write_once BEFORE UPDATE ON case_results
WHEN OLD.status <> 'pending' OR NEW.status = 'pending'
  OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id OR NEW.case_id IS NOT OLD.case_id
  OR NEW.created_at IS NOT OLD.created_at OR NEW.ord IS NOT OLD.ord
  OR COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') <> 'running'
BEGIN SELECT RAISE(ABORT, 'case results are write-once, and only on a running run'); END;

CREATE TRIGGER case_results_no_delete BEFORE DELETE ON case_results
BEGIN SELECT RAISE(ABORT, 'case results are permanent records'); END;

-- evaluator results ----------------------------------------------------------------------
CREATE TRIGGER evaluator_results_run_matches BEFORE INSERT ON evaluator_results
WHEN NEW.run_id IS NOT (SELECT run_id FROM case_results WHERE id = NEW.case_result_id)
BEGIN SELECT RAISE(ABORT, 'evaluator result must carry its case result''s run'); END;

CREATE TRIGGER evaluator_results_run_running BEFORE INSERT ON evaluator_results
WHEN COALESCE((SELECT status FROM runs WHERE id = NEW.run_id), '') <> 'running'
BEGIN SELECT RAISE(ABORT, 'results need a running run'); END;

-- a target failure is never scored: a failed case result only has `skipped` evaluator results,
-- a pending one has none, and `skipped` is reserved for a failed case result
CREATE TRIGGER evaluator_results_fit_case_result BEFORE INSERT ON evaluator_results
WHEN NOT (
  (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') = 'complete'
     AND NEW.status IN ('ok','not_applicable','failed'))
  OR (COALESCE((SELECT status FROM case_results WHERE id = NEW.case_result_id), '') = 'failed'
     AND NEW.status = 'skipped')
)
BEGIN SELECT RAISE(ABORT, 'evaluator result does not fit its case result''s status'); END;

CREATE TRIGGER evaluator_results_no_update BEFORE UPDATE ON evaluator_results
BEGIN SELECT RAISE(ABORT, 'evaluator results are write-once'); END;

CREATE TRIGGER evaluator_results_no_delete BEFORE DELETE ON evaluator_results
BEGIN SELECT RAISE(ABORT, 'evaluator results are permanent records'); END;

-- metrics --------------------------------------------------------------------------------
CREATE TRIGGER metrics_need_ok_result BEFORE INSERT ON metrics
WHEN NOT EXISTS (
  SELECT 1 FROM evaluator_results e JOIN runs r ON r.id = e.run_id
  WHERE e.id = NEW.evaluator_result_id AND e.status = 'ok'
    AND e.run_id = NEW.run_id AND e.evaluator_key = NEW.evaluator_key AND r.status = 'running'
)
BEGIN SELECT RAISE(ABORT, 'metrics belong to an ok evaluator result of a running run'); END;

CREATE TRIGGER metrics_no_update BEFORE UPDATE ON metrics
BEGIN SELECT RAISE(ABORT, 'metrics are write-once'); END;

CREATE TRIGGER metrics_no_delete BEFORE DELETE ON metrics
BEGIN SELECT RAISE(ABORT, 'metrics are permanent records'); END;

-- attempts -------------------------------------------------------------------------------
CREATE TRIGGER attempts_owner_running BEFORE INSERT ON attempts
WHEN NOT EXISTS (
  SELECT 1 FROM runs r WHERE r.status = 'running' AND r.id = COALESCE(
    (SELECT run_id FROM case_results WHERE id = NEW.case_result_id),
    (SELECT run_id FROM evaluator_results WHERE id = NEW.evaluator_result_id))
)
BEGIN SELECT RAISE(ABORT, 'attempts belong to a result of a running run'); END;

-- a target call cannot fail as an evaluator, nor an evaluator call as the target
CREATE TRIGGER attempts_class_fits_owner BEFORE INSERT ON attempts
WHEN (NEW.case_result_id IS NOT NULL AND NEW.error_class = 'evaluator')
  OR (NEW.evaluator_result_id IS NOT NULL AND NEW.error_class = 'target')
BEGIN SELECT RAISE(ABORT, 'attempt failure class does not fit what was called'); END;

CREATE TRIGGER attempts_no_update BEFORE UPDATE ON attempts
BEGIN SELECT RAISE(ABORT, 'attempts are write-once'); END;

CREATE TRIGGER attempts_no_delete BEFORE DELETE ON attempts
BEGIN SELECT RAISE(ABORT, 'attempts are permanent records'); END;
"""


# Analysis foundation: run summaries (immutable snapshots), run tags (baselines resolve by tag),
# `reviews` generalized to grade a case result / one evaluator's verdict (the table is rebuilt: an
# existing CHECK cannot be relaxed in place), and judge-check results.
ANALYSIS_SQL = """
CREATE TABLE run_summaries (
  run_id          TEXT NOT NULL REFERENCES runs(id),
  computed_at     TEXT NOT NULL,
  evalkit_version TEXT NOT NULL,
  summary_json    TEXT NOT NULL,
  PRIMARY KEY (run_id, computed_at)
) WITHOUT ROWID;
CREATE TRIGGER run_summaries_no_update BEFORE UPDATE ON run_summaries
BEGIN SELECT RAISE(ABORT, 'run summaries are snapshots'); END;
CREATE TRIGGER run_summaries_no_delete BEFORE DELETE ON run_summaries
BEGIN SELECT RAISE(ABORT, 'run summaries are snapshots'); END;

CREATE TABLE run_tags (
  run_id     TEXT NOT NULL REFERENCES runs(id),
  tag        TEXT NOT NULL CHECK (length(tag) BETWEEN 1 AND 128),
  created_at TEXT NOT NULL,
  PRIMARY KEY (run_id, tag)
) WITHOUT ROWID;
CREATE INDEX idx_run_tags_tag ON run_tags(tag);

CREATE TABLE reviews_new (
  id             TEXT PRIMARY KEY,
  evaluation_id  TEXT REFERENCES evaluations(id),
  case_result_id TEXT REFERENCES case_results(id),
  evaluator_key  TEXT,
  sample         TEXT NOT NULL DEFAULT 'targeted' CHECK (sample IN ('random', 'targeted')),
  reviewer       TEXT NOT NULL,
  score          REAL CHECK (score BETWEEN 0 AND 1),
  verdict        TEXT NOT NULL CHECK (verdict IN ('PASS', 'FAIL')),
  comment        TEXT,
  created_at     TEXT NOT NULL,
  -- exactly one subject: a legacy evaluation or a run's case result
  CHECK ((evaluation_id IS NULL) + (case_result_id IS NULL) = 1),
  CHECK (evaluator_key IS NULL OR case_result_id IS NOT NULL)
);
INSERT INTO reviews_new (id, evaluation_id, reviewer, score, verdict, comment, created_at)
  SELECT id, evaluation_id, reviewer, score, verdict, comment, created_at FROM reviews;
DROP TABLE reviews;
ALTER TABLE reviews_new RENAME TO reviews;
CREATE INDEX idx_reviews_evaluation_id ON reviews(evaluation_id);
CREATE INDEX idx_reviews_case_result ON reviews(case_result_id, evaluator_key);
CREATE TRIGGER reviews_no_update BEFORE UPDATE ON reviews
BEGIN SELECT RAISE(ABORT, 'reviews are append-only'); END;
CREATE TRIGGER reviews_no_delete BEFORE DELETE ON reviews
BEGIN SELECT RAISE(ABORT, 'reviews are append-only'); END;

CREATE TABLE judge_checks (
  id                  TEXT PRIMARY KEY,
  evaluator_key       TEXT NOT NULL,
  fixture_name        TEXT NOT NULL,
  fixture_hash        TEXT NOT NULL CHECK (length(fixture_hash) = 64),
  n                   INTEGER NOT NULL CHECK (n >= 1),
  correct             INTEGER NOT NULL CHECK (correct BETWEEN 0 AND n),
  adversarial_n       INTEGER NOT NULL CHECK (adversarial_n BETWEEN 0 AND n),
  adversarial_correct INTEGER NOT NULL CHECK (adversarial_correct BETWEEN 0 AND adversarial_n),
  failed              INTEGER NOT NULL CHECK (failed BETWEEN 0 AND n),
  results_json        TEXT NOT NULL,
  created_at          TEXT NOT NULL
);
CREATE INDEX idx_judge_checks_key ON judge_checks(evaluator_key, created_at);
CREATE TRIGGER judge_checks_no_update BEFORE UPDATE ON judge_checks
BEGIN SELECT RAISE(ABORT, 'judge checks are write-once'); END;
CREATE TRIGGER judge_checks_no_delete BEFORE DELETE ON judge_checks
BEGIN SELECT RAISE(ABORT, 'judge checks are permanent records'); END;
"""


# P1.1 hardening (audit of the P1 platform).
#  * cases.input_hash: what a case *asks* (prompt, context, reference, relevance), without the
#    system's output. Comparison pairs on it; pairing on the full content hash silently dropped
#    every case whose output had changed (audit P0-1). Backfilled below for existing versions.
#  * case_results.meta_json: what the target reported about its output (stop reason, truncation).
#  * a run only succeeds when every case is terminal AND every terminal case has all of the run's
#    evaluator results (a `succeeded` run with no evaluator results is not a measurement).
#  * a pending case result carries its processing order (paging by (ord, id) needs it).
HARDENING_SQL = """
ALTER TABLE cases ADD COLUMN input_hash TEXT;
ALTER TABLE case_results ADD COLUMN meta_json TEXT;

DROP TRIGGER runs_succeeded_needs_every_case;
CREATE TRIGGER runs_succeeded_needs_every_case BEFORE UPDATE ON runs
WHEN NEW.status = 'succeeded'
  AND (
    (SELECT COUNT(*) FROM case_results WHERE run_id = OLD.id AND status <> 'pending')
      <> (SELECT case_count FROM dataset_versions WHERE id = OLD.dataset_version_id)
    OR (SELECT COUNT(*) FROM evaluator_results WHERE run_id = OLD.id)
      <> (SELECT COUNT(*) FROM case_results WHERE run_id = OLD.id AND status <> 'pending')
         * OLD.evaluator_count
  )
BEGIN SELECT RAISE(ABORT,
  'a run succeeds only when every case has a terminal result and every evaluator result'); END;

CREATE TRIGGER case_results_pending_is_ordered BEFORE INSERT ON case_results
WHEN NEW.status = 'pending' AND NEW.ord IS NULL
BEGIN SELECT RAISE(ABORT, 'a pending case result needs its processing order'); END;

CREATE TRIGGER case_results_meta_insert BEFORE INSERT ON case_results
WHEN NEW.meta_json IS NOT NULL AND NEW.status <> 'complete'
BEGIN SELECT RAISE(ABORT, 'only a complete case result carries output metadata'); END;

CREATE TRIGGER case_results_meta_update BEFORE UPDATE ON case_results
WHEN NEW.meta_json IS NOT NULL AND NEW.status <> 'complete'
BEGIN SELECT RAISE(ABORT, 'only a complete case result carries output metadata'); END;
"""


# P2.1 production operations.
#  * attempts: cache hit/key, estimated cost + the price-table version it came from, retry round.
#  * case_results / evaluator_results: `retry_round`. A retryable *failed* result may be superseded
#    exactly once per round: its failure is archived in `result_history` first (append-only), the
#    row keeps its id (so every earlier attempt stays attached to it) and takes the new outcome.
#    Nothing else about a result can change, and a successful result can never be superseded. The
#    write-once triggers are replaced by ones that allow exactly that transition.
#  * run_executions: one row per `execute` (its environment, budget, spend, outcome). A `succeeded`
#    run can be reopened only by an execution that names its `finished_at` (`--retry-failed`).
#  * llm_cache: content-addressed provider responses (validated successes only).
OPERATIONS_SQL = """
ALTER TABLE attempts ADD COLUMN cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1));
ALTER TABLE attempts ADD COLUMN cache_key TEXT;
ALTER TABLE attempts ADD COLUMN cost_usd REAL CHECK (cost_usd IS NULL OR cost_usd >= 0);
ALTER TABLE attempts ADD COLUMN price_version TEXT;
ALTER TABLE attempts ADD COLUMN retry_round INTEGER NOT NULL DEFAULT 0 CHECK (retry_round >= 0);
ALTER TABLE case_results ADD COLUMN retry_round INTEGER NOT NULL DEFAULT 0
  CHECK (retry_round >= 0);
ALTER TABLE evaluator_results ADD COLUMN retry_round INTEGER NOT NULL DEFAULT 0
  CHECK (retry_round >= 0);

CREATE TABLE result_history (
  id                  TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL REFERENCES runs(id),
  scope               TEXT NOT NULL CHECK (scope IN ('case', 'evaluator')),
  case_result_id      TEXT NOT NULL REFERENCES case_results(id),
  evaluator_result_id TEXT REFERENCES evaluator_results(id),
  evaluator_key       TEXT,
  retry_round         INTEGER NOT NULL CHECK (retry_round >= 0),
  failure_class       TEXT NOT NULL,
  failure_kind        TEXT NOT NULL,
  failure_message     TEXT NOT NULL,
  retryable           INTEGER NOT NULL CHECK (retryable IN (0, 1)),
  started_at          TEXT,
  finished_at         TEXT,
  duration_ms         INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
  superseded_at       TEXT NOT NULL,
  CHECK (
    (scope = 'case' AND evaluator_result_id IS NULL AND evaluator_key IS NULL)
    OR (scope = 'evaluator' AND evaluator_result_id IS NOT NULL AND evaluator_key IS NOT NULL)
  )
);
CREATE UNIQUE INDEX ux_history_case ON result_history(case_result_id, retry_round)
  WHERE scope = 'case';
CREATE UNIQUE INDEX ux_history_evaluator ON result_history(evaluator_result_id, retry_round)
  WHERE scope = 'evaluator';
CREATE INDEX idx_history_run ON result_history(run_id, scope);
CREATE TRIGGER result_history_insert BEFORE INSERT ON result_history
WHEN COALESCE((SELECT status FROM runs WHERE id = NEW.run_id), '') <> 'running'
BEGIN SELECT RAISE(ABORT, 'history is recorded on a running run'); END;
CREATE TRIGGER result_history_no_update BEFORE UPDATE ON result_history
BEGIN SELECT RAISE(ABORT, 'result history is append-only'); END;
CREATE TRIGGER result_history_no_delete BEFORE DELETE ON result_history
BEGIN SELECT RAISE(ABORT, 'result history is append-only'); END;

CREATE TABLE run_executions (
  id                  TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL REFERENCES runs(id),
  seq                 INTEGER NOT NULL CHECK (seq >= 1),
  started_at          TEXT NOT NULL,
  finished_at         TEXT,
  status_before       TEXT NOT NULL,
  retry_failed        INTEGER NOT NULL CHECK (retry_failed IN (0, 1)),
  reopens_finished_at TEXT,
  reopened_cases      INTEGER NOT NULL DEFAULT 0 CHECK (reopened_cases >= 0),
  reopened_evaluators INTEGER NOT NULL DEFAULT 0 CHECK (reopened_evaluators >= 0),
  units               INTEGER,
  elapsed_s           REAL,
  outcome_status      TEXT,
  stop_reason         TEXT,
  budget_json         TEXT,
  spend_json          TEXT,
  environment_json    TEXT NOT NULL DEFAULT '{}',
  UNIQUE (run_id, seq)
);
CREATE INDEX idx_run_executions_run ON run_executions(run_id, seq);
CREATE TRIGGER run_executions_finish_once BEFORE UPDATE ON run_executions
WHEN OLD.finished_at IS NOT NULL OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id
  OR NEW.seq IS NOT OLD.seq OR NEW.started_at IS NOT OLD.started_at
  OR NEW.status_before IS NOT OLD.status_before OR NEW.retry_failed IS NOT OLD.retry_failed
  OR NEW.reopens_finished_at IS NOT OLD.reopens_finished_at
  OR NEW.environment_json IS NOT OLD.environment_json
BEGIN SELECT RAISE(ABORT, 'an execution record is finished once'); END;
CREATE TRIGGER run_executions_no_delete BEFORE DELETE ON run_executions
BEGIN SELECT RAISE(ABORT, 'execution records are permanent'); END;

CREATE TABLE llm_cache (
  key           TEXT PRIMARY KEY CHECK (length(key) = 64),
  provider      TEXT NOT NULL,
  model         TEXT NOT NULL,
  response_json TEXT NOT NULL,
  created_at    TEXT NOT NULL
) WITHOUT ROWID;

-- a succeeded run is reopened only by an execution that names the finished_at it reopens
DROP TRIGGER runs_legal_transition;
CREATE TRIGGER runs_legal_transition BEFORE UPDATE ON runs
WHEN NOT (
  (OLD.status = 'created' AND NEW.status = 'running')
  OR (OLD.status = 'running' AND NEW.status IN ('succeeded','partial','cancelled','failed'))
  OR (OLD.status IN ('partial','cancelled','failed') AND NEW.status = 'running')
  OR (OLD.status = 'succeeded' AND NEW.status = 'running' AND EXISTS (
        SELECT 1 FROM run_executions x
        WHERE x.run_id = OLD.id AND x.reopens_finished_at = OLD.finished_at))
)
BEGIN SELECT RAISE(ABORT, 'illegal run status change'); END;

-- a failed case result is reopened (pending again) only with its failure archived; every
-- other change to a non-pending result is refused
DROP TRIGGER case_results_write_once;
CREATE TRIGGER case_results_write_once BEFORE UPDATE ON case_results
WHEN NOT (
    (OLD.status = 'pending' AND NEW.status <> 'pending' AND NEW.retry_round = OLD.retry_round)
    OR (OLD.status = 'failed' AND NEW.status = 'pending'
        AND NEW.retry_round = OLD.retry_round + 1
        AND EXISTS (SELECT 1 FROM result_history h WHERE h.case_result_id = OLD.id
                    AND h.scope = 'case' AND h.retry_round = OLD.retry_round))
  )
  OR NEW.id IS NOT OLD.id OR NEW.run_id IS NOT OLD.run_id OR NEW.case_id IS NOT OLD.case_id
  OR NEW.created_at IS NOT OLD.created_at OR NEW.ord IS NOT OLD.ord
  OR COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') <> 'running'
BEGIN SELECT RAISE(ABORT, 'case results are write-once, and only on a running run'); END;

-- a failed evaluator result takes a new outcome, once per round, with its failure archived
DROP TRIGGER evaluator_results_no_update;
CREATE TRIGGER evaluator_results_no_update BEFORE UPDATE ON evaluator_results
WHEN NOT (
  OLD.status = 'failed' AND NEW.status IN ('ok','not_applicable','failed')
  AND NEW.retry_round = OLD.retry_round + 1
  AND NEW.id IS OLD.id AND NEW.case_result_id IS OLD.case_result_id AND NEW.run_id IS OLD.run_id
  AND NEW.evaluator_key IS OLD.evaluator_key AND NEW.created_at IS OLD.created_at
  AND EXISTS (SELECT 1 FROM result_history h WHERE h.evaluator_result_id = OLD.id
              AND h.scope = 'evaluator' AND h.retry_round = OLD.retry_round)
  AND COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') = 'running'
  AND COALESCE((SELECT status FROM case_results WHERE id = OLD.case_result_id), '') = 'complete'
)
BEGIN SELECT RAISE(ABORT, 'evaluator results are write-once'); END;

-- the `skipped` rows of a reopened (pending) case result carry no evidence and may go
DROP TRIGGER evaluator_results_no_delete;
CREATE TRIGGER evaluator_results_no_delete BEFORE DELETE ON evaluator_results
WHEN NOT (
  OLD.status = 'skipped'
  AND COALESCE((SELECT status FROM case_results WHERE id = OLD.case_result_id), '') = 'pending'
  AND COALESCE((SELECT status FROM runs WHERE id = OLD.run_id), '') = 'running'
)
BEGIN SELECT RAISE(ABORT, 'evaluator results are permanent records'); END;
"""


def _backfill_input_hash(conn: sqlite3.Connection) -> None:
    """Record the input hash of every existing case. Cases are immutable, so the immutability
    trigger is lifted for exactly this statement and restored in the same transaction (a failure
    rolls both back)."""
    import json

    from evalkit.hashing import case_input_hash

    conn.execute("DROP TRIGGER cases_no_update")
    rows = conn.execute(
        "SELECT id, prompt, reference, context, relevance_json FROM cases WHERE input_hash IS NULL"
    ).fetchall()
    conn.executemany(
        "UPDATE cases SET input_hash = ? WHERE id = ?",
        [
            (
                case_input_hash(r[1], r[2], r[3], None if r[4] is None else json.loads(r[4])),
                r[0],
            )
            for r in rows
        ],
    )
    conn.execute(
        "CREATE TRIGGER cases_no_update BEFORE UPDATE ON cases "
        "BEGIN SELECT RAISE(ABORT, 'cases are immutable'); END"
    )


def _backfill_rubric_versions(conn: sqlite3.Connection) -> None:
    """Register the rubric version -> content hash of every existing row (earliest first wins).

    A legacy database may already contain one version label used for different rubric
    content (the very problem the registry prevents). Those rows are left untouched and
    logged; the later variant will be refused when next used, with instructions.
    """
    from evalkit.models import Rubric  # local: models must not be needed to import this module

    rows = conn.execute(
        "SELECT rubric_version, rubric_json, MIN(created_at) FROM evaluations "
        "GROUP BY rubric_version, rubric_json ORDER BY MIN(created_at), MIN(rowid)"
    ).fetchall()
    conflicts = unreadable = 0
    for version, rubric_json, first_seen in rows:
        try:
            content_hash = Rubric.model_validate_json(rubric_json).content_hash
        except ValueError:
            unreadable += 1
            continue
        conn.execute(
            "INSERT OR IGNORE INTO rubric_versions (version, content_hash, first_seen_at) "
            "VALUES (?, ?, ?)",
            (version, content_hash, first_seen),
        )
        (stored,) = conn.execute(
            "SELECT content_hash FROM rubric_versions WHERE version = ?", (version,)
        ).fetchone()
        conflicts += stored != content_hash
    if conflicts or unreadable:
        log.warning(
            "rubric version backfill: %d version label(s) already used for different rubric "
            "content, %d stored rubric(s) unreadable; existing rows were left unchanged",
            conflicts,
            unreadable,
        )


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    hook: Callable[[sqlite3.Connection], None] | None = None

    @property
    def checksum(self) -> str:
        text = self.sql + (self.hook.__name__ if self.hook else "")
        return hashlib.sha256(text.encode()).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline", BASELINE_SQL),
    Migration(2, "p0_hardening", P0_SQL, _backfill_rubric_versions),
    Migration(3, "dataset_foundation", DATASETS_SQL),
    Migration(4, "runs_and_results", RUNS_SQL),
    Migration(5, "analysis_foundation", ANALYSIS_SQL),
    Migration(6, "p1_1_hardening", HARDENING_SQL, _backfill_input_hash),
    Migration(7, "p2_1_operations", OPERATIONS_SQL),
)

_V1_EVALUATION_COLUMNS = frozenset(
    "id created_at status error prompt model_output reference_output context rubric_json "
    "rubric_version judge_provider judge_model judge_temperature judge_prompt_version "
    "scores_json overall_score verdict latency_ms metadata_json tags_json".split()
)
_V0_EVALUATION_COLUMNS = _V1_EVALUATION_COLUMNS - {"context"}
_REVIEW_COLUMNS = frozenset("id evaluation_id reviewer score verdict comment created_at".split())


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _statements(sql: str) -> Iterator[str]:
    """Split a script into statements; complete_statement() understands trigger bodies."""
    buffer = ""
    for line in sql.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            if buffer.strip():
                yield buffer
            buffer = ""
    if buffer.strip():
        raise MigrationError(f"incomplete SQL statement in migration: {buffer.strip()[:80]!r}")


def _detect_legacy(conn: sqlite3.Connection) -> str:
    """'v0' (no `context` column) or 'v1'; MigrationError for anything unrecognized."""
    evaluations = _columns(conn, "evaluations")
    if evaluations == _V1_EVALUATION_COLUMNS:
        shape = "v1"
    elif evaluations == _V0_EVALUATION_COLUMNS:
        shape = "v0"
    else:
        raise MigrationError(
            "unrecognized 'evaluations' table (columns: "
            f"{sorted(evaluations)}): not an evalkit database, or an unsupported version. "
            "Refusing to modify it."
        )
    if not _table_exists(conn, "reviews") or _columns(conn, "reviews") != _REVIEW_COLUMNS:
        raise MigrationError(
            "unrecognized or missing 'reviews' table next to a valid 'evaluations' table. "
            "Refusing to modify the database."
        )
    return shape


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, "
    "applied_at TEXT NOT NULL)"
)


def _applied(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    if not _table_exists(conn, "schema_migrations"):
        return {}
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _record(conn: sqlite3.Connection, m: Migration) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
        (m.version, m.name, m.checksum, datetime.now(UTC).isoformat()),
    )


def _backup(conn: sqlite3.Connection, path: Path, from_version: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    dest = path.with_name(f"{path.name}.bak-v{from_version}-{stamp}")
    target = sqlite3.connect(dest)
    try:
        conn.backup(target)
    finally:
        target.close()
    os.chmod(dest, 0o600)
    log.info("backed up %s to %s before migrating", path, dest)
    return dest


def migrate(
    conn: sqlite3.Connection,
    *,
    path: Path | None = None,
    backup: bool = True,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> None:
    """Bring `conn`'s database to the latest schema. Safe to call repeatedly and concurrently."""
    latest = migrations[-1].version
    by_version = {m.version: m for m in migrations}
    previous_isolation = conn.isolation_level
    conn.isolation_level = None  # explicit BEGIN/COMMIT below
    try:
        applied = _applied(conn)
        if applied and max(applied) > latest:
            raise MigrationError(
                f"database schema version {max(applied)} is newer than this evalkit supports "
                f"({latest}); upgrade evalkit"
            )
        for version, (name, checksum) in applied.items():
            if version in by_version and by_version[version].checksum != checksum:
                raise MigrationError(
                    f"migration {version} ({name}) was modified after it was applied to this "
                    "database; refusing to continue"
                )

        legacy_shape = None
        if not applied and _table_exists(conn, "evaluations"):
            legacy_shape = _detect_legacy(conn)

        pending = [m for m in migrations if m.version not in applied]
        adopt = pending[0] if legacy_shape is not None else None  # the baseline, already present
        if adopt is not None:
            pending = pending[1:]
        if not pending and adopt is None:
            return

        if backup and path is not None and (applied or legacy_shape is not None):
            _backup(conn, path, max(applied, default=0))

        if adopt is not None:
            with _transaction(conn):
                conn.execute(_CREATE_TABLE)
                if not conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (adopt.version,)
                ).fetchone():
                    if legacy_shape == "v0":
                        conn.execute("ALTER TABLE evaluations ADD COLUMN context TEXT")
                    _record(conn, adopt)
        for m in pending:
            with _transaction(conn):
                conn.execute(_CREATE_TABLE)
                # another process may have applied it while we waited for the write lock
                if conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = ?", (m.version,)
                ).fetchone():
                    continue
                try:
                    for statement in _statements(m.sql):
                        conn.execute(statement)
                    if m.hook is not None:
                        m.hook(conn)
                except sqlite3.Error as e:
                    raise MigrationError(f"migration {m.version} ({m.name}) failed: {e}") from e
                _record(conn, m)
    finally:
        conn.isolation_level = previous_isolation
