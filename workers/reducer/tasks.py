"""Stage 5 (reducer) Celery task.

Queue: "reducer". Task name: "workers.reducer.tasks.reduce_job" (fixed — do
not rename, see the cross-stage Celery contract in CLAUDE.md).

This is a chord callback: Celery invokes it with the chord group's list of
results prepended as the first positional argument, with `job_id` bound via
`.s(job_id)` at construction time (see orchestrator/pipeline.py:
`chord(segment_chains)(reduce_job.s(job_id) | rank_and_persist.s(job_id))`).
Each item in that list matches models.schemas.SegmentPipelineResult, OR may
be `None`/malformed if a chain member hard-failed — workers.reducer.logic
handles both explicitly.

Failure-handling choice (documented per the build brief's "pick one and
explain" instruction): on InsufficientSegmentsError, or literally any other
exception raised while reducing, this task:
  1. logs at error level with full context (job_id, which segments were bad,
     why) via structlog,
  2. transitions the Job to JobStatus.failed (models.orm.Job has no
     dedicated error-message column — see the TODO below — so the message
     itself lives in the structured log line and in the re-raised
     exception's text, which Celery's result backend captures as this task's
     traceback),
  3. RE-RAISES (does not return a "clearly-marked failure dict").

Why re-raise rather than return a failure dict: this task's return value
flows directly into workers.ranker.tasks.rank_and_persist via the chain
`reduce_job.s(job_id) | rank_and_persist.s(job_id)` (see
orchestrator/pipeline.py). rank_and_persist unconditionally sets
JobStatus.completed on success. If reduce_job instead returned a
failure-shaped dict, rank_and_persist would still run against it (chains
only stop on an exception, not on a "soft failure" return value), silently
persisting zero highlights AND overwriting the JobStatus.failed we just set
back to JobStatus.completed — exactly the misleadingly-confident outcome
this stage must never produce. Re-raising causes Celery to mark this chain
member FAILED and skip rank_and_persist entirely, so the JobStatus.failed
transition we already made is the one that sticks.
"""

import time
import uuid
from typing import Any

import structlog

from models.db import session_scope
from models.enums import JobStatus
from orchestrator.celery_app import celery_app
from orchestrator.state import sync_segment_statuses_from_pipeline_results, transition_job_status
from workers.reducer.logic import InsufficientSegmentsError, reduce_segment_results

logger = structlog.get_logger()


@celery_app.task(name="workers.reducer.tasks.reduce_job")
def reduce_job(segment_pipeline_results: list[Any], job_id: str) -> dict:
    log = logger.bind(job_id=job_id, stage="reducer")
    log.info("stage.started", segment_result_count=len(segment_pipeline_results or []))
    start = time.monotonic()

    # Step 4: segment_worker/span_builder/scorer are stateless (pure JSON in/
    # out) and never touch Postgres — this is the one place with the full
    # per-segment picture, so it's the natural place to sync each Segment's
    # final status/error to the DB. Runs regardless of whether reduction
    # itself succeeds below: a segment's own outcome is independent of
    # whether the JOB as a whole ends up reducible.
    try:
        with session_scope() as sync_session:
            synced = sync_segment_statuses_from_pipeline_results(sync_session, segment_pipeline_results)
        log.info("segment_status_sync.completed", synced_count=synced)
    except Exception as exc:  # noqa: BLE001 - a DB-sync hiccup must not mask the actual reduce
        log.error("segment_status_sync.failed", error=str(exc))

    try:
        output = reduce_segment_results(segment_pipeline_results, job_id)
    except InsufficientSegmentsError as exc:
        elapsed = time.monotonic() - start
        log.error(
            "stage.failed",
            elapsed_seconds=elapsed,
            reason="insufficient_segments",
            error=str(exc),
        )
        _fail_job(job_id, log)
        raise
    except Exception as exc:  # noqa: BLE001 - must never silently vanish; see module docstring
        elapsed = time.monotonic() - start
        log.error(
            "stage.failed",
            elapsed_seconds=elapsed,
            reason="unexpected_reducer_error",
            error=str(exc),
        )
        _fail_job(job_id, log)
        raise

    elapsed = time.monotonic() - start
    log.info(
        "stage.completed",
        elapsed_seconds=elapsed,
        span_count=len(output.get("spans", [])),
        degraded_segment_count=len(output.get("degraded_segment_ids", [])),
        dropped_duplicate_count=output.get("dropped_duplicate_count", 0),
    )
    return output


def _fail_job(job_id: str, log) -> None:
    """Best-effort transition of the Job row to JobStatus.failed. Wrapped in
    its own try/except: a DB hiccup (or an unparseable job_id) while
    recording the failure must not mask or replace the original reducer
    exception (the `raise` in the caller still happens either way — this
    function never re-raises)."""
    try:
        job_uuid = uuid.UUID(str(job_id))
        with session_scope() as session:
            transition_job_status(session, job_uuid, JobStatus.failed)
    except Exception as db_exc:  # noqa: BLE001 - recording the failure must not itself crash unhandled
        log.error("job.status_transition_failed", target_status=JobStatus.failed.value, error=str(db_exc))
