"""Job orchestration entrypoint.

`run_job` is the one place that assembles the full per-job Celery graph:
it loads a Job + its already-persisted Segments from Postgres (Segments are
created ahead of time by the caller — see api/main.py's POST /jobs handler,
which calls orchestrator.splitter.compute_segment_plan +
orchestrator.state.create_job/create_segments before enqueuing this task),
builds one Celery chain per segment (segment_worker -> span_builder ->
scorer), fans all of those chains out under a single chord, and wires the
chord callback to reduce_job -> rank_and_persist.

Queue note (deviation flagged per the build brief's "your call, note it in
your report" instruction): this task runs on a dedicated "orchestrator"
queue (see orchestrator/celery_app.py's task_routes), matching the
one-queue-per-stage pattern already used by the other 5 stages, rather than
the default "celery" queue. docker-compose.yml has an `orchestrator` service
consuming this queue (added during Step 1 integration review — it was
missing from the initial parallel build).

Job status lifecycle owned by *this* task: created -> queued -> sharding ->
running_segments. This task returns as soon as the chord is dispatched
(fire-and-forget from Celery's perspective), so it is not present when the
chord callback actually executes later — JobStatus.reducing / ranking /
completed are transitioned by workers.reducer.tasks.reduce_job and
workers.ranker.tasks.rank_and_persist respectively (rank_and_persist is
documented in the fixed contract as the one that sets JobStatus.completed).
"""

import time
import uuid

import structlog
from celery import chain, chord

from config import get_settings
from models.db import session_scope
from models.enums import JobStatus
from models.orm import Job, Segment
from orchestrator.celery_app import celery_app
from orchestrator.state import transition_job_status
from orchestrator.watchdog import schedule_watchdog
from workers.global_selector.tasks import select_and_score_job
from workers.ranker.tasks import rank_and_persist
from workers.reducer.tasks import reduce_job
from workers.scorer.tasks import score_spans
from workers.segment_worker.tasks import run_segment_worker
from workers.span_builder.tasks import build_spans

logger = structlog.get_logger()


def _segment_payload(job: Job, segment: Segment) -> dict:
    """Builds the JSON-safe models.schemas.SegmentTaskPayload dict for one
    segment. UUIDs become str because Celery serializes task args as JSON."""
    return {
        "job_id": str(job.id),
        "segment_id": str(segment.id),
        "sequence_index": segment.sequence_index,
        "source_video_url": job.source_video_url,
        "start_ts": segment.start_ts,
        "end_ts": segment.end_ts,
        "overlap_start": segment.overlap_start,
        "overlap_end": segment.overlap_end,
    }


@celery_app.task(name="orchestrator.pipeline.run_job")
def run_job(job_id: str) -> dict:
    """Loads the job + its segments, fans out one chain per segment under a
    chord, and wires the chord callback through reduce_job -> rank_and_persist.

    Returns a small dict describing what was dispatched (job_id, the async
    chord's id, segment_count) — this task's own return value is not part of
    the fixed inter-stage contract (nothing downstream consumes it; the
    contract only governs the per-segment chain + chord callback shapes).
    """
    start = time.monotonic()
    log = logger.bind(job_id=job_id, stage="orchestrator.run_job")
    log.info("stage.started")

    with session_scope() as session:
        job = session.get(Job, uuid.UUID(str(job_id)))
        if job is None:
            log.error("stage.failed", error="job_not_found", elapsed_seconds=time.monotonic() - start)
            raise ValueError(f"Job {job_id} not found")

        transition_job_status(session, job, JobStatus.queued)
        log.info("job.status_transition", status=JobStatus.queued.value)

        segments = (
            session.query(Segment)
            .filter(Segment.job_id == job.id)
            .order_by(Segment.sequence_index)
            .all()
        )
        if not segments:
            transition_job_status(session, job, JobStatus.failed)
            log.error("stage.failed", error="no_segments", elapsed_seconds=time.monotonic() - start)
            raise ValueError(f"Job {job_id} has no segments")

        transition_job_status(session, job, JobStatus.sharding)
        log.info("job.status_transition", status=JobStatus.sharding.value)

        payloads = [_segment_payload(job, segment) for segment in segments]
        segment_count = len(segments)
        sla_target_seconds = job.sla_target_seconds  # captured before the session closes

        transition_job_status(session, job, JobStatus.running_segments)
        log.info("job.status_transition", status=JobStatus.running_segments.value)

    # Global-memory pipeline (config.Settings.use_global_memory_pipeline,
    # opt-in): scoring moves OUT of the per-segment chain and into a single
    # job-wide chord callback (workers.global_selector.tasks
    # .select_and_score_job) that can see every segment's candidates + local
    # memory at once before deciding what to score — see that module's
    # docstring. The default pipeline is completely unaffected: this only
    # changes which tasks are chained/chorded, not any task's own contract.
    use_global_memory_pipeline = get_settings().use_global_memory_pipeline

    segment_chains = [
        chain(
            run_segment_worker.s(payload),
            # overlap_start/overlap_end must be bound here explicitly:
            # models.schemas.SegmentWorkerOutput (build_spans's chained input)
            # does not carry them, so without this the reducer's touches_boundary
            # dedup logic would silently never engage in the real pipeline.
            # source_video_url is bound the same way — CandidateSpanPayload
            # needs it so Gemma4Scorer can crop/save each span's actual
            # video+audio clip and attach real keyframe images/audio to the
            # multimodal scoring call, not just the aggregate feature_vector.
            build_spans.s(
                overlap_start=payload["overlap_start"],
                overlap_end=payload["overlap_end"],
                source_video_url=payload["source_video_url"],
            ),
            *([] if use_global_memory_pipeline else [score_spans.s()]),
        )
        for payload in payloads
    ]

    callback = (
        (select_and_score_job.s(job_id) | reduce_job.s(job_id) | rank_and_persist.s(job_id))
        if use_global_memory_pipeline
        else (reduce_job.s(job_id) | rank_and_persist.s(job_id))
    )
    async_result = chord(segment_chains)(callback)

    # Step 4: independent bound on the whole job, in case a chord member is
    # permanently lost and reduce_job never fires at all — see
    # orchestrator/watchdog.py's module docstring for why the reducer's own
    # fallback logic can't cover that case on its own.
    schedule_watchdog(job_id, sla_target_seconds)

    elapsed = time.monotonic() - start
    log.info("stage.completed", elapsed_seconds=elapsed, segment_count=segment_count)
    return {"job_id": job_id, "chord_id": async_result.id, "segment_count": segment_count}
