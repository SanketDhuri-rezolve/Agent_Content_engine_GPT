"""DB helper functions for Job/Segment lifecycle transitions.

Every function here takes an already-open SQLAlchemy `Session` as its first
argument — callers open that session via `models.db.session_scope()` (see
orchestrator/pipeline.py for the canonical usage) so a whole unit of work
commits/rolls back atomically. These helpers never open, commit, or close a
session themselves; they only `flush()` so generated PKs/FKs (e.g. a new
Job's `id`) are available to the caller immediately without a full commit.

Compatibility note: the `job_id` / `segment_id` parameters below accept
either the raw id (UUID or str) OR the already-loaded ORM instance
(`models.orm.Job` / `models.orm.Segment`). This is deliberately permissive:
other Step-1 modules built in parallel against this same contract (e.g.
api/main.py's `state.create_segments(db, job, segment_plan)` and
workers/ranker/tasks.py's `transition_job_status(session, job_row, ...)`)
already call these functions with the ORM object in hand rather than
re-fetching by id, and there's no reason to force a redundant round-trip
just to satisfy a stricter type.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from models.enums import JobStatus, SegmentStatus
from models.orm import Job, Segment

JobRef = Job | uuid.UUID | str
SegmentRef = Segment | uuid.UUID | str


def _resolve_job(session: Session, job_ref: JobRef) -> Job:
    if isinstance(job_ref, Job):
        return job_ref
    job = session.get(Job, job_ref)
    if job is None:
        raise ValueError(f"Job {job_ref} not found")
    return job


def _resolve_segment(session: Session, segment_ref: SegmentRef) -> Segment:
    if isinstance(segment_ref, Segment):
        return segment_ref
    segment = session.get(Segment, segment_ref)
    if segment is None:
        raise ValueError(f"Segment {segment_ref} not found")
    return segment


def create_job(
    session: Session,
    source_video_url: str,
    sla_target_seconds: int,
    total_segments: int,
) -> Job:
    """Creates and flushes a new Job row (status defaults to
    JobStatus.created). Returns the ORM instance with `id`/`created_at`
    populated."""
    job = Job(
        source_video_url=source_video_url,
        sla_target_seconds=sla_target_seconds,
        total_segments=total_segments,
        status=JobStatus.created,
    )
    session.add(job)
    session.flush()
    return job


def create_segments(session: Session, job_id: JobRef, segment_plan: list[dict]) -> list[Segment]:
    """Creates and flushes one Segment row per entry in `segment_plan` (the
    shape returned by orchestrator.splitter.compute_segment_plan), all
    pointing at `job_id`. Status defaults to SegmentStatus.pending. Returns
    the ORM instances in the same order as `segment_plan`."""
    resolved_job_id = job_id.id if isinstance(job_id, Job) else job_id
    segments = [
        Segment(
            job_id=resolved_job_id,
            sequence_index=item["sequence_index"],
            start_ts=item["start_ts"],
            end_ts=item["end_ts"],
            overlap_start=item["overlap_start"],
            overlap_end=item["overlap_end"],
            status=SegmentStatus.pending,
        )
        for item in segment_plan
    ]
    session.add_all(segments)
    session.flush()
    return segments


def transition_job_status(session: Session, job_id: JobRef, new_status: JobStatus) -> Job:
    """Moves a Job to `new_status`. Also stamps `completed_at` when
    transitioning to JobStatus.completed (the only terminal-success status;
    JobStatus.failed does not get a completed_at, matching models.orm.Job's
    docstring intent that completed_at means "finished successfully")."""
    job = _resolve_job(session, job_id)
    job.status = new_status
    if new_status == JobStatus.completed:
        job.completed_at = datetime.now(timezone.utc)
    session.flush()
    return job


def mark_segment_started(session: Session, segment_id: SegmentRef) -> Segment:
    segment = _resolve_segment(session, segment_id)
    segment.status = SegmentStatus.running
    segment.worker_started_at = datetime.now(timezone.utc)
    session.flush()
    return segment


def mark_segment_completed(session: Session, segment_id: SegmentRef) -> Segment:
    segment = _resolve_segment(session, segment_id)
    segment.status = SegmentStatus.completed
    segment.worker_completed_at = datetime.now(timezone.utc)
    session.flush()
    return segment


def mark_segment_failed(session: Session, segment_id: SegmentRef, error: str) -> Segment:
    segment = _resolve_segment(session, segment_id)
    segment.status = SegmentStatus.failed
    segment.error = error
    segment.worker_completed_at = datetime.now(timezone.utc)
    session.flush()
    return segment


def mark_segment_timeout(session: Session, segment_id: SegmentRef, error: str) -> Segment:
    segment = _resolve_segment(session, segment_id)
    segment.status = SegmentStatus.timeout
    segment.error = error
    segment.worker_completed_at = datetime.now(timezone.utc)
    session.flush()
    return segment


def sync_segment_statuses_from_pipeline_results(session: Session, segment_pipeline_results: list) -> int:
    """Step 4: called by workers.reducer.tasks.reduce_job once it has the
    full picture (every chain member's final SegmentPipelineResult-shaped
    dict). segment_worker/span_builder/scorer are deliberately stateless
    (pure JSON in/out — see their own module docstrings), so nothing writes a
    Segment's final status/error to Postgres before this point; without this,
    every Segment row would stay `pending` forever regardless of how the job
    actually went, even after it completes successfully.

    A bare `None` entry (hard chain failure — see workers/reducer/logic.py's
    classify_segments) has no segment_id to key off of and is skipped; that
    row stays `pending` unless orchestrator.watchdog.check_job_timeout later
    marks it `timeout` (only happens if the whole job also fails to complete
    in time — a hard-failed-but-the-job-still-succeeded segment has no
    id-bearing record anywhere to update).

    Returns the number of Segment rows actually updated."""
    updated = 0
    for entry in segment_pipeline_results or []:
        if not isinstance(entry, dict):
            continue
        segment_id = entry.get("segment_id")
        status = entry.get("status")
        if not segment_id:
            continue
        try:
            segment_uuid = uuid.UUID(str(segment_id))
        except (ValueError, AttributeError, TypeError):
            continue

        is_completed = status == SegmentStatus.completed.value or status == SegmentStatus.completed
        is_timeout = status == SegmentStatus.timeout.value or status == SegmentStatus.timeout
        try:
            if is_completed:
                mark_segment_completed(session, segment_uuid)
            elif is_timeout:
                mark_segment_timeout(session, segment_uuid, entry.get("error") or "segment reported status=timeout")
            else:
                mark_segment_failed(session, segment_uuid, entry.get("error") or f"segment reported status={status!r}")
            updated += 1
        except ValueError:
            # _resolve_segment raises ValueError if segment_id doesn't match
            # any row (shouldn't happen for a contract-conformant entry, but
            # a cosmetic DB-sync issue must never crash the reducer's own
            # success path).
            continue
    return updated


def mark_stuck_segments_timeout(session: Session, job_id: JobRef, error: str) -> list[Segment]:
    """Step 4: called by orchestrator.watchdog.check_job_timeout when a job
    has exceeded its time budget without reaching a terminal status. Any
    Segment still in `pending` or `running` at that point has no chord result
    that will ever arrive (that's the whole reason the job is stuck) — this
    marks them `SegmentStatus.timeout` (distinct from `.failed`: nothing about
    THIS segment necessarily errored, the surrounding job simply gave up
    waiting for it) so the DB record is consistent with the JobStatus.failed
    transition the watchdog makes right after calling this. Segments already
    `completed`/`failed` are left untouched — they finished on their own,
    they just weren't enough (or arrived too late) to save the job."""
    resolved_job_id = job_id.id if isinstance(job_id, Job) else job_id
    stuck_segments = (
        session.query(Segment)
        .filter(
            Segment.job_id == resolved_job_id,
            Segment.status.in_([SegmentStatus.pending, SegmentStatus.running]),
        )
        .all()
    )
    for segment in stuck_segments:
        segment.status = SegmentStatus.timeout
        segment.error = error
        segment.worker_completed_at = datetime.now(timezone.utc)
    session.flush()
    return stuck_segments
