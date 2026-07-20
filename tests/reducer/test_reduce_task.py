"""Celery/DB-level adversarial coverage for workers.reducer.tasks.reduce_job.

tests/reducer/test_reduce_logic.py and test_reduce_adversarial.py cover the
pure `workers.reducer.logic` functions in isolation. This module instead
drives the actual `@celery_app.task`-decorated `reduce_job` against a real
Postgres row (via tests/conftest.py's db_session/postgres_available
fixtures, auto-skipped when no Postgres is reachable) to answer the one
question pure-logic tests structurally cannot: when `reduce_job` fails a job
(InsufficientSegmentsError, or any other exception), does the Job row it
touched actually end up in `JobStatus.failed` — or can it be left stuck in
whatever mid-pipeline status it was in (e.g. `running_segments`), which
would look to any poller/UI like a job that is silently still "running"
forever?
"""

import uuid

import pytest

from models.enums import JobStatus, SegmentStatus
from models.orm import Job
from orchestrator.state import create_job, transition_job_status
from workers.reducer.logic import InsufficientSegmentsError
from workers.reducer.tasks import reduce_job


def _segment_result(segment_id, sequence_index, *, status=SegmentStatus.completed.value, scored_spans=None):
    return {
        "segment_id": str(segment_id),
        "sequence_index": sequence_index,
        "status": status,
        "scored_spans": scored_spans or [],
        "error": None,
    }


def test_insufficient_segments_leaves_job_failed_not_orphaned_running(db_session):
    job = create_job(
        db_session,
        source_video_url="file:///tmp/adversarial_test_movie.mp4",
        sla_target_seconds=240,
        total_segments=4,
    )
    # Simulate the real pipeline state at the moment reduce_job would fire:
    # mid-pipeline, not yet resolved either way.
    transition_job_status(db_session, job, JobStatus.running_segments)
    db_session.commit()

    # 0/4 usable -> guaranteed below the default 0.6 required fraction,
    # regardless of what it's configured to in this environment.
    bad_results = [None, None, None, None]

    with pytest.raises(InsufficientSegmentsError):
        reduce_job(bad_results, job_id=str(job.id))

    # reduce_job's failure handling opens its OWN session (models.db.session_scope)
    # to record the failure — refresh this test's session to see that commit.
    db_session.expire_all()
    refreshed = db_session.get(Job, job.id)

    assert refreshed is not None
    assert refreshed.status == JobStatus.failed
    # completed_at is documented (orchestrator/state.py) to mean "finished
    # successfully" — a failed job must never get one, or a naive UI reading
    # "completed_at is set" as "done" would misreport a failed job as done.
    assert refreshed.completed_at is None


def test_zero_length_segment_results_list_also_leaves_job_failed_not_orphaned(db_session):
    """Distinct code path from the "all entries present but None/degraded"
    case above: `total == len(segment_pipeline_results) == 0` hits
    classify_segments' `total == 0` branch directly (a chord over an empty
    header — e.g. a job sharded into zero segments) rather than the
    fraction-below-threshold branch. Must still fail the job cleanly."""
    job = create_job(
        db_session,
        source_video_url="file:///tmp/adversarial_test_movie_3.mp4",
        sla_target_seconds=240,
        total_segments=0,
    )
    transition_job_status(db_session, job, JobStatus.running_segments)
    db_session.commit()

    with pytest.raises(InsufficientSegmentsError):
        reduce_job([], job_id=str(job.id))

    db_session.expire_all()
    refreshed = db_session.get(Job, job.id)

    assert refreshed.status == JobStatus.failed
    assert refreshed.completed_at is None


def test_unexpected_exception_during_reduce_also_fails_the_job_not_orphaned(db_session, monkeypatch):
    """Same DB-consistency guarantee, but via the generic `except Exception`
    branch rather than the typed InsufficientSegmentsError branch — these
    are two separate code paths in reduce_job and both must leave the job
    consistently failed, not just the one with a dedicated exception type."""
    import workers.reducer.tasks as tasks_module

    job = create_job(
        db_session,
        source_video_url="file:///tmp/adversarial_test_movie_2.mp4",
        sla_target_seconds=240,
        total_segments=1,
    )
    transition_job_status(db_session, job, JobStatus.reducing)
    db_session.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated unexpected reducer crash")

    monkeypatch.setattr(tasks_module, "reduce_segment_results", _boom)

    with pytest.raises(RuntimeError):
        reduce_job([_segment_result(uuid.uuid4(), 0)], job_id=str(job.id))

    db_session.expire_all()
    refreshed = db_session.get(Job, job.id)

    assert refreshed.status == JobStatus.failed
    assert refreshed.completed_at is None
