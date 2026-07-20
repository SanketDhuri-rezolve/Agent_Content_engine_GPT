"""Tests for orchestrator.watchdog — the job-level bound against a
permanently-lost chord member hanging a job forever (see watchdog.py's module
docstring). is_job_overdue is pure (no DB/Celery); check_job_timeout itself is
exercised against a real Postgres via the shared db_session fixture."""

from datetime import datetime, timedelta, timezone

import pytest

from models.enums import JobStatus, SegmentStatus
from orchestrator.state import create_job, create_segments, transition_job_status
from orchestrator.watchdog import check_job_timeout, is_job_overdue


class TestIsJobOverdue:
    def test_well_within_budget_is_not_overdue(self):
        created_at = datetime.now(timezone.utc)
        now = created_at + timedelta(seconds=10)
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=60, now=now) is False

    def test_past_sla_but_within_grace_is_not_overdue(self):
        created_at = datetime.now(timezone.utc)
        now = created_at + timedelta(seconds=250)  # past 240s SLA, within +60s grace
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=60, now=now) is False

    def test_past_sla_and_grace_is_overdue(self):
        created_at = datetime.now(timezone.utc)
        now = created_at + timedelta(seconds=301)  # past 240 + 60
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=60, now=now) is True

    def test_exactly_at_deadline_is_overdue(self):
        created_at = datetime.now(timezone.utc)
        now = created_at + timedelta(seconds=300)  # exactly 240 + 60
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=60, now=now) is True

    def test_zero_grace_period(self):
        created_at = datetime.now(timezone.utc)
        just_before = created_at + timedelta(seconds=239)
        just_after = created_at + timedelta(seconds=241)
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=0, now=just_before) is False
        assert is_job_overdue(created_at, sla_target_seconds=240, grace_period_seconds=0, now=just_after) is True


@pytest.fixture
def stuck_job(db_session):
    """A job left in running_segments with one completed and two still-pending
    segments, created far enough in the past that any real sla_target_seconds/
    grace_period combination will consider it overdue."""
    job = create_job(db_session, source_video_url="https://example.com/stuck.mp4", sla_target_seconds=1, total_segments=3)
    segment_plan = [
        {"sequence_index": i, "start_ts": float(i * 10), "end_ts": float((i + 1) * 10), "overlap_start": float(i * 10), "overlap_end": float((i + 1) * 10)}
        for i in range(3)
    ]
    segments = create_segments(db_session, job, segment_plan)
    segments[0].status = SegmentStatus.completed
    transition_job_status(db_session, job, JobStatus.running_segments)
    # Backdate created_at well past any sla_target_seconds(1) + grace_period_seconds combination.
    job.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db_session.commit()
    return job


class TestCheckJobTimeout:
    def test_force_fails_overdue_job_and_marks_stuck_segments_timeout(self, db_session, stuck_job):
        result = check_job_timeout(str(stuck_job.id))

        assert result["action"] == "force_failed"
        assert result["stuck_segment_count"] == 2  # the 2 still-pending segments, not the 1 already completed

        db_session.refresh(stuck_job)
        assert stuck_job.status == JobStatus.failed

        from models.orm import Segment

        segments = db_session.query(Segment).filter(Segment.job_id == stuck_job.id).order_by(Segment.sequence_index).all()
        assert segments[0].status == SegmentStatus.completed  # untouched — it finished on its own
        assert segments[1].status == SegmentStatus.timeout
        assert segments[2].status == SegmentStatus.timeout
        assert segments[1].error is not None and "watchdog" in segments[1].error

    def test_no_op_on_already_completed_job(self, db_session):
        job = create_job(db_session, source_video_url="https://example.com/done.mp4", sla_target_seconds=1, total_segments=1)
        segment_plan = [{"sequence_index": 0, "start_ts": 0.0, "end_ts": 10.0, "overlap_start": 0.0, "overlap_end": 10.0}]
        create_segments(db_session, job, segment_plan)
        transition_job_status(db_session, job, JobStatus.completed)
        job.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        db_session.commit()

        result = check_job_timeout(str(job.id))

        assert result["action"] == "no_op"
        db_session.refresh(job)
        assert job.status == JobStatus.completed  # untouched

    def test_no_op_on_already_failed_job(self, db_session):
        job = create_job(db_session, source_video_url="https://example.com/failed.mp4", sla_target_seconds=1, total_segments=1)
        segment_plan = [{"sequence_index": 0, "start_ts": 0.0, "end_ts": 10.0, "overlap_start": 0.0, "overlap_end": 10.0}]
        create_segments(db_session, job, segment_plan)
        transition_job_status(db_session, job, JobStatus.failed)
        db_session.commit()

        result = check_job_timeout(str(job.id))

        assert result["action"] == "no_op"

    def test_unknown_job_id_does_not_raise(self, db_session):
        import uuid

        result = check_job_timeout(str(uuid.uuid4()))
        assert result["action"] == "job_not_found"

    def test_not_yet_overdue_is_a_no_op_not_a_force_fail(self, db_session):
        """A job created just now, well within its budget, should never be
        force-failed even if check_job_timeout is somehow invoked early
        (clock skew, a misconfigured countdown)."""
        job = create_job(db_session, source_video_url="https://example.com/fresh.mp4", sla_target_seconds=240, total_segments=1)
        segment_plan = [{"sequence_index": 0, "start_ts": 0.0, "end_ts": 10.0, "overlap_start": 0.0, "overlap_end": 10.0}]
        create_segments(db_session, job, segment_plan)
        transition_job_status(db_session, job, JobStatus.running_segments)
        db_session.commit()

        result = check_job_timeout(str(job.id))

        assert result["action"] == "no_op_not_yet_overdue"
        db_session.refresh(job)
        assert job.status == JobStatus.running_segments  # untouched, not force-failed
