"""Step 4: job-level watchdog.

Why this exists: workers.reducer.tasks.reduce_job (the chord callback) only
ever runs once EVERY chord member has returned some result — a soft/catchable
failure (an adapter exception, a Celery SoftTimeLimitExceeded) already
produces a fail-soft dict that lets the chord complete normally, and the
reducer's own classify_segments/InsufficientSegmentsError logic (see
workers/reducer/logic.py) handles that gracefully. But if a chord member is
genuinely LOST — a worker process is killed (SIGKILL, OOM, node failure) with
no other worker of that queue available to redeliver the task to (even with
task_acks_late=True/task_reject_on_worker_lost=True, see
orchestrator/celery_app.py) — no result for that member EVER arrives, the
chord callback never fires, and the job would otherwise hang forever
regardless of how good the reducer's own fallback logic is: there is no
Python code running that could time out, because reduce_job is simply never
invoked.

This module is the independent, orchestrator-level bound on that: after
dispatching the chord in orchestrator.pipeline.run_job, a single one-shot
`check_job_timeout` task is scheduled `sla_target_seconds +
config.Settings.watchdog_grace_period_seconds` seconds later. If the job
hasn't reached a terminal status (completed/failed) by then, this task force-
fails it — marking any still-pending/running segments as SegmentStatus.timeout
(orchestrator.state.mark_stuck_segments_timeout) and the Job as
JobStatus.failed — rather than leaving it to hang indefinitely.

Deliberately does NOT try to reconstruct a partial ranked result from
whatever segments DID complete: doing that would require persisting each
segment's individual Celery task id and querying the result backend directly
(non-trivial additional plumbing) for a benefit — an automatic partial re-rank
of a job that's already this far off the rails — that isn't what "degrade
gracefully instead of hanging" requires. A bounded, clearly-explained failure
that the caller can act on (retry the whole job) is graceful; guessing at a
best-effort partial result from a job whose orchestration already misbehaved
is a different, larger feature this brief doesn't ask for.
"""

import time
import uuid
from datetime import datetime, timezone

import structlog

from models.db import session_scope
from models.enums import JobStatus
from models.orm import Job
from orchestrator.celery_app import celery_app
from orchestrator.state import mark_stuck_segments_timeout, transition_job_status

logger = structlog.get_logger()

_TERMINAL_STATUSES = {JobStatus.completed, JobStatus.failed}


def _status_str(status: JobStatus | str) -> str:
    """models.orm.Job.status is declared as a plain String(32) column, not a
    native SQL enum type — SQLAlchemy returns it as a bare Python str once
    loaded from the DB (session.get(...) never gives back a JobStatus
    instance), so `.value` isn't available on it. Handles both forms
    (matches the same defensive str-or-enum pattern already used in
    workers/reducer/logic.py's classify_segments, for the same reason)."""
    return status.value if isinstance(status, JobStatus) else str(status)


def is_job_overdue(
    created_at: datetime,
    sla_target_seconds: float,
    grace_period_seconds: float,
    now: datetime,
) -> bool:
    """Pure predicate (no DB/Celery) so the watchdog's actual timing logic is
    unit-testable without a running worker or Postgres. `now` is a parameter
    rather than datetime.now() internally for the same reason (deterministic
    tests)."""
    deadline = created_at.timestamp() + sla_target_seconds + grace_period_seconds
    return now.timestamp() >= deadline


@celery_app.task(name="orchestrator.watchdog.check_job_timeout")
def check_job_timeout(job_id: str) -> dict:
    """Scheduled via .apply_async(countdown=...) — see
    orchestrator/pipeline.py's run_job. Idempotent: if the job already
    reached a terminal status (the overwhelmingly common case — this check
    firing at all means the job finished right around its own SLA budget,
    which is expected, not a bug), this is a no-op."""
    log = logger.bind(job_id=job_id, stage="watchdog")

    with session_scope() as session:
        job = session.get(Job, uuid.UUID(str(job_id)))
        if job is None:
            log.error("watchdog.job_not_found")
            return {"job_id": job_id, "action": "job_not_found"}

        if job.status in _TERMINAL_STATUSES:
            log.info("watchdog.no_op", job_status=_status_str(job.status))
            return {"job_id": job_id, "action": "no_op", "job_status": _status_str(job.status)}

        overdue = is_job_overdue(
            job.created_at,
            job.sla_target_seconds,
            _grace_period_seconds(),
            datetime.now(timezone.utc),
        )
        if not overdue:
            # Should not normally happen (the countdown scheduled by
            # schedule_watchdog is derived from this exact same
            # sla_target_seconds + grace period) — a slow watchdog queue or
            # clock skew could fire this early. Deliberately does NOT
            # self-reschedule (that would risk an unbounded recheck loop if
            # something is systematically wrong with the countdown
            # calculation) — just log and leave the job alone. It is still
            # running within its own budget; the worst case here is simply
            # that this particular early, spurious check does nothing, not
            # that a legitimately-stuck job goes unnoticed (schedule_watchdog
            # is only ever called once per job, from a countdown that was
            # already correct when computed).
            log.info("watchdog.not_yet_overdue_no_op", job_status=_status_str(job.status))
            return {"job_id": job_id, "action": "no_op_not_yet_overdue"}

        previous_status = _status_str(job.status)  # captured before transition_job_status mutates it below
        error = (
            f"watchdog: job did not reach a terminal status within "
            f"sla_target_seconds={job.sla_target_seconds} + "
            f"grace_period_seconds={_grace_period_seconds()} of creation — "
            f"at least one segment's chord result never arrived (worker likely "
            f"lost mid-task with no replacement available). Job status was "
            f"{previous_status!r} when the watchdog fired."
        )
        stuck_segments = mark_stuck_segments_timeout(session, job, error)
        transition_job_status(session, job, JobStatus.failed)

        log.error(
            "watchdog.job_force_failed",
            previous_status=previous_status,
            stuck_segment_count=len(stuck_segments),
            stuck_segment_ids=[str(s.id) for s in stuck_segments],
        )
        return {
            "job_id": job_id,
            "action": "force_failed",
            "stuck_segment_count": len(stuck_segments),
        }


def _grace_period_seconds() -> float:
    from config import get_settings

    return get_settings().watchdog_grace_period_seconds


def schedule_watchdog(job_id: str, sla_target_seconds: float) -> None:
    """Called once, right after dispatching a job's chord (see
    orchestrator/pipeline.py). `time` import used only for documentation
    parity with other stage-timing modules — the actual delay is Celery's own
    countdown mechanism, not a blocking sleep."""
    countdown = sla_target_seconds + _grace_period_seconds()
    check_job_timeout.apply_async(args=[job_id], countdown=countdown)
    logger.bind(job_id=job_id, stage="watchdog").info(
        "watchdog.scheduled", countdown_seconds=countdown, fires_at_epoch=time.time() + countdown
    )
