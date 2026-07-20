"""Celery application definition — the single source of truth for the
pipeline's task queues.

Broker/backend come from config.get_settings() so environment differences
(local/runpod/prod) are expressed only via .env, never via code branches.

Queue-per-stage layout: each pipeline stage gets its own named queue so it
can be scaled independently in production (e.g. more segment_worker replicas
than ranker replicas — see docker-compose.yml, which runs one Celery worker
process per queue). `task_routes` is the authoritative queue assignment;
individual task modules should not need to (and in this codebase mostly do
not) repeat `queue=...` on their own `@celery_app.task` decorator.

`include` is how a real `celery -A orchestrator.celery_app worker` process
discovers task modules at startup. It is intentionally NOT the same as
importing those modules here at the top of this file: Celery only imports
`include` entries lazily (via the worker bootstrap / loader), not at Celery()
construction time, so this file can safely name modules that don't exist yet
during Step 1's parallel build — by the time a real worker process starts,
all pieces are expected to exist together.
"""

from celery import Celery

from config import get_settings
from config.logging import configure_logging

configure_logging()

_settings = get_settings()

celery_app = Celery(
    "movie_highlight_pipeline",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=[
        "workers.segment_worker.tasks",
        "workers.span_builder.tasks",
        "workers.scorer.tasks",
        # Opt-in (config.Settings.use_global_memory_pipeline) Stage 3.5 —
        # replaces workers.scorer.tasks.score_spans as the chord callback
        # when enabled. See workers/global_selector/tasks.py.
        "workers.global_selector.tasks",
        "workers.reducer.tasks",
        "workers.ranker.tasks",
        # Not part of the 5-stage worker contract, but this is the module
        # (in this same orchestrator/ scope) that defines the run_job task —
        # included here too so a real worker consuming its queue can find it.
        # See orchestrator/pipeline.py's module docstring for the queue-name
        # deviation note (no docker-compose service consumes it yet).
        "orchestrator.pipeline",
        # Step 4: job-level watchdog (see orchestrator/watchdog.py's module
        # docstring) — same queue as run_job.
        "orchestrator.watchdog",
    ],
)

celery_app.conf.task_routes = {
    "workers.segment_worker.tasks.*": {"queue": "segment_worker"},
    "workers.span_builder.tasks.*": {"queue": "span_builder"},
    "workers.scorer.tasks.*": {"queue": "scorer"},
    "workers.global_selector.tasks.*": {"queue": "global_selector"},
    "workers.reducer.tasks.*": {"queue": "reducer"},
    "workers.ranker.tasks.*": {"queue": "ranker"},
    # Additive beyond the fixed 5-stage contract (see note above / final
    # report) — orchestrator.pipeline.run_job needs a queue too.
    "orchestrator.pipeline.*": {"queue": "orchestrator"},
    "orchestrator.watchdog.*": {"queue": "orchestrator"},
}

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    # Step 4: if a worker is killed (SIGKILL, OOM, node loss) while holding a
    # task, acks_late means that task's message was never acked and gets
    # requeued for another worker to pick up, instead of silently vanishing
    # forever (the default task_acks_late=False acks a message the moment a
    # worker receives it, before running it — a mid-task kill then loses it
    # permanently with no chance of redelivery). task_reject_on_worker_lost
    # makes this explicit: Celery detects the worker's death and rejects
    # (requeues) the message rather than leaving it in limbo. This does NOT
    # by itself prevent a hang if no other worker of that type ever becomes
    # available — that's what orchestrator.watchdog.check_job_timeout is for.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
