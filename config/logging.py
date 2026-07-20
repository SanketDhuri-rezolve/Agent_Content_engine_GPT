"""Centralized structlog configuration.

Every module across this codebase already does `logger = structlog.get_logger()`
at its own call sites (stage.started/stage.completed/stage.failed, with
job_id/segment_id/stage/elapsed_seconds fields) — none of those call sites
change. This module only controls the RENDER step: without it, structlog
falls back to its default human-readable ConsoleRenderer (what you see in
`docker compose logs`, e.g. "2026-07-11 15:55:29 [info ] stage.completed
elapsed_seconds=... job_id=..."). Configuring JSONRenderer instead makes every
log line a single parseable JSON object — this is what lets
scripts/measure_latency.py reliably reconstruct per-stage timing from
`docker compose logs` output via `json.loads(line)` rather than a fragile
key=value regex.

Call `configure_logging()` once per process. Already wired into the two
process entrypoints: orchestrator/celery_app.py (covers every Celery worker,
since all of them import celery_app) and api/main.py.
"""

import logging
import sys

import structlog

from config import get_settings

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
