"""Stage 3 Celery task: workers.span_builder.tasks.build_spans (queue:
"span_builder").

Thin wrapper around workers.span_builder.logic.build_candidate_spans. Fails
soft: any exception raised while building spans is caught here and turned
into a status="failed" result with a populated `error` field rather than
being allowed to propagate — one bad segment must never raise up through a
chain/chord and take the rest of the job down with it (same fail-soft
pattern as workers.segment_worker).

Note on overlap_start/overlap_end: models.schemas.SegmentWorkerOutput (Stage
2's output / this task's primary input) does not carry these fields — they
live on SegmentTaskPayload (Stage 2's *input*). Per the Stage 3 contract,
build_spans accepts them as optional extra keyword arguments so the
orchestrator can bind them onto the chain signature at construction time
(e.g. ``build_spans.s(overlap_start=payload.overlap_start,
overlap_end=payload.overlap_end)``), while the primary positional contract —
build_spans(segment_worker_output: dict) -> dict — is unchanged.
"""

from __future__ import annotations

import time

import structlog

from orchestrator.celery_app import celery_app
from workers.span_builder.logic import build_candidate_spans

logger = structlog.get_logger()

# Statuses inherited from Stage 2 that mean "nothing valid was produced" —
# span_builder should not pretend it built spans from data that doesn't exist.
_FAILED_UPSTREAM_STATUSES = {"failed", "timeout"}


@celery_app.task(name="workers.span_builder.tasks.build_spans", queue="span_builder")
def build_spans(
    segment_worker_output: dict,
    overlap_start: float | None = None,
    overlap_end: float | None = None,
    source_video_url: str | None = None,
) -> dict:
    segment_id = segment_worker_output.get("segment_id")
    sequence_index = segment_worker_output.get("sequence_index")
    segment_id_str = str(segment_id) if segment_id is not None else None

    logger.info(
        "stage.started",
        segment_id=segment_id_str,
        sequence_index=sequence_index,
        stage="span_builder",
    )
    started_at = time.monotonic()

    upstream_status = segment_worker_output.get("status")
    upstream_error = segment_worker_output.get("error")

    try:
        if upstream_status in _FAILED_UPSTREAM_STATUSES:
            # Stage 2 already failed for this segment — there is nothing
            # valid to build spans from. Propagate the failure rather than
            # silently returning an empty-but-"completed" result.
            result = {
                "segment_id": segment_id_str,
                "sequence_index": sequence_index,
                "status": upstream_status,
                "candidate_spans": [],
                "local_memory": None,
                "error": upstream_error or f"upstream segment_worker reported status={upstream_status}",
            }
        else:
            candidate_spans = build_candidate_spans(
                segment_worker_output, overlap_start, overlap_end, source_video_url
            )
            result = {
                "segment_id": segment_id_str,
                "sequence_index": sequence_index,
                "status": "completed",
                "candidate_spans": candidate_spans,
                # Passed through unchanged from Stage 2 (see
                # models.schemas.SegmentWorkerOutput.local_memory) — only
                # non-None when config.Settings.use_global_memory_pipeline is
                # True. Consumed by workers.global_selector.tasks
                # .select_and_score_job, not by anything in this module.
                "local_memory": segment_worker_output.get("local_memory"),
                "error": None,
            }

        elapsed = time.monotonic() - started_at
        logger.info(
            "stage.completed",
            segment_id=segment_id_str,
            sequence_index=sequence_index,
            stage="span_builder",
            elapsed_seconds=elapsed,
            status=result["status"],
            span_count=len(result["candidate_spans"]),
        )
        return result
    except Exception as exc:  # noqa: BLE001 - fail-soft boundary, must never raise
        elapsed = time.monotonic() - started_at
        logger.error(
            "stage.failed",
            segment_id=segment_id_str,
            sequence_index=sequence_index,
            stage="span_builder",
            elapsed_seconds=elapsed,
            error=str(exc),
        )
        return {
            "segment_id": segment_id_str,
            "sequence_index": sequence_index,
            "status": "failed",
            "candidate_spans": [],
            "local_memory": None,
            "error": str(exc),
        }
