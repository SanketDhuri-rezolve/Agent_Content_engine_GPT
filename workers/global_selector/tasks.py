"""Stage 3.5 Celery task (opt-in, config.Settings.use_global_memory_pipeline
only): select_and_score_job.

Replaces reduce_job as the chord callback when the global-memory pipeline is
enabled (see orchestrator/pipeline.py's branch). Unlike the default
pipeline's per-segment workers.scorer.tasks.score_spans (which scores EVERY
candidate span independently, chunk-blind), this task:

  1. Merges every segment's local_memory (produced by segment_worker's
     MemoryExtractor — see workers.segment_worker.adapters.memory_extractor)
     into one global movie memory.
  2. Flattens every usable segment's candidate_spans into one list and asks
     workers.global_selector.adapters.get_global_selector() (text-only,
     cheap, one call for the whole job) to pick the top
     config.Settings.global_selector_top_n candidates using the global
     memory for full-story context.
  3. Scores ONLY the selected winners with the real multimodal
     workers.scorer.adapters.get_scorer() — concurrently, via
     workers.scorer.tasks.score_candidates_concurrently, the exact same
     rich-schema scoring path the default pipeline uses per span.
  4. Reassembles per-segment SegmentPipelineResult-shaped dicts (a segment
     with no selected winners simply has an empty scored_spans list) so
     workers.reducer.tasks.reduce_job and workers.ranker.tasks.rank_and_persist
     consume this task's output identically to the default pipeline's
     score_spans output — this is a drop-in chord-callback replacement, not
     a change to the fixed reducer/ranker contract.

Fail-soft: a segment already marked failed/timeout by span_builder keeps
that status/error, contributes no candidates to selection, and passes
through unchanged — same policy as workers.scorer.tasks.score_spans. If the
global selection call itself fails, this task does not fail the job — it
logs a warning and proceeds with zero selected candidates (every segment
ships with empty scored_spans, same as span_builder legitimately finding
nothing), since a selection failure isn't a per-segment problem and
re-raising here would take the whole job down over one LLM call.
"""

import time

import structlog

from config import get_settings
from models.enums import SegmentStatus
from models.schemas import ScoredSpanPayload, SegmentPipelineResult
from orchestrator.celery_app import celery_app
from workers.global_selector.adapters import get_global_selector
from workers.scorer.adapters import get_scorer
from workers.scorer.tasks import score_candidates_concurrently

logger = structlog.get_logger()

# Mirrors workers.span_builder.tasks._FAILED_UPSTREAM_STATUSES exactly.
_FAILED_UPSTREAM_STATUSES = {SegmentStatus.failed.value, SegmentStatus.timeout.value}


@celery_app.task(name="workers.global_selector.tasks.select_and_score_job", queue="global_selector")
def select_and_score_job(span_builder_outputs: list[dict], job_id: str) -> list[dict]:
    start_time = time.monotonic()
    logger.info(
        "stage.started",
        job_id=job_id,
        stage="global_selector",
        segment_count=len(span_builder_outputs),
    )

    settings = get_settings()

    segment_meta: dict[str, dict] = {}
    all_candidates: list[dict] = []
    candidate_owner: list[str] = []  # parallel to all_candidates: owning segment_id

    for output in span_builder_outputs:
        segment_id = output.get("segment_id")
        segment_meta[segment_id] = {
            "sequence_index": output.get("sequence_index"),
            "status": output.get("status"),
            "error": output.get("error"),
        }
        if output.get("status") in _FAILED_UPSTREAM_STATUSES:
            continue
        for raw_span in output.get("candidate_spans") or []:
            all_candidates.append(raw_span)
            candidate_owner.append(segment_id)

    global_memory = {
        "chunks": [output.get("local_memory") for output in span_builder_outputs if output.get("local_memory")]
    }

    selected_indices: list[int] = []
    selection_error: str | None = None
    if all_candidates:
        try:
            selector = get_global_selector()
            selected_indices = selector.select(global_memory, all_candidates, settings.global_selector_top_n)
        except Exception as exc:  # noqa: BLE001 - fail-soft: selection failure must not kill the job
            selection_error = str(exc)
            logger.warning(
                "stage.global_selection_failed", job_id=job_id, stage="global_selector", error=selection_error
            )

    selected_raw_spans = [all_candidates[i] for i in selected_indices]
    selected_owners = [candidate_owner[i] for i in selected_indices]

    scorer = get_scorer()
    score_results = score_candidates_concurrently(
        scorer, selected_raw_spans, settings.gemma4_max_concurrent_requests
    )

    scored_by_segment: dict[str, list[ScoredSpanPayload]] = {}
    failed_count = 0
    for (span_index, scored, error), owner_segment_id in zip(score_results, selected_owners):
        if scored is None:
            failed_count += 1
            logger.warning(
                "stage.span_score_failed",
                job_id=job_id,
                stage="global_selector",
                segment_id=owner_segment_id,
                span_index=span_index,
                error=error,
            )
            continue
        scored_by_segment.setdefault(owner_segment_id, []).append(scored)

    results: list[dict] = []
    for segment_id, meta in segment_meta.items():
        result = SegmentPipelineResult(
            segment_id=segment_id,
            sequence_index=meta["sequence_index"],
            status=meta["status"],
            scored_spans=scored_by_segment.get(segment_id, []),
            error=meta["error"],
        )
        results.append(result.model_dump(mode="json"))

    elapsed = time.monotonic() - start_time
    logger.info(
        "stage.completed",
        job_id=job_id,
        stage="global_selector",
        elapsed_seconds=elapsed,
        candidate_count=len(all_candidates),
        selected_count=len(selected_indices),
        scored_count=sum(len(v) for v in scored_by_segment.values()),
        failed_count=failed_count,
        selection_error=selection_error,
    )
    return results
