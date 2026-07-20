"""Stage 6 (ranker) Celery task.

Queue: "ranker". Task name: "workers.ranker.tasks.rank_and_persist" (fixed —
do not rename, see orchestrator wiring contract).

Input is exactly workers.reducer.tasks.reduce_job's return shape
(models.schemas.ReducerOutput as a dict). This task:
  1. Runs MMR re-rank + top-k selection (workers.ranker.logic.mmr_rerank).
  2. Persists ScoredSpan rows for the ranked spans (creating the backing
     CandidateSpan row too, if one doesn't already exist for that
     segment_id/start_ts/end_ts — see _get_or_create_* below).
  3. Persists HighlightResult rows (rank, span_id, final_score).
  4. Marks the Job completed.
  5. Returns {"job_id": ..., "results": [RankedHighlight dicts]}.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import delete, select

from models.db import session_scope
from models.enums import JobStatus
from models.orm import CandidateSpan, HighlightResult, Job, ScoredSpan
from models.schemas import RankedHighlight, ReducedSpan
from orchestrator.celery_app import celery_app
from workers.ranker.logic import mmr_rerank

logger = structlog.get_logger()

# top_k / lambda_param are pipeline-tuning decisions, not per-environment
# config, so (unlike e.g. database_url) they don't belong in config.Settings.
# Not finalized by product yet — same "unconfirmed placeholder" status as
# config.Settings.provisional_dev_segment_count. Revisit together.
DEFAULT_TOP_K = 10
DEFAULT_LAMBDA_PARAM = 0.7

try:
    from orchestrator.state import transition_job_status
except ImportError:  # orchestrator.state not built yet (parallel batch) — see CLAUDE.md contract note.
    transition_job_status = None  # type: ignore[assignment]


@celery_app.task(name="workers.ranker.tasks.rank_and_persist")
def rank_and_persist(reducer_output: dict[str, Any], job_id: str) -> dict[str, Any]:
    log = logger.bind(job_id=job_id, stage="ranker")
    log.info("stage.started")
    start = time.monotonic()

    spans = reducer_output.get("spans", [])

    ranked = mmr_rerank(spans, top_k=DEFAULT_TOP_K, lambda_param=DEFAULT_LAMBDA_PARAM)

    results: list[dict[str, Any]] = []
    job_uuid = uuid.UUID(str(job_id))

    with session_scope() as session:
        # Idempotency on retry: this is the terminal stage for a job, so wipe
        # any highlight rows from a previous (failed/retried) attempt before
        # writing this run's ranking.
        session.execute(delete(HighlightResult).where(HighlightResult.job_id == job_uuid))

        for rank_idx, span in enumerate(ranked, start=1):
            scored_span_row = _get_or_create_scored_span(session, span)

            highlight_row = HighlightResult(
                job_id=job_uuid,
                rank=rank_idx,
                span_id=scored_span_row.id,
                final_score=span["final_score"],
            )
            session.add(highlight_row)
            session.flush()

            reduced_span = ReducedSpan(**{k: v for k, v in span.items() if k != "final_score"})
            ranked_highlight = RankedHighlight(
                rank=rank_idx,
                span=reduced_span,
                final_score=span["final_score"],
            )
            results.append(ranked_highlight.model_dump(mode="json"))

        job_row = session.get(Job, job_uuid)
        if job_row is not None:
            if transition_job_status is not None:
                transition_job_status(session, job_row, JobStatus.completed)
            else:
                # TODO: switch to orchestrator.state.transition_job_status once
                # that module lands (see contract note above). Inline update
                # kept deliberately minimal in the meantime.
                job_row.status = JobStatus.completed
                job_row.completed_at = datetime.now(timezone.utc)
        else:
            log.warning("job.not_found", job_id=job_id)

    elapsed = time.monotonic() - start
    log.info("stage.completed", elapsed_seconds=elapsed, ranked_count=len(results))

    return {"job_id": job_id, "results": results}


def _get_or_create_scored_span(session, span: dict[str, Any]) -> ScoredSpan:
    """Insert-if-not-present, keyed by segment_id + start_ts + end_ts."""
    segment_id = uuid.UUID(str(span["segment_id"]))

    existing = session.execute(
        select(ScoredSpan).where(
            ScoredSpan.segment_id == segment_id,
            ScoredSpan.start_ts == span["start_ts"],
            ScoredSpan.end_ts == span["end_ts"],
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    candidate_span_row = _get_or_create_candidate_span(session, span)

    row = ScoredSpan(
        candidate_span_id=candidate_span_row.id,
        segment_id=segment_id,
        start_ts=span["start_ts"],
        end_ts=span["end_ts"],
        transcript_excerpt=span.get("transcript_excerpt"),
        feature_vector=span.get("feature_vector") or {},
        touches_boundary=span.get("touches_boundary", False),
        raw_score=span["raw_score"],
        normalized_score=span.get("normalized_score"),
        justification=span.get("justification"),
        llm_model_version=span["llm_model_version"],
        clip_url=span.get("clip_url"),
        rich_data=span.get("rich_data"),
    )
    session.add(row)
    session.flush()
    return row


def _get_or_create_candidate_span(session, span: dict[str, Any]) -> CandidateSpan:
    """Insert-if-not-present, keyed by segment_id + start_ts + end_ts.

    ScoredSpan.candidate_span_id is a NOT NULL unique FK into candidate_spans
    (models/orm.py), so a backing CandidateSpan row must exist. Earlier
    stages (span_builder/scorer) may or may not have already persisted one
    by the time ranker runs — this covers both cases.
    """
    segment_id = uuid.UUID(str(span["segment_id"]))

    existing = session.execute(
        select(CandidateSpan).where(
            CandidateSpan.segment_id == segment_id,
            CandidateSpan.start_ts == span["start_ts"],
            CandidateSpan.end_ts == span["end_ts"],
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    row = CandidateSpan(
        segment_id=segment_id,
        start_ts=span["start_ts"],
        end_ts=span["end_ts"],
        transcript_excerpt=span.get("transcript_excerpt"),
        feature_vector=span.get("feature_vector") or {},
        touches_boundary=span.get("touches_boundary", False),
    )
    session.add(row)
    session.flush()
    return row
