"""Stage 4: score_spans Celery task.

LLM-as-judge scoring of every candidate span produced by span_builder (Stage
3). Each span is scored independently via the active `ScorerAdapter` (see
`workers.scorer.adapters.get_scorer` — MockScorer for Step 1, Gemma4Scorer
wired in behind the same interface for Step 2), and the segment's
`SegmentPipelineResult` is reassembled for the Stage 5 (reducer) chord
callback.

Fail-soft policy (documented here since the task contract requires one):
  - If the upstream stage (span_builder) already marked this segment
    `failed` OR `timeout` (see workers.span_builder.tasks's own
    _FAILED_UPSTREAM_STATUSES — Step 4 added `timeout` there for a
    SoftTimeLimitExceeded caught in segment_worker; this task must recognize
    the same set, or a timed-out segment silently reads as "span_builder
    legitimately found zero candidates" i.e. `completed` with empty
    scored_spans — a real bug this module had until a live Step 4 test
    caught it), that status/error is propagated UNCHANGED (not collapsed to
    `failed` — a timeout must stay a timeout all the way to the reducer/DB)
    and no scoring is attempted — there is nothing scoreable, and re-raising
    here would turn one bad segment into a chord-wide failure, which the
    contract explicitly forbids.
  - An empty `candidate_spans` list is NOT an error. It produces an empty
    `scored_spans` list and a `completed` status — span_builder finding zero
    candidates in a segment is a legitimate outcome, not a scorer failure.
  - Each candidate span is scored independently inside its own try/except.
    A single span's scoring exception (including a malformed span shape) is
    caught, logged (`stage.span_score_failed`), and that span is dropped —
    it does NOT fail the whole segment. This keeps one flaky/slow LLM call
    from discarding every other span the segment successfully produced.
  - EXCEPTION: if `candidate_spans` is non-empty but every single span fails
    to score, the segment as a whole is degraded to `failed`. There is
    nothing useful left for the reducer to consume, and silently returning
    `scored_spans=[]` would look identical to "span_builder found nothing",
    which is a different and misleading signal downstream. This is logged as
    `stage.completed` with `status=failed`.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import structlog

from config import get_settings
from models.enums import SegmentStatus
from models.schemas import CandidateSpanPayload, ScoredSpanPayload, SegmentPipelineResult
from orchestrator.celery_app import celery_app
from workers.scorer.adapters import ScorerAdapter, get_scorer

logger = structlog.get_logger()

# Mirrors workers.span_builder.tasks._FAILED_UPSTREAM_STATUSES exactly — any
# status span_builder can propagate as "not usable" must be recognized here
# too, or it silently reads as span_builder having found zero candidates.
_FAILED_UPSTREAM_STATUSES = {SegmentStatus.failed.value, SegmentStatus.timeout.value}


def _score_one(
    scorer: ScorerAdapter, span_index: int, raw_span: dict
) -> tuple[int, ScoredSpanPayload | None, str | None]:
    """One span's worth of work for score_candidates_concurrently — runs in
    a worker thread. Returns (span_index, scored_or_None, error_or_None)
    rather than raising, so the caller never needs a try/except around
    future.result() for the expected "this span failed to score" case."""
    try:
        candidate = CandidateSpanPayload.model_validate(raw_span)
        raw_score, justification, llm_model_version, clip_url, rich_data = scorer.score(
            candidate.model_dump(mode="json")
        )
    except Exception as exc:  # noqa: BLE001 - deliberate: any per-span failure is fail-soft
        return span_index, None, str(exc)

    scored = ScoredSpanPayload(
        **candidate.model_dump(),
        raw_score=raw_score,
        justification=justification,
        llm_model_version=llm_model_version,
        clip_url=clip_url,
        rich_data=rich_data,
    )
    return span_index, scored, None


def score_candidates_concurrently(
    scorer: ScorerAdapter, candidate_spans_raw: list[dict], max_concurrent_requests: int
) -> list[tuple[int, ScoredSpanPayload | None, str | None]]:
    """Fires every span's scorer.score() call concurrently (bounded by
    max_concurrent_requests) instead of one at a time. Each Gemma4Scorer.score()
    call is a single blocking HTTP round-trip (crop clip, build multimodal
    payload, httpx.post) — a plain sequential for-loop over N spans means
    vLLM never sees more than one request in flight, so its continuous
    batching is never exercised. Measured on the real 13-min test clip:
    ~26s/span average with the old sequential loop.

    Results are returned in the SAME order as candidate_spans_raw (not
    completion order) so callers can build output deterministically without
    re-sorting — ThreadPoolExecutor still runs every submitted call
    concurrently regardless of the order .result() is collected in."""
    if not candidate_spans_raw:
        return []
    worker_count = max(1, min(len(candidate_spans_raw), max_concurrent_requests))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_score_one, scorer, span_index, raw_span)
            for span_index, raw_span in enumerate(candidate_spans_raw)
        ]
        return [future.result() for future in futures]


@celery_app.task(name="workers.scorer.tasks.score_spans")
def score_spans(span_builder_output: dict) -> dict:
    segment_id = span_builder_output["segment_id"]
    sequence_index = span_builder_output["sequence_index"]
    upstream_status = span_builder_output.get("status")
    upstream_error = span_builder_output.get("error")
    candidate_spans_raw = span_builder_output.get("candidate_spans") or []

    start_time = time.monotonic()
    logger.info(
        "stage.started",
        segment_id=segment_id,
        sequence_index=sequence_index,
        stage="scorer",
        candidate_span_count=len(candidate_spans_raw),
    )

    if upstream_status in _FAILED_UPSTREAM_STATUSES:
        elapsed = time.monotonic() - start_time
        logger.info(
            "stage.completed",
            segment_id=segment_id,
            sequence_index=sequence_index,
            stage="scorer",
            status=upstream_status,
            elapsed_seconds=elapsed,
            reason="upstream_failure",
        )
        result = SegmentPipelineResult(
            segment_id=segment_id,
            sequence_index=sequence_index,
            status=upstream_status,
            scored_spans=[],
            error=upstream_error or f"upstream stage (span_builder) reported status={upstream_status!r}",
        )
        return result.model_dump(mode="json")

    scorer = get_scorer()
    scored_spans: list[ScoredSpanPayload] = []
    failure_count = 0

    results = score_candidates_concurrently(
        scorer, candidate_spans_raw, get_settings().gemma4_max_concurrent_requests
    )
    for span_index, scored, error in results:
        if scored is None:
            failure_count += 1
            logger.warning(
                "stage.span_score_failed",
                segment_id=segment_id,
                sequence_index=sequence_index,
                stage="scorer",
                span_index=span_index,
                error=error,
            )
            continue
        scored_spans.append(scored)

    total_candidates = len(candidate_spans_raw)
    all_failed = total_candidates > 0 and failure_count == total_candidates
    status = SegmentStatus.failed if all_failed else SegmentStatus.completed
    error = (
        f"scorer: all {total_candidates} candidate span(s) failed to score"
        if all_failed
        else None
    )

    elapsed = time.monotonic() - start_time
    logger.info(
        "stage.completed",
        segment_id=segment_id,
        sequence_index=sequence_index,
        stage="scorer",
        status=status.value,
        elapsed_seconds=elapsed,
        scored_span_count=len(scored_spans),
        failed_span_count=failure_count,
    )

    result = SegmentPipelineResult(
        segment_id=segment_id,
        sequence_index=sequence_index,
        status=status,
        scored_spans=scored_spans,
        error=error,
    )
    return result.model_dump(mode="json")
