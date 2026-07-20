"""Unit tests for workers.reducer.logic — pure logic, no DB/Celery/network.

Fixtures are plain dicts shaped like models.schemas.SegmentPipelineResult /
models.schemas.ScoredSpanPayload, matching what actually crosses the wire
between Celery tasks (see models/schemas.py module docstring).

`reduce_segment_results` reads config.get_settings().reducer_min_segments_
required_fraction, which defaults to 0.6 (config/settings.py) and has no
.env override in this repo — tests that exercise the full pipeline pick
fractions clearly on one side or the other of that documented default rather
than monkeypatching it, so they double as an integration check against the
real default. `classify_segments` takes the fraction as an explicit
parameter, so tests targeting it directly need no settings dependency at
all.
"""

import uuid

import pytest

from models.enums import SegmentStatus
from workers.reducer.logic import (
    DEDUP_IOU_THRESHOLD,
    InsufficientSegmentsError,
    classify_segments,
    dedupe_boundary_spans,
    normalize_scores,
    reduce_segment_results,
)


def _scored_span(
    segment_id,
    *,
    start_ts: float,
    end_ts: float,
    raw_score: float = 0.5,
    touches_boundary: bool = False,
    excerpt: str = "a line of dialogue",
) -> dict:
    return {
        "segment_id": str(segment_id),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_excerpt": excerpt,
        "feature_vector": {},
        "touches_boundary": touches_boundary,
        "raw_score": raw_score,
        "justification": "test fixture",
        "llm_model_version": "mock-scorer-v0",
    }


def _segment_result(
    segment_id,
    sequence_index: int,
    *,
    status: str = SegmentStatus.completed.value,
    scored_spans: list[dict] | None = None,
    error: str | None = None,
) -> dict:
    return {
        "segment_id": str(segment_id),
        "sequence_index": sequence_index,
        "status": status,
        "scored_spans": scored_spans or [],
        "error": error,
    }


# ---------------------------------------------------------------------------
# (a) Happy path — no duplicates
# ---------------------------------------------------------------------------


def test_happy_path_no_duplicates_end_to_end():
    seg_ids = [uuid.uuid4() for _ in range(4)]
    results = [
        _segment_result(
            seg_ids[i],
            i,
            scored_spans=[_scored_span(seg_ids[i], start_ts=float(i * 100), end_ts=float(i * 100 + 10), raw_score=float(i))],
        )
        for i in range(4)
    ]

    output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

    assert len(output["spans"]) == 4
    assert output["dropped_duplicate_count"] == 0
    assert output["degraded_segment_ids"] == []
    # 4 distinct raw_scores (0,1,2,3) z-normalized: mean ~0, std ~1.
    normalized = [span["normalized_score"] for span in output["spans"]]
    assert sum(normalized) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# (b) One segment timed out with empty scored_spans — proceed, record degraded
# ---------------------------------------------------------------------------


def test_timed_out_segment_is_degraded_not_fatal():
    seg_ids = [uuid.uuid4() for _ in range(4)]
    results = [
        _segment_result(seg_ids[0], 0, scored_spans=[_scored_span(seg_ids[0], start_ts=0.0, end_ts=10.0)]),
        _segment_result(seg_ids[1], 1, scored_spans=[_scored_span(seg_ids[1], start_ts=100.0, end_ts=110.0)]),
        _segment_result(seg_ids[2], 2, status=SegmentStatus.timeout.value, scored_spans=[]),
        _segment_result(seg_ids[3], 3, scored_spans=[_scored_span(seg_ids[3], start_ts=300.0, end_ts=310.0)]),
    ]

    output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

    assert len(output["spans"]) == 3
    assert output["degraded_segment_ids"] == [str(seg_ids[2])]


# ---------------------------------------------------------------------------
# (c) status=completed with partial data is usable
# ---------------------------------------------------------------------------


def test_completed_segment_with_partial_spans_is_usable():
    seg_ids = [uuid.uuid4() for _ in range(4)]
    # seg_ids[2] is "completed" but only produced one span instead of several
    # — still fully usable, not degraded.
    results = [
        _segment_result(seg_ids[0], 0, scored_spans=[_scored_span(seg_ids[0], start_ts=0.0, end_ts=10.0)]),
        _segment_result(seg_ids[1], 1, scored_spans=[_scored_span(seg_ids[1], start_ts=100.0, end_ts=110.0)]),
        _segment_result(seg_ids[2], 2, scored_spans=[_scored_span(seg_ids[2], start_ts=200.0, end_ts=210.0)]),
        _segment_result(seg_ids[3], 3, scored_spans=[_scored_span(seg_ids[3], start_ts=300.0, end_ts=310.0)]),
    ]

    usable, degraded_ids, degraded_labels = classify_segments(results, min_required_fraction=0.6)

    assert len(usable) == 4
    assert degraded_ids == []
    assert degraded_labels == []


# ---------------------------------------------------------------------------
# (d) Too many failed segments trips InsufficientSegmentsError
# ---------------------------------------------------------------------------


def test_insufficient_segments_raises_with_named_segments():
    seg_ids = [uuid.uuid4() for _ in range(4)]
    results = [
        _segment_result(seg_ids[0], 0, scored_spans=[_scored_span(seg_ids[0], start_ts=0.0, end_ts=10.0)]),
        _segment_result(seg_ids[1], 1, status=SegmentStatus.failed.value, scored_spans=[], error="boom"),
        _segment_result(seg_ids[2], 2, status=SegmentStatus.timeout.value, scored_spans=[]),
        _segment_result(seg_ids[3], 3, status=SegmentStatus.failed.value, scored_spans=[], error="boom2"),
    ]

    with pytest.raises(InsufficientSegmentsError) as exc_info:
        classify_segments(results, min_required_fraction=0.6)

    message = str(exc_info.value)
    assert "sequence_index=1" in message
    assert "sequence_index=2" in message
    assert "sequence_index=3" in message


def test_insufficient_segments_raises_end_to_end_via_reduce_segment_results():
    # 1/4 usable = 25%, well below the documented default fraction (0.6).
    seg_ids = [uuid.uuid4() for _ in range(4)]
    results = [
        _segment_result(seg_ids[0], 0, scored_spans=[_scored_span(seg_ids[0], start_ts=0.0, end_ts=10.0)]),
        _segment_result(seg_ids[1], 1, status=SegmentStatus.failed.value, scored_spans=[]),
        _segment_result(seg_ids[2], 2, status=SegmentStatus.failed.value, scored_spans=[]),
        _segment_result(seg_ids[3], 3, status=SegmentStatus.timeout.value, scored_spans=[]),
    ]

    with pytest.raises(InsufficientSegmentsError):
        reduce_segment_results(results, job_id=str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# (e) Boundary-touching duplicate spans from adjacent segments collapse
# ---------------------------------------------------------------------------


def test_adjacent_boundary_spans_heavily_overlapping_are_deduped():
    seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
    span_a = _scored_span(seg_a, start_ts=95.0, end_ts=105.0, raw_score=0.9, touches_boundary=True, excerpt="hello")
    span_b = _scored_span(seg_b, start_ts=96.0, end_ts=104.0, raw_score=0.4, touches_boundary=True, excerpt="world")

    segment_id_to_sequence_index = {str(seg_a): 0, str(seg_b): 1}
    surviving, dropped_count = dedupe_boundary_spans([span_a, span_b], segment_id_to_sequence_index)

    assert dropped_count == 1
    assert len(surviving) == 1
    # Higher raw_score (span_a) wins; transcript_excerpt merges both.
    assert surviving[0]["raw_score"] == 0.9
    assert "hello" in surviving[0]["transcript_excerpt"]
    assert "world" in surviving[0]["transcript_excerpt"]


def test_adjacent_boundary_spans_deduped_end_to_end():
    seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
    results = [
        _segment_result(
            seg_a, 0,
            scored_spans=[_scored_span(seg_a, start_ts=95.0, end_ts=105.0, raw_score=0.9, touches_boundary=True)],
        ),
        _segment_result(
            seg_b, 1,
            scored_spans=[_scored_span(seg_b, start_ts=96.0, end_ts=104.0, raw_score=0.4, touches_boundary=True)],
        ),
    ]

    output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

    assert len(output["spans"]) == 1
    assert output["dropped_duplicate_count"] == 1


def test_non_boundary_spans_are_never_deduped_even_if_identical_windows():
    seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
    span_a = _scored_span(seg_a, start_ts=10.0, end_ts=20.0, touches_boundary=False)
    span_b = _scored_span(seg_b, start_ts=10.0, end_ts=20.0, touches_boundary=False)

    surviving, dropped_count = dedupe_boundary_spans(
        [span_a, span_b], {str(seg_a): 0, str(seg_b): 1}
    )

    assert dropped_count == 0
    assert len(surviving) == 2


def test_non_adjacent_boundary_spans_are_not_merged():
    """Segments 2 apart never share an overlap window, so even a perfect
    timestamp match must not be collapsed."""
    seg_a, seg_c = uuid.uuid4(), uuid.uuid4()
    span_a = _scored_span(seg_a, start_ts=95.0, end_ts=105.0, touches_boundary=True)
    span_c = _scored_span(seg_c, start_ts=95.0, end_ts=105.0, touches_boundary=True)

    surviving, dropped_count = dedupe_boundary_spans(
        [span_a, span_c], {str(seg_a): 0, str(seg_c): 2}
    )

    assert dropped_count == 0
    assert len(surviving) == 2


def test_dedup_threshold_boundary_exactly_50_percent_iou_is_not_merged():
    """DEDUP_IOU_THRESHOLD is a strict `>` comparison at 0.5 — an exact 50%
    IoU pair must survive as two spans, documenting the exact cutoff."""
    assert DEDUP_IOU_THRESHOLD == 0.5
    seg_a, seg_b = uuid.uuid4(), uuid.uuid4()
    # [0, 10] and [5, 15]: intersection = 5, union = 15 -> IoU exactly 1/3,
    # below threshold. Use windows engineered for exactly 0.5 IoU instead:
    # [0, 20] and [10, 30]: intersection = 10, union = 30 -> IoU = 1/3 too.
    # Exact 0.5 IoU: [0, 20] and [0, 10] -> intersection=10, union=20 -> 0.5.
    span_a = _scored_span(seg_a, start_ts=0.0, end_ts=20.0, touches_boundary=True)
    span_b = _scored_span(seg_b, start_ts=0.0, end_ts=10.0, touches_boundary=True)

    surviving, dropped_count = dedupe_boundary_spans(
        [span_a, span_b], {str(seg_a): 0, str(seg_b): 1}
    )

    assert dropped_count == 0
    assert len(surviving) == 2


# ---------------------------------------------------------------------------
# (f) All-identical raw_score does not raise ZeroDivisionError
# ---------------------------------------------------------------------------


def test_normalize_scores_all_identical_raw_scores_no_division_error():
    spans = [
        _scored_span(uuid.uuid4(), start_ts=float(i), end_ts=float(i) + 1, raw_score=0.5)
        for i in range(3)
    ]

    normalized = normalize_scores(spans)

    assert all(span["normalized_score"] == 0.0 for span in normalized)


def test_normalize_scores_single_span_no_division_error():
    spans = [_scored_span(uuid.uuid4(), start_ts=0.0, end_ts=1.0, raw_score=0.75)]

    normalized = normalize_scores(spans)

    assert normalized[0]["normalized_score"] == 0.0


def test_normalize_scores_empty_list_returns_empty():
    assert normalize_scores([]) == []


# ---------------------------------------------------------------------------
# (g) A bare None chord entry (hard task failure) is handled, not crashed
# ---------------------------------------------------------------------------


def test_bare_none_entry_is_treated_as_degraded_segment():
    seg_ids = [uuid.uuid4() for _ in range(3)]
    results: list = [
        _segment_result(seg_ids[0], 0, scored_spans=[_scored_span(seg_ids[0], start_ts=0.0, end_ts=10.0)]),
        None,  # hard task failure — a chain member raised instead of returning
        _segment_result(seg_ids[1], 2, scored_spans=[_scored_span(seg_ids[1], start_ts=200.0, end_ts=210.0)]),
    ]

    output = reduce_segment_results(results, job_id=str(uuid.uuid4()))

    assert len(output["spans"]) == 2
    # The None entry has no recoverable segment_id, so it does not appear in
    # degraded_segment_ids (which is typed as a list of UUIDs) — but it still
    # counted toward the usable/total fraction without raising.
    assert output["degraded_segment_ids"] == []


def test_bare_none_entry_labeled_by_position_when_threshold_still_met():
    # A single None among enough completed entries to stay above the 60%
    # threshold (3/4 = 75%) — isolates the label-formatting behavior for a
    # bare None entry without tripping InsufficientSegmentsError, which is
    # covered separately by test_all_none_entries_raise_insufficient_segments
    # and test_empty_results_list_raises_insufficient_segments below.
    completed = {"segment_id": "s", "sequence_index": 0, "status": "completed", "scored_spans": []}
    usable, degraded_ids, degraded_labels = classify_segments(
        [completed, completed, completed, None], min_required_fraction=0.6
    )

    assert usable == [completed, completed, completed]
    assert degraded_ids == []
    assert len(degraded_labels) == 1
    assert "position=3" in degraded_labels[0]


def test_all_none_entries_raise_insufficient_segments():
    with pytest.raises(InsufficientSegmentsError):
        classify_segments([None, None, None], min_required_fraction=0.6)


def test_empty_results_list_raises_insufficient_segments():
    with pytest.raises(InsufficientSegmentsError):
        classify_segments([], min_required_fraction=0.6)
