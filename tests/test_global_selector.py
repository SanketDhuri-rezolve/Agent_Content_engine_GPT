"""Unit tests for workers.global_selector — the opt-in (config.Settings.
use_global_memory_pipeline) chord-callback replacement that selects top_n
moments job-wide before scoring only those, instead of scoring every
candidate span independently. Pure logic + Mock adapters, no DB/network."""

import uuid

from models.enums import SegmentStatus
from workers.global_selector.adapters import MockGlobalSelector
from workers.global_selector.tasks import select_and_score_job


def _span_builder_output(segment_id, sequence_index, candidate_spans, local_memory=None, status="completed"):
    return {
        "segment_id": str(segment_id),
        "sequence_index": sequence_index,
        "status": status,
        "candidate_spans": candidate_spans,
        "local_memory": local_memory,
        "error": None,
    }


def _candidate(segment_id, start_ts, end_ts, excerpt):
    return {
        "segment_id": str(segment_id),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "transcript_excerpt": excerpt,
        "feature_vector": {},
        "touches_boundary": False,
        "source_video_url": None,
    }


def test_mock_global_selector_picks_top_n_by_excerpt_length():
    selector = MockGlobalSelector()
    candidates = [
        _candidate(uuid.uuid4(), 0.0, 5.0, "short"),
        _candidate(uuid.uuid4(), 5.0, 10.0, "a much longer and more substantial line of dialogue here"),
        _candidate(uuid.uuid4(), 10.0, 15.0, "medium length excerpt"),
    ]

    selected = selector.select(global_memory={}, candidates=candidates, top_n=2)

    assert selected == [1, 2]  # longest excerpt first, then next-longest


def test_select_and_score_job_only_scores_selected_winners():
    segment_a = uuid.uuid4()
    segment_b = uuid.uuid4()

    span_builder_outputs = [
        _span_builder_output(
            segment_a,
            0,
            [
                _candidate(segment_a, 0.0, 5.0, "a"),
                _candidate(segment_a, 5.0, 10.0, "a much longer and more substantial line of dialogue"),
            ],
            local_memory={"segment_id": str(segment_a), "events": ["thing happened"]},
        ),
        _span_builder_output(
            segment_b,
            1,
            [_candidate(segment_b, 20.0, 25.0, "bb")],
            local_memory={"segment_id": str(segment_b), "events": ["another thing"]},
        ),
    ]

    result = select_and_score_job(span_builder_outputs, job_id=str(uuid.uuid4()))

    assert len(result) == 2
    total_scored = sum(len(r["scored_spans"]) for r in result)
    # top_n defaults to config.Settings.global_selector_top_n (10) — every
    # candidate across both segments fits under that, so all 3 get scored.
    assert total_scored == 3
    for r in result:
        assert r["status"] == SegmentStatus.completed.value
        for scored_span in r["scored_spans"]:
            assert scored_span["rich_data"]["moment_id"]


def test_select_and_score_job_respects_top_n_and_skips_unselected_segments():
    segment_a = uuid.uuid4()
    segment_b = uuid.uuid4()

    span_builder_outputs = [
        _span_builder_output(
            segment_a,
            0,
            [_candidate(segment_a, 0.0, 5.0, "the longest and most substantial transcript excerpt by far")],
        ),
        _span_builder_output(segment_b, 1, [_candidate(segment_b, 20.0, 25.0, "short")]),
    ]

    from config import get_settings

    get_settings().global_selector_top_n = 1
    try:
        result = select_and_score_job(span_builder_outputs, job_id=str(uuid.uuid4()))
    finally:
        get_settings().global_selector_top_n = 10

    total_scored = sum(len(r["scored_spans"]) for r in result)
    assert total_scored == 1
    scored_segment_ids = {r["segment_id"] for r in result if r["scored_spans"]}
    assert scored_segment_ids == {str(segment_a)}  # the longer excerpt wins


def test_select_and_score_job_propagates_upstream_segment_failure():
    segment_a = uuid.uuid4()
    span_builder_outputs = [
        _span_builder_output(segment_a, 0, [], status=SegmentStatus.failed.value),
    ]
    span_builder_outputs[0]["error"] = "span_builder blew up"

    result = select_and_score_job(span_builder_outputs, job_id=str(uuid.uuid4()))

    assert len(result) == 1
    assert result[0]["status"] == SegmentStatus.failed.value
    assert result[0]["error"] == "span_builder blew up"
    assert result[0]["scored_spans"] == []


def test_select_and_score_job_with_no_candidates_returns_empty_scored_spans():
    segment_a = uuid.uuid4()
    span_builder_outputs = [_span_builder_output(segment_a, 0, [])]

    result = select_and_score_job(span_builder_outputs, job_id=str(uuid.uuid4()))

    assert result[0]["status"] == SegmentStatus.completed.value
    assert result[0]["scored_spans"] == []
    assert result[0]["error"] is None
