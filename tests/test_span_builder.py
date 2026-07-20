"""Unit tests for workers.span_builder (Stage 3).

Pure-logic tests only — no Postgres/broker required. segment_worker_output
fixtures are plain JSON-safe dicts, matching what actually crosses the wire
between Celery tasks (see models/schemas.py module docstring).
"""

import uuid

from workers.span_builder import tasks as span_builder_tasks
from workers.span_builder.logic import build_candidate_spans


def _segment_id() -> str:
    return str(uuid.uuid4())


def _base_output(segment_id: str, **overrides) -> dict:
    output = {
        "segment_id": segment_id,
        "sequence_index": 0,
        "status": "completed",
        "shot_boundaries": [],
        "transcript": [],
        "visual_embeddings": {},
        "motion_features": {},
        "audio_features": {},
        "error": None,
    }
    output.update(overrides)
    return output


def test_normal_merge_produces_sane_spans():
    segment_id = _segment_id()
    output = _base_output(
        segment_id,
        shot_boundaries=[
            {"ts": 10.0, "keyframe_ref": "kf_10"},
            {"ts": 12.0, "keyframe_ref": "kf_12"},
            {"ts": 30.0, "keyframe_ref": "kf_30"},
        ],
        transcript=[
            {"start_ts": 11.0, "end_ts": 13.5, "text": "hello there", "speaker_label": "A"},
        ],
        visual_embeddings={"kf_10": [1.0, 0.0], "kf_12": [0.0, 1.0]},
        motion_features={"avg_motion": 0.5},
        audio_features={"avg_volume": 0.8},
    )

    spans = build_candidate_spans(output)

    # Shot windows [10,12] and [12,30] both overlap the transcript window
    # [11,13.5], so everything sweep-merges into one contiguous span.
    assert len(spans) == 1
    span = spans[0]
    assert span["segment_id"] == segment_id
    assert span["start_ts"] == 10.0
    assert span["end_ts"] == 30.0
    assert span["transcript_excerpt"] == "hello there"
    assert span["feature_vector"]["avg_motion"] == 0.5
    assert span["feature_vector"]["avg_volume"] == 0.8
    assert span["feature_vector"]["shot_count"] == 2
    assert span["feature_vector"]["visual_embedding_mean"] == [0.5, 0.5]
    assert span["touches_boundary"] is False


def test_no_shots_and_empty_transcript_produces_empty_span_list():
    output = _base_output(_segment_id())

    assert build_candidate_spans(output) == []


def test_single_orphan_shot_boundary_produces_empty_span_list():
    # A single boundary has no partner to form a shot window with, and there
    # is no transcript either — must not crash, must yield nothing.
    output = _base_output(
        _segment_id(),
        shot_boundaries=[{"ts": 5.0, "keyframe_ref": "kf_5"}],
    )

    assert build_candidate_spans(output) == []


def test_span_touching_overlap_window_is_flagged():
    segment_id = _segment_id()
    output = _base_output(
        segment_id,
        shot_boundaries=[
            {"ts": 100.0, "keyframe_ref": "kf_a"},
            {"ts": 104.0, "keyframe_ref": "kf_b"},
        ],
    )

    # The span is exactly [100, 104]; the overlap window starts exactly at
    # its end_ts (104) — an inclusive edge case that must still count as
    # "touches the boundary".
    touching = build_candidate_spans(output, overlap_start=104.0, overlap_end=110.0)
    assert len(touching) == 1
    assert touching[0]["end_ts"] == 104.0
    assert touching[0]["touches_boundary"] is True

    # A window nowhere near the span must not be flagged.
    not_touching = build_candidate_spans(output, overlap_start=200.0, overlap_end=210.0)
    assert not_touching[0]["touches_boundary"] is False


def test_overlap_window_fallback_from_payload_fields():
    # If overlap_start/overlap_end are embedded directly in
    # segment_worker_output (rather than passed as explicit args), they are
    # still honored.
    segment_id = _segment_id()
    output = _base_output(
        segment_id,
        shot_boundaries=[
            {"ts": 0.0, "keyframe_ref": "kf_a"},
            {"ts": 2.0, "keyframe_ref": "kf_b"},
        ],
        overlap_start=0.0,
        overlap_end=1.0,
    )

    spans = build_candidate_spans(output)
    assert spans[0]["touches_boundary"] is True


def test_task_returns_failed_status_when_logic_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(span_builder_tasks, "build_candidate_spans", _boom)

    payload = _base_output(_segment_id(), sequence_index=3)

    result = span_builder_tasks.build_spans(payload)

    assert result["status"] == "failed"
    assert result["candidate_spans"] == []
    assert result["error"] is not None
    assert "boom" in result["error"]
    assert result["segment_id"] == payload["segment_id"]
    assert result["sequence_index"] == 3


def test_task_happy_path_returns_completed_status():
    segment_id = _segment_id()
    payload = _base_output(
        segment_id,
        sequence_index=7,
        transcript=[{"start_ts": 1.0, "end_ts": 2.0, "text": "hi", "speaker_label": None}],
    )

    result = span_builder_tasks.build_spans(payload)

    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["segment_id"] == segment_id
    assert result["sequence_index"] == 7
    assert len(result["candidate_spans"]) == 1


def test_task_propagates_upstream_failed_status_without_building_spans():
    payload = _base_output(_segment_id(), status="failed", error="segment_worker exploded")

    result = span_builder_tasks.build_spans(payload)

    assert result["status"] == "failed"
    assert result["candidate_spans"] == []
    assert result["error"] == "segment_worker exploded"
