"""Unit tests for workers.segment_worker — pure logic + deterministic mock
adapters, no DB/Postgres/Redis/Celery-broker needed (unlike tests/test_api.py,
this module never uses the db_session fixture). Calling the Celery-decorated
`run_segment_worker` task directly (not via .delay()/.apply_async()) executes
its underlying function synchronously in-process, so no worker/broker is
required either.
"""

import uuid

import pytest
from celery.exceptions import SoftTimeLimitExceeded

from config import get_settings
from models.enums import SegmentStatus
from models.schemas import SegmentTaskPayload
from workers.segment_worker import tasks


def _make_payload(start_ts: float, end_ts: float, sequence_index: int = 0) -> dict:
    payload = SegmentTaskPayload(
        job_id=uuid.uuid4(),
        segment_id=uuid.uuid4(),
        sequence_index=sequence_index,
        source_video_url="file:///tmp/fake_movie.mp4",
        start_ts=start_ts,
        end_ts=end_ts,
        overlap_start=start_ts,
        overlap_end=end_ts,
    )
    return payload.model_dump(mode="json")


class _BrokenShotDetector:
    """Deliberately-broken adapter used to exercise the task's failure path."""

    def detect(self, *args, **kwargs):
        raise RuntimeError("boom - shot detector exploded")


class _BrokenTranscriber:
    """Deliberately-broken adapter used to exercise the task's failure path.
    Transcriber (unlike shot_detector/diarizer/etc.) always runs regardless
    of config.Settings.enable_secondary_adapters (Whisper+Gemma4-only mode),
    so failure-path tests inject a broken adapter here, not on a
    secondary-adapter, to stay valid regardless of that setting."""

    def transcribe(self, *args, **kwargs):
        raise RuntimeError("boom - transcriber exploded")


def test_normal_payload_produces_nonempty_shots_and_transcript(monkeypatch):
    # This test specifically asserts on shot_boundaries/visual_embeddings,
    # which are only populated when secondary adapters are enabled (see
    # config.Settings.enable_secondary_adapters — default False/Whisper+
    # Gemma4-only mode as of CLAUDE.md's latest architecture decision).
    monkeypatch.setattr(get_settings(), "enable_secondary_adapters", True)
    payload = _make_payload(start_ts=100.0, end_ts=130.0, sequence_index=2)

    result = tasks.run_segment_worker(payload)

    assert result["status"] == SegmentStatus.completed.value
    assert result["error"] is None
    assert result["segment_id"] == payload["segment_id"]
    assert result["sequence_index"] == 2

    assert len(result["shot_boundaries"]) > 0
    assert len(result["transcript"]) > 0

    # Timestamps stay GLOBAL (relative to the whole film), never renormalized
    # to segment-local 0 — the reducer's dedup logic depends on this.
    assert result["shot_boundaries"][0]["ts"] >= payload["start_ts"]
    assert result["transcript"][0]["start_ts"] >= payload["start_ts"]
    assert all(payload["start_ts"] <= line["end_ts"] <= payload["end_ts"] for line in result["transcript"])

    # Diarization got merged onto the transcript lines.
    assert any(line["speaker_label"] for line in result["transcript"])

    # VisualEmbedder only ever runs on the keyframes ShotDetector emitted.
    keyframe_refs = {s["keyframe_ref"] for s in result["shot_boundaries"] if s["keyframe_ref"]}
    assert keyframe_refs
    assert set(result["visual_embeddings"].keys()) == keyframe_refs

    assert result["motion_features"]
    assert result["audio_features"]


def test_local_memory_is_none_by_default():
    """config.Settings.use_global_memory_pipeline defaults to False — the
    default per-span pipeline must never populate local_memory, since
    nothing downstream (span_builder, scorer, reducer) is prepared to see
    it in that mode."""
    payload = _make_payload(start_ts=0.0, end_ts=30.0)

    result = tasks.run_segment_worker(payload)

    assert result["local_memory"] is None


def test_local_memory_is_populated_when_global_memory_pipeline_enabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "use_global_memory_pipeline", True)
    payload = _make_payload(start_ts=0.0, end_ts=30.0)

    result = tasks.run_segment_worker(payload)

    assert result["local_memory"] is not None
    assert result["local_memory"]["segment_id"] == payload["segment_id"]


def test_output_is_deterministic_across_runs():
    """Mock adapters must be reproducible, not Math.random-style flaky."""
    payload = _make_payload(start_ts=50.0, end_ts=90.0)

    first = tasks.run_segment_worker(payload)
    second = tasks.run_segment_worker(payload)

    assert first == second


@pytest.mark.parametrize("start_ts,end_ts", [(42.0, 42.0), (100.0, 99.0)])
def test_zero_or_inverted_duration_segment_handled_gracefully(start_ts, end_ts):
    """start_ts == end_ts (or an inverted range) must not divide-by-zero or
    otherwise crash a naive implementation — it should degrade to an empty
    but well-formed, non-failed result."""
    payload = _make_payload(start_ts=start_ts, end_ts=end_ts)

    result = tasks.run_segment_worker(payload)

    assert result["status"] == SegmentStatus.completed.value
    assert result["error"] is None
    assert result["shot_boundaries"] == []
    assert result["transcript"] == []
    assert result["visual_embeddings"] == {}


def test_injected_adapter_exception_sets_failed_status_not_raise():
    payload = _make_payload(start_ts=0.0, end_ts=30.0, sequence_index=3)
    broken_adapters = tasks.SegmentWorkerAdapters(transcriber=_BrokenTranscriber())

    result = tasks.run_segment_worker(payload, adapters=broken_adapters)

    assert result["status"] == SegmentStatus.failed.value
    assert result["segment_id"] == payload["segment_id"]
    assert result["sequence_index"] == 3
    assert result["error"]
    assert "boom" in result["error"]
    assert result["shot_boundaries"] == []


def test_malformed_payload_missing_required_field_sets_failed_status_not_raise():
    payload = _make_payload(start_ts=0.0, end_ts=10.0)
    del payload["source_video_url"]

    result = tasks.run_segment_worker(payload)

    assert result["status"] == SegmentStatus.failed.value
    assert result["error"]
    assert result["segment_id"] == payload["segment_id"]


def test_unparseable_segment_id_falls_back_to_raw_dict_without_raising():
    """Even a segment_id too malformed to build a valid SegmentWorkerOutput
    (the primary failure-path model) must still come back as a JSON-safe
    soft failure, never an unhandled exception."""
    payload = _make_payload(start_ts=0.0, end_ts=10.0)
    payload["segment_id"] = "not-a-uuid"

    result = tasks.run_segment_worker(payload)

    assert result["status"] == SegmentStatus.failed.value
    assert result["error"]
    assert result["segment_id"] == "not-a-uuid"


class _SoftTimeLimitShotDetector:
    """Simulates what Celery actually raises inside a task when its soft time
    limit is hit — used since we can't trigger a real SoftTimeLimitExceeded
    signal from a plain pytest process without an actual Celery worker."""

    def detect(self, *args, **kwargs):
        raise SoftTimeLimitExceeded()


class _SoftTimeLimitTranscriber:
    """Same as _SoftTimeLimitShotDetector, but on Transcriber — the one
    adapter that always runs regardless of enable_secondary_adapters."""

    def transcribe(self, *args, **kwargs):
        raise SoftTimeLimitExceeded()


class TestFailureInjection:
    """Step 4: workers.segment_worker.tasks._apply_failure_injection —
    gated behind config.Settings.enable_failure_injection, which MUST default
    to False (a magic query string must never accidentally trigger simulated
    failures outside a deliberate test)."""

    def test_disabled_by_default_marker_has_no_effect(self):
        assert get_settings().enable_failure_injection is False
        payload = _make_payload(start_ts=0.0, end_ts=10.0)
        payload["source_video_url"] = "https://example.com/movie.mp4?__inject_failure=crash"

        result = tasks.run_segment_worker(payload)

        assert result["status"] == SegmentStatus.completed.value
        assert result["error"] is None

    def test_crash_directive_raises_and_is_caught_as_failed(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "enable_failure_injection", True)
        payload = _make_payload(start_ts=0.0, end_ts=10.0)
        payload["source_video_url"] = "https://example.com/movie.mp4?__inject_failure=crash"

        result = tasks.run_segment_worker(payload)

        assert result["status"] == SegmentStatus.failed.value
        assert "failure injection" in result["error"]

    def test_seq_target_only_affects_matching_sequence_index(self, monkeypatch):
        """Every segment of a job shares the same source_video_url — the
        :seqN suffix is what lets a test target just one segment, so the
        reducer's mixed completed/degraded path can be demonstrated live
        (not just with hand-built dicts in tests/reducer/)."""
        monkeypatch.setattr(get_settings(), "enable_failure_injection", True)
        url = "https://example.com/movie.mp4?__inject_failure=crash:seq2"

        untargeted = _make_payload(start_ts=0.0, end_ts=10.0, sequence_index=0)
        untargeted["source_video_url"] = url
        targeted = _make_payload(start_ts=0.0, end_ts=10.0, sequence_index=2)
        targeted["source_video_url"] = url

        untargeted_result = tasks.run_segment_worker(untargeted)
        targeted_result = tasks.run_segment_worker(targeted)

        assert untargeted_result["status"] == SegmentStatus.completed.value
        assert targeted_result["status"] == SegmentStatus.failed.value

    def test_delay_directive_actually_sleeps(self, monkeypatch):
        import time

        monkeypatch.setattr(get_settings(), "enable_failure_injection", True)
        payload = _make_payload(start_ts=0.0, end_ts=10.0)
        payload["source_video_url"] = "https://example.com/movie.mp4?__inject_failure=delay:0.2"

        start = time.monotonic()
        result = tasks.run_segment_worker(payload)
        elapsed = time.monotonic() - start

        assert result["status"] == SegmentStatus.completed.value
        assert elapsed >= 0.2

    def test_unrecognized_directive_is_ignored_not_crashed_on(self, monkeypatch):
        monkeypatch.setattr(get_settings(), "enable_failure_injection", True)
        payload = _make_payload(start_ts=0.0, end_ts=10.0)
        payload["source_video_url"] = "https://example.com/movie.mp4?__inject_failure=not_a_real_directive"

        result = tasks.run_segment_worker(payload)

        assert result["status"] == SegmentStatus.completed.value

    def test_soft_time_limit_exceeded_sets_timeout_status_not_failed(self):
        """A caught SoftTimeLimitExceeded must produce SegmentStatus.timeout,
        distinct from a plain adapter crash (SegmentStatus.failed) — the DB
        record should say WHY a segment didn't make it, and the reducer
        (workers/reducer/logic.py) treats both as equally degraded but a
        human reading Segment.status/.error should be able to tell a timeout
        from a crash."""
        payload = _make_payload(start_ts=0.0, end_ts=10.0, sequence_index=5)
        adapters = tasks.SegmentWorkerAdapters(transcriber=_SoftTimeLimitTranscriber())

        result = tasks.run_segment_worker(payload, adapters=adapters)

        assert result["status"] == SegmentStatus.timeout.value
        assert result["sequence_index"] == 5
        assert "soft time limit" in result["error"]
