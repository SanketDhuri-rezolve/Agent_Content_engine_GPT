"""Splits a film's total duration into N segments with configurable overlap
at each internal boundary.

Segment/span timestamps are always GLOBAL (seconds from the start of the
whole film), never segment-local — see models.orm.Segment's docstring. This
matters downstream: the reducer (Stage 5) dedupes boundary-touching spans
across adjacent segments by comparing these timestamps directly, so any
segment-local numbering here would silently corrupt that logic.
"""

from abc import ABC, abstractmethod


def compute_segment_plan(
    total_duration_seconds: float,
    segment_count: int,
    overlap_seconds: float,
) -> list[dict]:
    """Builds the sharding plan for one job.

    Returns a list of `segment_count` dicts, one per segment, each shaped:
        {sequence_index, start_ts, end_ts, overlap_start, overlap_end}
    All four timestamp fields are GLOBAL seconds from the start of the film.

    Core `[start_ts, end_ts)` boundaries are computed as exact fractions of
    `total_duration_seconds` (`total * i / segment_count`), so they tile the
    full duration exactly with no gap and no overlap between adjacent
    segments' *core* windows — this holds even when `total_duration_seconds`
    doesn't divide evenly by `segment_count`.

    `overlap_start`/`overlap_end` widen each segment's *processing* window by
    `overlap_seconds` into the neighboring segment, so a worker processing
    segment i also sees a little of segment i-1 and i+1 and can capture spans
    that straddle a boundary; the reducer later dedups those using the
    global timestamps. The first segment's `overlap_start` is clamped to 0
    and the last segment's `overlap_end` is clamped to `total_duration_seconds`
    — there is nothing before the start or after the end of the film to
    overlap with.

    Args:
        total_duration_seconds: full film duration, in seconds. Must be > 0.
        segment_count: number of segments to shard into. Must be >= 1. Not a
            value this function should ever default on its own behalf — see
            config.Settings.provisional_dev_segment_count for why callers
            must pass this explicitly.
        overlap_seconds: seconds of overlap applied at each internal
            boundary. Must be >= 0. 0 means segments are contiguous with no
            overlap.
    """
    if segment_count < 1:
        raise ValueError(f"segment_count must be >= 1, got {segment_count}")
    if total_duration_seconds <= 0:
        raise ValueError(f"total_duration_seconds must be > 0, got {total_duration_seconds}")
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")

    plan: list[dict] = []
    for i in range(segment_count):
        start_ts = total_duration_seconds * i / segment_count
        end_ts = total_duration_seconds * (i + 1) / segment_count
        overlap_start = max(0.0, start_ts - overlap_seconds)
        overlap_end = min(total_duration_seconds, end_ts + overlap_seconds)
        plan.append(
            {
                "sequence_index": i,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
            }
        )
    return plan


class DurationProbe(ABC):
    """Step-2 hook point. A real implementation will determine a source
    video's total duration (ffprobe against a downloaded/streamed file, a
    metadata API call, etc.). Step 1 only needs this interface plus a mock so
    orchestrator logic can be exercised end-to-end without a real video file
    or network call — see CLAUDE.md's "every model call is behind an adapter
    interface" convention."""

    @abstractmethod
    def probe(self, source_video_url: str) -> float:
        """Returns the total duration of the video at source_video_url, in
        seconds."""


class MockDurationProbe(DurationProbe):
    """Zero-network, zero-GPU stand-in for Step 1. Always returns the same
    configured duration regardless of source_video_url. Step 2 swaps in a
    real ffprobe/download-based DurationProbe implementation behind this same
    interface — callers of DurationProbe never change."""

    def __init__(self, fixed_duration_seconds: float = 3600.0):
        self._fixed_duration_seconds = fixed_duration_seconds

    def probe(self, source_video_url: str) -> float:
        return self._fixed_duration_seconds


class FfprobeDurationProbe(DurationProbe):
    """Step 2 real implementation — shells out to ffprobe. No GPU dependency
    (ffprobe is CPU-only), but does require the `ffmpeg` package's `ffprobe`
    binary on PATH (see workers/segment_worker/Dockerfile.gpu, or install
    system-wide for local Step 2 testing). Import of this class has zero
    cost either way — the subprocess call only happens inside probe()."""

    def probe(self, source_video_url: str) -> float:
        from workers.segment_worker.adapters._media import probe_duration_seconds

        return probe_duration_seconds(source_video_url)


def get_duration_probe() -> DurationProbe:
    """Single call site controlling which DurationProbe implementation is
    active, mirroring workers.scorer.adapters.get_scorer()'s pattern: Step 1
    environments (no ffmpeg/ffprobe assumed present) get MockDurationProbe;
    Step 2 environments opt in via config.Settings.use_real_duration_probe."""
    from config import get_settings

    if get_settings().use_real_duration_probe:
        return FfprobeDurationProbe()
    return MockDurationProbe()
