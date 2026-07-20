"""MotionFeatureExtractor adapter interface.

Step 2 backs this with an OpenCV frame-diff pass (or RAFT optical flow) over
the segment's video frames. Step 1 only needs the interface plus a
deterministic mock keyed off the segment's global timestamps.
"""

from abc import ABC, abstractmethod

from workers.segment_worker.adapters._determinism import stable_unit_interval

# 3 fps is a middle ground for frame-diff motion analysis: dense enough to
# catch most motion transitions within a typical ~5s segment (see
# DEFAULT_SHOT_INTERVAL_SECONDS in shot_detector.py) without decoding/diffing
# an excessive number of frames per segment. Override via the constructor if
# a caller needs finer/coarser temporal resolution.
DEFAULT_MOTION_SAMPLE_FPS = 3.0


class MotionFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, source_video_url: str, start_ts: float, end_ts: float) -> dict[str, float]:
        """Return scalar motion features summarizing [start_ts, end_ts) —
        GLOBAL timestamps, not segment-local."""


class MockMotionFeatureExtractor(MotionFeatureExtractor):
    """Step 1 stand-in for OpenCV frame-diff/RAFT. Values are deterministic
    functions of (start_ts, end_ts) — no frames are actually decoded."""

    def extract(self, source_video_url: str, start_ts: float, end_ts: float) -> dict[str, float]:
        duration = max(end_ts - start_ts, 0.0)
        mean_motion = stable_unit_interval(start_ts, end_ts, "motion_mean")
        variance = stable_unit_interval(start_ts, end_ts, "motion_variance") * 0.25
        # Guarded so a zero-duration segment can't produce a nonsensical
        # peak_motion_ts (e.g. dividing by a zero-length span).
        peak_ts = start_ts + duration / 2.0 if duration > 0 else start_ts
        return {
            "mean_motion_magnitude": round(mean_motion, 4),
            "motion_variance": round(variance, 4),
            "peak_motion_ts": round(peak_ts, 4),
        }


class OpenCVMotionFeatureExtractor(MotionFeatureExtractor):
    """Step 2 real implementation backed by OpenCV frame-differencing.

    Samples frames uniformly across [start_ts, end_ts) via
    `extracted_frames_uniform` (see _media.py) at `fps` (default
    DEFAULT_MOTION_SAMPLE_FPS = 3.0 — see rationale above), converts each
    consecutive pair to grayscale, and measures per-pair motion as the mean
    absolute pixel difference (`cv2.absdiff`), normalized from the raw
    [0, 255] pixel-intensity scale down to [0, 1] so results are roughly
    comparable in magnitude to MockMotionFeatureExtractor's
    stable_unit_interval-derived values.

    `peak_motion_ts` is defined as `start_ts + frame_index/fps`, where
    `frame_index` is the (0-based) index of the *earlier* frame in the
    highest-diff pair — i.e. the sampled instant motion was detected
    transitioning away from.

    A more accurate alternative (not implemented here — left as a documented
    future option rather than guessed at) would be dense optical flow via
    RAFT, e.g. `torchvision.models.optical_flow.raft_large`; this requires
    adding `torchvision` as a new dependency (not currently declared in
    pyproject.toml's `gpu` extra) plus GPU-aware batching/preprocessing, so
    it was deliberately left out of this change.
    """

    def __init__(self, fps: float = DEFAULT_MOTION_SAMPLE_FPS):
        self._fps = fps

    def extract(self, source_video_url: str, start_ts: float, end_ts: float) -> dict[str, float]:
        import statistics

        import cv2

        from workers.segment_worker.adapters._media import extracted_frames_uniform

        with extracted_frames_uniform(source_video_url, start_ts, end_ts, self._fps) as frame_paths:
            if len(frame_paths) < 2:
                # Too short/degenerate a window to diff any frame pair (e.g.
                # a sub-frame-interval segment) — neutral result rather than
                # a crash, matching MockMotionFeatureExtractor's guarded
                # zero-duration handling above.
                return {
                    "mean_motion_magnitude": 0.0,
                    "motion_variance": 0.0,
                    "peak_motion_ts": round(start_ts, 4),
                }

            diffs: list[float] = []
            for prev_path, curr_path in zip(frame_paths, frame_paths[1:]):
                prev_gray = cv2.cvtColor(cv2.imread(prev_path), cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(cv2.imread(curr_path), cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, curr_gray)
                diffs.append(float(diff.mean()) / 255.0)

            peak_index = max(range(len(diffs)), key=diffs.__getitem__)
            return {
                "mean_motion_magnitude": round(statistics.fmean(diffs), 4),
                "motion_variance": round(statistics.pvariance(diffs), 4),
                "peak_motion_ts": round(start_ts + peak_index / self._fps, 4),
            }
