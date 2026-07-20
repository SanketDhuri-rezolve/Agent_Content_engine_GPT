"""Shared ffmpeg/ffprobe-based media extraction helpers for Step 2's real
adapters (TransNetV2, faster-whisper, pyannote.audio, InternVideo2, OpenCV/
RAFT, OpenSMILE/torchaudio all need to pull actual frames/audio out of
`source_video_url` for a given GLOBAL [start_ts, end_ts) window).

Deliberately NOT imported by anything in Step 1 (MockShotDetector etc. never
touch a real video/audio file) — this module has no import-time dependency on
anything beyond the stdlib, so it's always safe to import even without
ffmpeg installed; only calling its functions requires the `ffmpeg`/`ffprobe`
binaries to actually be on PATH (see workers/segment_worker/Dockerfile.gpu).

Keyframe reference contract (fixes the seam between ShotDetector and
VisualEmbedder so two independently-built adapters agree without a shared
cache/store): a real ShotDetector's `keyframe_ref` MUST be
`f"kf_t{timestamp:.3f}"` where `timestamp` is the GLOBAL timestamp (seconds)
of that exact keyframe. `parse_keyframe_timestamp` below recovers it. This
lets VisualEmbedder re-extract the single frame it needs directly from
`source_video_url` at that timestamp, statelessly — no keyframe image needs
to be persisted/handed off between adapters or between Celery tasks.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_KEYFRAME_REF_PATTERN = re.compile(r"^kf_t(?P<timestamp>\d+(?:\.\d+)?)$")


def make_keyframe_ref(timestamp: float) -> str:
    return f"kf_t{timestamp:.3f}"


def parse_keyframe_timestamp(keyframe_ref: str) -> float:
    """Inverse of make_keyframe_ref. Raises ValueError if keyframe_ref wasn't
    produced by a real (Step 2) ShotDetector in this format — e.g. a Step 1
    Mock-style ref like "kf_0.000_0001" will not match."""
    match = _KEYFRAME_REF_PATTERN.match(keyframe_ref)
    if not match:
        raise ValueError(
            f"keyframe_ref {keyframe_ref!r} is not in the real-adapter kf_t<timestamp> "
            "format (see workers/segment_worker/adapters/_media.py)"
        )
    return float(match.group("timestamp"))


class MediaExtractionError(Exception):
    """ffmpeg/ffprobe failed, or returned no usable output."""


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise MediaExtractionError(
            "ffmpeg binary not found on PATH — install it in the runtime image "
            "(see workers/segment_worker/Dockerfile.gpu)"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaExtractionError(f"ffmpeg timed out: {' '.join(args)}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise MediaExtractionError(f"ffmpeg failed ({' '.join(args)}): {stderr[:1000]}") from exc


def probe_duration_seconds(source_video_url: str) -> float:
    """Real (Step 2) implementation backing orchestrator.splitter.DurationProbe.
    Uses ffprobe directly against source_video_url (works against remote HTTP(S)
    URLs as well as local paths — ffprobe reads container-level metadata
    without downloading the full file for most formats/servers that support
    HTTP range requests)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                source_video_url,
            ],
            check=True,
            capture_output=True,
            timeout=60,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MediaExtractionError(
            "ffprobe binary not found on PATH — install ffmpeg in the runtime image"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaExtractionError(f"ffprobe timed out probing {source_video_url}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise MediaExtractionError(f"ffprobe failed probing {source_video_url}: {stderr[:1000]}") from exc

    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise MediaExtractionError(
            f"ffprobe returned a non-numeric duration for {source_video_url}: {result.stdout!r}"
        ) from exc


@contextmanager
def extracted_frame(source_video_url: str, timestamp: float) -> Iterator[str]:
    """Extracts a single frame at GLOBAL `timestamp` to a temp JPEG, yields
    its path, deletes it on exit. `-ss` before `-i` seeks fast (input
    demuxer-level seek) at the cost of exact-frame precision, which is
    acceptable here — callers use this for keyframe embedding, not
    frame-perfect analysis."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = str(Path(tmp_dir) / "frame.jpg")
        _run_ffmpeg(
            ["-ss", f"{timestamp:.3f}", "-i", source_video_url, "-frames:v", "1", "-q:v", "2", out_path]
        )
        yield out_path


@contextmanager
def extracted_frames_uniform(
    source_video_url: str, start_ts: float, end_ts: float, fps: float
) -> Iterator[list[str]]:
    """Extracts frames sampled at `fps` across [start_ts, end_ts) to temp
    JPEGs (numbered frame_0001.jpg, frame_0002.jpg, ...), yields their paths
    in order, deletes them on exit. For shot-boundary detection, which needs
    a dense-ish sequence rather than one exact frame."""
    duration = max(end_ts - start_ts, 0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        if duration <= 0.0 or fps <= 0.0:
            yield []
            return
        pattern = str(Path(tmp_dir) / "frame_%05d.jpg")
        _run_ffmpeg(
            [
                "-ss", f"{start_ts:.3f}",
                "-i", source_video_url,
                "-t", f"{duration:.3f}",
                "-vf", f"fps={fps}",
                "-q:v", "2",
                pattern,
            ]
        )
        yield sorted(str(p) for p in Path(tmp_dir).glob("frame_*.jpg"))


@contextmanager
def extracted_audio_wav(
    source_video_url: str, start_ts: float, end_ts: float, sample_rate: int = 16000
) -> Iterator[str]:
    """Extracts the mono PCM WAV audio track for [start_ts, end_ts) to a temp
    file at `sample_rate` Hz, yields its path, deletes it on exit. Used by
    Transcriber (faster-whisper), Diarizer (pyannote.audio), and
    AudioFeatureExtractor (OpenSMILE/torchaudio) — all three want the same
    extracted clip, just processed differently, so each adapter should call
    this rather than re-implement its own ffmpeg invocation."""
    duration = max(end_ts - start_ts, 0.0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = str(Path(tmp_dir) / "audio.wav")
        if duration <= 0.0:
            # No audio to extract for a degenerate window — write a minimal
            # valid empty WAV rather than raising, so callers' "zero/inverted
            # duration segment" handling stays a normal empty-result path
            # (consistent with every Mock adapter's own behavior) instead of
            # a special-cased exception.
            _run_ffmpeg(
                ["-f", "lavfi", "-i", "anullsrc=r=%d:cl=mono" % sample_rate, "-t", "0.01", out_path]
            )
            yield out_path
            return

        _run_ffmpeg(
            [
                "-ss", f"{start_ts:.3f}",
                "-i", source_video_url,
                "-t", f"{duration:.3f}",
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", "wav",
                out_path,
            ]
        )
        yield out_path


@contextmanager
def extracted_av_clip(source_video_url: str, start_ts: float, end_ts: float) -> Iterator[str]:
    """Extracts a real, playable video+audio clip for [start_ts, end_ts) to a
    temp MP4, yields its path, deletes it on exit. Used by
    workers.scorer.adapters.Gemma4Scorer to (a) save each scored highlight as
    an actual deliverable clip via storage.object_storage, and (b) source
    keyframe images + an audio track for the multimodal scoring call itself,
    rather than only the aggregate feature_vector.

    Re-encodes (does not stream-copy) so the cut lands on the exact requested
    boundaries — `-c copy` can only cut on the source's keyframe boundaries,
    which would visibly shift a human-facing highlight clip's start/end away
    from what was actually requested. `veryfast`/CRF 23 is a reasonable
    quality/speed trade-off for a short highlight clip, not tuned further."""
    duration = max(end_ts - start_ts, 0.0)
    if duration <= 0.0:
        raise MediaExtractionError(
            f"extracted_av_clip requires end_ts > start_ts, got start_ts={start_ts}, end_ts={end_ts}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = str(Path(tmp_dir) / "clip.mp4")
        _run_ffmpeg(
            [
                "-ss", f"{start_ts:.3f}",
                "-i", source_video_url,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-c:a", "aac",
                out_path,
            ]
        )
        yield out_path
