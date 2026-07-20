"""Transcriber adapter interface.

Step 2 backs this with faster-whisper running against the segment's audio
track. Step 1 only needs the interface plus a deterministic mock.
"""

from abc import ABC, abstractmethod

from models.schemas import TranscriptSegment
from workers.segment_worker.adapters._media import extracted_audio_wav

DEFAULT_LINE_INTERVAL_SECONDS = 8.0


class Transcriber(ABC):
    @abstractmethod
    def transcribe(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[TranscriptSegment]:
        """Return transcript lines inside [start_ts, end_ts) — GLOBAL
        timestamps, not segment-local. `speaker_label` is left unset here;
        diarization is a separate adapter (see diarizer.py) merged in by the
        caller (workers.segment_worker.tasks).

        `audio_path`: optional pre-extracted WAV for this exact window — see
        AudioFeatureExtractor.extract's docstring for why (audio was
        independently re-extracted by Transcriber/Diarizer/
        AudioFeatureExtractor for the same window; now extracted once and
        shared). Falls back to extracting it internally if None."""


class MockTranscriber(Transcriber):
    """Step 1 stand-in for faster-whisper. Emits one fake transcript line
    every `line_interval_seconds`, deterministically derived from
    start_ts/end_ts — no audio is actually decoded."""

    def __init__(self, line_interval_seconds: float = DEFAULT_LINE_INTERVAL_SECONDS):
        self._interval = line_interval_seconds

    def transcribe(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[TranscriptSegment]:
        if end_ts <= start_ts or self._interval <= 0:
            # Degenerate/zero-duration segment — nothing to transcribe, not an error.
            return []

        lines: list[TranscriptSegment] = []
        index = 0
        ts = start_ts
        while ts < end_ts:
            line_end = min(ts + self._interval, end_ts)
            lines.append(
                TranscriptSegment(
                    start_ts=round(ts, 4),
                    end_ts=round(line_end, 4),
                    text=f"[mock transcript line {index} @ {ts:.1f}s-{line_end:.1f}s]",
                    speaker_label=None,
                )
            )
            index += 1
            ts += self._interval
        return lines


class FasterWhisperTranscriber(Transcriber):
    """Step 2 real implementation, backed by faster-whisper (CTranslate2)
    running "large-v3-turbo" against the segment's extracted audio track.

    All heavy imports (`faster_whisper`, and whatever it pulls in
    transitively — ctranslate2, torch, etc.) are deferred to inside
    `__init__`/`_get_model`, never at module import time: this module is
    imported unconditionally by workers.segment_worker.tasks and by a large
    local test suite that runs with none of those packages installed, so a
    top-level import here would break every one of those tests.

    The loaded model is cached per (model_size, device, compute_type) on the
    class itself, so weights are loaded once per process — not once per
    `transcribe()` call, and shared across instances constructed with the
    same config — even though loading happens lazily on first use rather
    than in `__init__`.
    """

    # Class-level cache shared by every instance in this process, keyed by
    # the (model_size, device, compute_type) config a given instance was
    # built with. Left untyped against `faster_whisper.WhisperModel` on
    # purpose, so referencing this attribute never requires importing
    # faster_whisper.
    _MODEL_CACHE: dict[tuple[str, str, str], object] = {}

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        # No faster_whisper/torch import happens here — only on first
        # transcribe() call (see _get_model) — so constructing this class is
        # always cheap and safe even without those packages installed.
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type

    def _resolve_compute_type(self) -> str:
        if self._compute_type != "auto":
            return self._compute_type
        # CTranslate2's own "default" compute_type does NOT necessarily pick
        # the GPU-optimized precision — confirmed against CTranslate2's own
        # docs (https://opennmt.net/CTranslate2/quantization.html): "default"
        # falls back to the type the model was saved in, which is not
        # guaranteed to be float16 on CUDA. float16 is CTranslate2's own
        # documented, standard recommendation for running on a CUDA GPU
        # (roughly 2x+ throughput vs. float32, no meaningful transcription
        # quality loss for Whisper-family models) — explicitly requesting it
        # here rather than trusting "default" to have picked it already.
        import torch

        return "float16" if torch.cuda.is_available() else "default"

    def _get_model(self):
        compute_type = self._resolve_compute_type()
        cache_key = (self._model_size, self._device, compute_type)
        cached = FasterWhisperTranscriber._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        from faster_whisper import WhisperModel  # deferred: heavy/optional dependency

        model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=compute_type,
        )
        FasterWhisperTranscriber._MODEL_CACHE[cache_key] = model
        return model

    def transcribe(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[TranscriptSegment]:
        if end_ts <= start_ts:
            # Degenerate/zero-duration segment — nothing to transcribe, not an error
            # (consistent with MockTranscriber and extracted_audio_wav's own handling).
            return []

        model = self._get_model()

        def _transcribe_from(path: str) -> list:
            # condition_on_previous_text=False: faster-whisper's own default
            # (True) feeds each chunk's transcribed text back in as context
            # for the next chunk. Confirmed on real GPU hardware to be the
            # cause of runaway repetition loops on noisy/musical passages
            # (e.g. "Piano! Prove it! Prove it! Prove it!...") — once a chunk
            # comes out garbled, conditioning on that garbled text makes the
            # model MORE likely to keep repeating it, not less. This is
            # faster-whisper/Whisper's own well-documented failure mode;
            # disabling this makes every chunk transcribed independently, so
            # one bad chunk can't drag down the rest of the segment.
            raw_segments, _info = model.transcribe(path, condition_on_previous_text=False)
            # faster-whisper returns `raw_segments` as a lazy generator that reads
            # from `path` as it's iterated — it must be fully consumed here, before
            # a temp WAV (if any) gets deleted on context exit.
            return list(raw_segments)

        if audio_path is not None:
            raw_segments = _transcribe_from(audio_path)
        else:
            with extracted_audio_wav(source_video_url, start_ts, end_ts) as extracted_path:
                raw_segments = _transcribe_from(extracted_path)

        lines: list[TranscriptSegment] = []
        for segment in raw_segments:
            # faster-whisper's segment.start/segment.end are LOCAL to the extracted
            # clip we handed it (i.e. relative to `audio_path`, which itself starts
            # at `start_ts` of the source video) — add start_ts back to recover
            # GLOBAL timestamps against the full source video.
            lines.append(
                TranscriptSegment(
                    start_ts=round(start_ts + segment.start, 4),
                    end_ts=round(start_ts + segment.end, 4),
                    text=segment.text.strip(),
                    speaker_label=None,
                )
            )
        return lines
