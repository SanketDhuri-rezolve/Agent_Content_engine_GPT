"""Diarizer adapter interface.

Step 2 backs this with pyannote.audio running against the segment's audio
track. Step 1 only needs the interface plus a deterministic mock. Diarizer
output is intentionally independent of Transcriber's line boundaries (real
pyannote/faster-whisper don't share boundaries either) — the caller
(workers.segment_worker.tasks) merges the two by overlap.

`PyannoteDiarizer` below is the Step 2 real implementation. Its heavy
third-party dependencies (`pyannote.audio`, `torch`) are imported lazily
(inside `PyannoteDiarizer.__init__`/`_pipeline`, never at module top level)
so that importing this module — which `workers.segment_worker.tasks` always
does — never requires those packages to be installed. Only *instantiating*
`PyannoteDiarizer` (or calling `diarize()` on one) does.
"""

import os
from abc import ABC, abstractmethod
from typing import TypedDict

from workers.segment_worker.adapters._media import extracted_audio_wav

DEFAULT_TURN_INTERVAL_SECONDS = 10.0
DEFAULT_NUM_SPEAKERS = 2

# pyannote.audio 4.0 (Nov 2025) superseded the long-standing
# "pyannote/speaker-diarization-3.1" pipeline with an open pipeline that
# significantly improves speaker counting/assignment; both are gated HF
# models requiring an access token that has accepted the model's terms.
# See https://huggingface.co/pyannote/speaker-diarization-community-1
DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"

# Env var checked when `hf_auth_token` isn't passed explicitly to
# PyannoteDiarizer. "PYANNOTE_AUTH_TOKEN" is used in preference to a generic
# "HUGGINGFACE_TOKEN"/"HF_TOKEN" name so it can't silently collide with an
# unrelated HF token some other part of the stack might set for a different
# purpose; still falls back to those common names if that's what's set.
_PYANNOTE_TOKEN_ENV_VARS = ("PYANNOTE_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN")


class SpeakerTurn(TypedDict):
    start_ts: float
    end_ts: float
    speaker_label: str


class Diarizer(ABC):
    @abstractmethod
    def diarize(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[SpeakerTurn]:
        """Return speaker turns covering [start_ts, end_ts) — GLOBAL
        timestamps, not segment-local.

        `audio_path`: optional pre-extracted WAV for this exact window — see
        AudioFeatureExtractor.extract's docstring for why (audio was
        independently re-extracted by Transcriber/Diarizer/
        AudioFeatureExtractor for the same window; now extracted once and
        shared). Falls back to extracting it internally if None."""


class MockDiarizer(Diarizer):
    """Step 1 stand-in for pyannote.audio. Alternates between
    `num_speakers` fake speaker labels every `turn_interval_seconds`,
    deterministically derived from start_ts/end_ts — no audio is actually
    decoded."""

    def __init__(
        self,
        turn_interval_seconds: float = DEFAULT_TURN_INTERVAL_SECONDS,
        num_speakers: int = DEFAULT_NUM_SPEAKERS,
    ):
        self._interval = turn_interval_seconds
        self._num_speakers = max(num_speakers, 1)

    def diarize(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[SpeakerTurn]:
        if end_ts <= start_ts or self._interval <= 0:
            # Degenerate/zero-duration segment — nothing to diarize, not an error.
            return []

        turns: list[SpeakerTurn] = []
        index = 0
        ts = start_ts
        while ts < end_ts:
            turn_end = min(ts + self._interval, end_ts)
            turns.append(
                {
                    "start_ts": round(ts, 4),
                    "end_ts": round(turn_end, 4),
                    "speaker_label": f"SPEAKER_{index % self._num_speakers:02d}",
                }
            )
            index += 1
            ts += self._interval
        return turns


class PyannoteDiarizer(Diarizer):
    """Step 2 real diarizer backed by pyannote.audio's pretrained speaker
    diarization pipeline (default: "pyannote/speaker-diarization-community-1",
    see DEFAULT_PYANNOTE_MODEL above).

    Requires the `pyannote-audio` PyPI package (>=4.0) plus `torch` to be
    installed — neither is imported until a `PyannoteDiarizer` is actually
    constructed, so importing this module stays dependency-free (see module
    docstring).

    The pretrained pipeline is gated on HuggingFace: you must accept its
    terms of use while logged in at the model URL above, then supply an HF
    access token (https://hf.co/settings/tokens) either as `hf_auth_token`
    or via one of the environment variables in `_PYANNOTE_TOKEN_ENV_VARS`
    (PYANNOTE_AUTH_TOKEN checked first).
    """

    # Class-level, keyed by (model_name, token) — NOT instance-level. A fresh
    # PyannoteDiarizer is constructed on every single run_segment_worker task
    # call (see workers.segment_worker.tasks._default_diarizer), so an
    # instance-level cache would reload the pretrained pipeline (network
    # fetch + model init) on every segment; a class-level cache means it
    # loads once per worker process and every instance shares it (mirrors
    # TransNetV2ShotDetector._MODEL_CACHE / FasterWhisperTranscriber._MODEL_CACHE
    # / InternVideo2VisualEmbedder._MODEL_CACHE elsewhere in this package).
    _PIPELINE_CACHE: dict[tuple[str, str | None], object] = {}

    def __init__(self, hf_auth_token: str | None = None, model_name: str | None = None):
        self._hf_auth_token = hf_auth_token
        if self._hf_auth_token is None:
            for env_var in _PYANNOTE_TOKEN_ENV_VARS:
                value = os.environ.get(env_var)
                if value:
                    self._hf_auth_token = value
                    break
        self._model_name = model_name or DEFAULT_PYANNOTE_MODEL

    def _get_pipeline(self):
        cache_key = (self._model_name, self._hf_auth_token)
        cached = PyannoteDiarizer._PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if not self._hf_auth_token:
            env_names = " / ".join(_PYANNOTE_TOKEN_ENV_VARS)
            raise RuntimeError(
                f"PyannoteDiarizer requires a HuggingFace access token that has accepted "
                f"the gated model's terms at https://huggingface.co/{self._model_name} — "
                f"pass hf_auth_token= explicitly or set one of: {env_names}."
            )
        import torch
        from pyannote.audio import Pipeline  # deferred heavy import — see module docstring

        pipeline = Pipeline.from_pretrained(self._model_name, token=self._hf_auth_token)
        # Confirmed necessary on real GPU hardware: Pipeline.from_pretrained()
        # defaults to CPU — pyannote never moves itself onto CUDA on its own.
        # Without this, diarization ran entirely on CPU (measured ~65s for a
        # 90s clip, by far the single largest per-segment cost — see
        # CLAUDE.md's adapter timing breakdown) despite torch/CUDA being
        # available and every other adapter using the GPU.
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        PyannoteDiarizer._PIPELINE_CACHE[cache_key] = pipeline
        return pipeline

    def diarize(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> list[SpeakerTurn]:
        if end_ts <= start_ts:
            # Degenerate/zero-duration segment — nothing to diarize, not an error.
            return []

        pipeline = self._get_pipeline()
        turns: list[SpeakerTurn] = []

        def _diarize_from(wav_path: str) -> None:
            # Confirmed necessary on real GPU hardware: passing a bare file
            # path here makes pyannote read it via torchcodec, which failed
            # to load its CUDA extension (libnvrtc.so.13 missing/version
            # mismatch with the installed torch build — an environment issue,
            # not something this adapter can fix). pyannote's own error
            # message documents the workaround: read the audio ourselves
            # (soundfile — a plain libsndfile wrapper, no CUDA/torchcodec
            # dependency at all, so it can't hit the same problem) and hand
            # it a waveform dict instead of a path.
            import numpy as np
            import soundfile as sf
            import torch

            audio_array, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(np.ascontiguousarray(audio_array.T))
            output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            # pyannote.audio 4.0's community-1 pipeline returns a DiarizeOutput
            # wrapper, not the classic Annotation directly (confirmed on real
            # GPU hardware — this differs from pyannote 3.x's return shape).
            # .exclusive_speaker_diarization is the non-overlapping-turns view
            # (an Annotation, has .itertracks()) — matches our SpeakerTurn
            # schema, which models one speaker per turn, not simultaneous
            # overlapping speakers (that's .speaker_diarization instead, if
            # ever needed).
            diarization = output.exclusive_speaker_diarization
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                # `turn.start`/`turn.end` are LOCAL to the extracted clip
                # (i.e. relative to `start_ts`) — add start_ts back to
                # recover GLOBAL timestamps, per the Diarizer contract.
                turns.append(
                    {
                        "start_ts": round(start_ts + turn.start, 4),
                        "end_ts": round(start_ts + turn.end, 4),
                        "speaker_label": speaker_label,
                    }
                )

        if audio_path is not None:
            _diarize_from(audio_path)
        else:
            with extracted_audio_wav(source_video_url, start_ts, end_ts) as wav_path:
                _diarize_from(wav_path)
        return turns
