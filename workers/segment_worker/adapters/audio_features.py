"""AudioFeatureExtractor adapter interface.

Step 2 backs this with OpenSMILE/torchaudio feature extraction over the
segment's audio track. Step 1 only needs the interface plus a deterministic
mock keyed off the segment's global timestamps.
"""

from abc import ABC, abstractmethod

from workers.segment_worker.adapters._determinism import stable_unit_interval
from workers.segment_worker.adapters._media import extracted_audio_wav


class AudioFeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> dict[str, float]:
        """Return scalar audio features summarizing [start_ts, end_ts) —
        GLOBAL timestamps, not segment-local.

        `audio_path`: optional pre-extracted WAV for this exact window (see
        workers.segment_worker.tasks.build_segment_output, which extracts
        audio ONCE per segment and shares it across Transcriber/Diarizer/
        AudioFeatureExtractor — measured on real GPU hardware to be
        redundant otherwise: all three independently re-extracted the same
        audio from the same network-mounted source video). If None, real
        implementations fall back to extracting it themselves (keeps this
        adapter usable standalone, e.g. in isolated tests/scripts)."""


class MockAudioFeatureExtractor(AudioFeatureExtractor):
    """Step 1 stand-in for OpenSMILE/torchaudio. Values are deterministic
    functions of (start_ts, end_ts) — no audio is actually decoded."""

    def extract(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> dict[str, float]:
        rms = stable_unit_interval(start_ts, end_ts, "rms_energy")
        pitch_hz = 80.0 + stable_unit_interval(start_ts, end_ts, "pitch_hz") * 300.0
        loudness_db = -30.0 + stable_unit_interval(start_ts, end_ts, "loudness_db") * 30.0
        return {
            "rms_energy_mean": round(rms, 4),
            "pitch_mean_hz": round(pitch_hz, 2),
            "loudness_peak_db": round(loudness_db, 2),
        }


class OpenSmileAudioFeatureExtractor(AudioFeatureExtractor):
    """Step 2 real implementation, backed by the `opensmile` PyPI package
    (audEERING's official Python wrapper around the openSMILE toolkit;
    https://github.com/audeering/opensmile-python), using the eGeMAPSv02
    Functionals feature set.

    All heavy imports (`opensmile`, and whatever it pulls in transitively —
    audinterface, pandas, numpy, etc.) are deferred to inside
    `_get_smile()`, never at module import time: this module is imported
    unconditionally by workers.segment_worker.tasks and by a large local
    test suite that runs with none of those packages installed, so a
    top-level import here would break every one of those tests (see
    MockAudioFeatureExtractor above, and FasterWhisperTranscriber /
    PyannoteDiarizer for the same deferred-import pattern elsewhere in this
    package).

    Audio is extracted at 16 kHz mono via `extracted_audio_wav`'s default
    `sample_rate` — this matches openSMILE/eGeMAPS's own documented
    convention (16 kHz/16-bit mono is the standard input format used across
    eGeMAPS feature-extraction pipelines), so there's no need to request a
    higher rate here.

    Feature mapping (eGeMAPSv02 Functionals column -> required output key —
    confirmed against opensmile-python's docs/README
    (audeering.github.io/opensmile-python) and the eGeMAPSv02 Functionals
    column list; eGeMAPSv02 has 88 Functionals columns total):

    - `pitch_mean_hz` <- `F0semitoneFrom27.5Hz_sma3nz_amean`: eGeMAPS' mean
      voiced-F0 functional, expressed in semitones relative to 27.5 Hz (the
      standard eGeMAPS pitch scale; see Eyben et al., "The Geneva
      Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and
      Affective Computing", IEEE Trans. Affective Computing, 2016). Converted
      back to linear Hz via the inverse of
      `semitones = 12 * log2(f0_hz / 27.5)`, i.e.
      `f0_hz = 27.5 * 2 ** (semitones / 12)`.

    - `rms_energy_mean` <- `loudness_sma3_amean`: eGeMAPS has no literal "RMS
      energy" functional. `loudness_sma3_amean` (mean of the Loudness LLD —
      an estimate of perceived signal intensity, summed across the auditory
      spectrum's equivalent-rectangular-bandwidth bands) is the closest
      available proxy for overall signal energy. APPROXIMATION: this is an
      auditory-perceptual loudness measure on openSMILE's own scale, not a
      raw normalized RMS amplitude — treat it as an energy-correlated proxy,
      not literal RMS.

    - `loudness_peak_db` <- `equivalentSoundLevel_dBp`: eGeMAPS has no
      literal "peak loudness in dB" functional either. The only other
      loudness-peak-shaped feature, `loudnessPeaksPerSec`, is a *rate*
      (peaks per second), not a level, so it can't stand in for a dB value.
      `equivalentSoundLevel_dBp` — the overall equivalent sound level in dB,
      integrated across the whole clip — is used here as the closest
      dB-scale proxy for "peak" loudness. APPROXIMATION: this is a
      clip-wide equivalent/average level, not a true instantaneous peak.
    """

    # Class-level, NOT instance-level — a fresh OpenSmileAudioFeatureExtractor
    # is constructed on every single run_segment_worker task call (see
    # workers.segment_worker.tasks._default_audio_extractor), so an
    # instance-level cache would rebuild the opensmile.Smile feature-extractor
    # config on every segment; a class-level cache builds it once per worker
    # process (mirrors TransNetV2ShotDetector._MODEL_CACHE /
    # FasterWhisperTranscriber._MODEL_CACHE / InternVideo2VisualEmbedder._MODEL_CACHE
    # / PyannoteDiarizer._PIPELINE_CACHE elsewhere in this package).
    _SMILE_CACHE: dict[tuple, object] = {}

    def __init__(self) -> None:
        # No opensmile import happens here — only on first extract() call
        # (see _get_smile) — so constructing this class is always cheap and
        # safe even without opensmile installed.
        pass

    def _get_smile(self):
        cache_key = ("eGeMAPSv02", "Functionals")
        cached = OpenSmileAudioFeatureExtractor._SMILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        import opensmile  # deferred: heavy/optional dependency

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        OpenSmileAudioFeatureExtractor._SMILE_CACHE[cache_key] = smile
        return smile

    def extract(
        self, source_video_url: str, start_ts: float, end_ts: float, audio_path: str | None = None
    ) -> dict[str, float]:
        if end_ts <= start_ts:
            # Degenerate/zero-duration segment — no audio to analyze.
            # openSMILE's F0/loudness functionals are undefined (NaN) over a
            # near-silent sub-frame clip, so short-circuit to neutral/zeroed
            # values rather than crashing — consistent with the guarded
            # degenerate-window handling used by sibling real adapters
            # (FasterWhisperTranscriber, PyannoteDiarizer,
            # OpenCVMotionFeatureExtractor) and with extracted_audio_wav's
            # own zero-duration handling in _media.py.
            return {
                "rms_energy_mean": 0.0,
                "pitch_mean_hz": 0.0,
                "loudness_peak_db": 0.0,
            }

        smile = self._get_smile()
        if audio_path is not None:
            features = smile.process_file(audio_path)
        else:
            with extracted_audio_wav(source_video_url, start_ts, end_ts) as wav_path:
                features = smile.process_file(wav_path)

        # Functionals-level output for a single whole file is exactly one
        # row (MultiIndex: file/start/end); columns are the named eGeMAPSv02
        # functionals themselves.
        row = features.iloc[0]

        pitch_semitones = float(row["F0semitoneFrom27.5Hz_sma3nz_amean"])
        # See class docstring: semitones-from-27.5Hz -> linear Hz.
        pitch_hz = 27.5 * (2.0 ** (pitch_semitones / 12.0))

        rms_energy = float(row["loudness_sma3_amean"])
        loudness_peak_db = float(row["equivalentSoundLevel_dBp"])

        return {
            "rms_energy_mean": round(rms_energy, 4),
            "pitch_mean_hz": round(pitch_hz, 2),
            "loudness_peak_db": round(loudness_peak_db, 2),
        }
