"""Stage 2 (segment_worker) Celery task.

Queue: "segment_worker". Task name:
"workers.segment_worker.tasks.run_segment_worker" (fixed — do not rename,
see the cross-stage Celery contract in CLAUDE.md; other stages are being
built in parallel against this exact dotted path/shape).

Input is models.schemas.SegmentTaskPayload as a dict (job_id, segment_id,
sequence_index, source_video_url, start_ts, end_ts, overlap_start,
overlap_end — UUIDs/datetimes as str, everything else JSON-safe). Output is
models.schemas.SegmentWorkerOutput as a dict, consumed next by
workers.span_builder.tasks.build_spans.

Step 1: every model call (shot detection, transcription, diarization, visual
embedding, motion/audio features) is fully mocked behind the adapter
interfaces in workers.segment_worker.adapters — no GPU, no network calls, no
real video/audio decoding happens anywhere in this module. Step 2 swaps in
real adapter implementations (TransNetV2, faster-whisper, pyannote.audio,
InternVideo2, OpenCV, OpenSMILE) behind those same interfaces, without
touching this file's public contract.

Step 2 selection is config-driven, per-adapter, all defaulting to Mock (see
config.Settings.use_real_*): SegmentWorkerAdapters' defaults only construct a
Real* class when its flag is explicitly set, so importing/running this module
in a Step 1 (no-GPU) environment is completely unaffected. Each Real* class's
heavy imports (torch, transformers, faster_whisper, pyannote.audio, cv2,
opensmile) are deferred inside its own __init__/methods — importing the class
itself here is always safe.

start_ts/end_ts (and overlap_start/overlap_end) are GLOBAL timestamps
relative to the whole film, never segment-local — required by the reducer's
cross-segment dedup logic (see models/orm.py::Segment).
"""

import time
import urllib.parse
import uuid
from contextlib import nullcontext
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded

from config import get_settings
from models.enums import SegmentStatus
from models.schemas import SegmentTaskPayload, SegmentWorkerOutput, TranscriptSegment
from orchestrator.celery_app import celery_app
from workers.segment_worker.adapters._media import extracted_audio_wav
from workers.segment_worker.adapters.audio_features import (
    AudioFeatureExtractor,
    MockAudioFeatureExtractor,
    OpenSmileAudioFeatureExtractor,
)
from workers.segment_worker.adapters.diarizer import (
    Diarizer,
    MockDiarizer,
    PyannoteDiarizer,
    SpeakerTurn,
)
from workers.segment_worker.adapters.face_detector import (
    FaceDetector,
    MockFaceDetector,
    MTCNNFaceDetector,
)
from workers.segment_worker.adapters.memory_extractor import (
    MemoryExtractor,
    get_memory_extractor,
)
from workers.segment_worker.adapters.motion_features import (
    MockMotionFeatureExtractor,
    MotionFeatureExtractor,
    OpenCVMotionFeatureExtractor,
)
from workers.segment_worker.adapters.object_detector import (
    GroundingDinoObjectDetector,
    MockObjectDetector,
    ObjectDetector,
)
from workers.segment_worker.adapters.shot_detector import (
    MockShotDetector,
    ShotDetector,
    TransNetV2ShotDetector,
)
from workers.segment_worker.adapters.transcriber import (
    FasterWhisperTranscriber,
    MockTranscriber,
    Transcriber,
)
from workers.segment_worker.adapters.visual_embedder import (
    ClipVisualEmbedder,
    MockVisualEmbedder,
    VisualEmbedder,
)

logger = structlog.get_logger()

# Read once at import time (matches orchestrator/celery_app.py's own
# _settings = get_settings() pattern) — Celery task decorators run at module
# load, so soft_time_limit/time_limit must be plain values here, not a
# get_settings() call deferred to inside the task body.
_settings = get_settings()


def _apply_failure_injection(source_video_url: str, sequence_index: Any) -> None:
    """Step 4 test-only hook — gated behind config.Settings.enable_failure_injection
    (default False), so a magic query string can never accidentally trigger
    simulated failures outside a deliberate test run.

    Parses a `__inject_failure` query param out of source_video_url:
      - `__inject_failure=crash` raises immediately — a controllable,
        reproducible way to exercise the existing fail-soft exception path
        (see run_segment_worker's except Exception clause) without waiting
        for a real adapter to actually break.
      - `__inject_failure=delay:<seconds>` sleeps for that long — lets a real
        Celery soft/hard time limit (or a live `docker compose kill
        segment_worker` during a test) actually trigger mid-task on demand,
        rather than needing to race a near-instant mock call.
      - Either directive may be suffixed with `:seq<N>` (e.g.
        `delay:20:seq1`) to apply ONLY to the segment whose sequence_index is
        N, leaving every other segment of the same job untouched. Every
        segment of one job shares the exact same source_video_url (it's the
        whole film's URL, not per-segment), so without this a bare directive
        would apply uniformly to every segment — fine for an "everything
        fails" test, useless for demonstrating that the reducer still
        degrades gracefully when only SOME segments fail.
    """
    if not get_settings().enable_failure_injection:
        return

    query = urllib.parse.urlparse(source_video_url).query
    directives = urllib.parse.parse_qs(query).get("__inject_failure")
    if not directives:
        return

    directive = directives[0]
    if ":seq" in directive:
        directive, _, target_sequence_str = directive.rpartition(":seq")
        try:
            if int(target_sequence_str) != sequence_index:
                return
        except ValueError:
            return

    if directive == "crash":
        raise RuntimeError("failure injection: simulated crash (__inject_failure=crash)")
    if directive.startswith("delay:"):
        try:
            delay_seconds = float(directive.split(":", 1)[1])
        except ValueError:
            return
        time.sleep(delay_seconds)


def _default_shot_detector() -> ShotDetector:
    return TransNetV2ShotDetector() if get_settings().use_real_shot_detector else MockShotDetector()


def _default_transcriber() -> Transcriber:
    return FasterWhisperTranscriber() if get_settings().use_real_transcriber else MockTranscriber()


def _default_diarizer() -> Diarizer:
    return PyannoteDiarizer() if get_settings().use_real_diarizer else MockDiarizer()


def _default_visual_embedder() -> VisualEmbedder:
    return ClipVisualEmbedder() if get_settings().use_real_visual_embedder else MockVisualEmbedder()


def _default_motion_extractor() -> MotionFeatureExtractor:
    return (
        OpenCVMotionFeatureExtractor() if get_settings().use_real_motion_extractor else MockMotionFeatureExtractor()
    )


def _default_audio_extractor() -> AudioFeatureExtractor:
    return (
        OpenSmileAudioFeatureExtractor() if get_settings().use_real_audio_extractor else MockAudioFeatureExtractor()
    )


def _default_face_detector() -> FaceDetector:
    return MTCNNFaceDetector() if get_settings().use_real_face_detector else MockFaceDetector()


def _default_object_detector() -> ObjectDetector:
    return GroundingDinoObjectDetector() if get_settings().use_real_object_detector else MockObjectDetector()


class SegmentWorkerAdapters:
    """Bundle of the Stage-2 model adapters. Defaults are config-driven
    (config.Settings.use_real_*) via the _default_* functions above; any
    caller (this module, or tests) can still inject a specific adapter
    explicitly, e.g. the injection point tests use to exercise the
    exception-handling path with a deliberately broken adapter.

    memory_extractor defaults via get_memory_extractor() (Gemma4 vs Mock
    based on gemma4_endpoint_url being configured — same pattern as
    workers.scorer.adapters.get_scorer), not a use_real_* flag: it is only
    ever invoked when config.Settings.use_global_memory_pipeline is True (see
    build_segment_output), so there is no separate opt-in flag for the
    adapter itself."""

    def __init__(
        self,
        shot_detector: ShotDetector | None = None,
        transcriber: Transcriber | None = None,
        diarizer: Diarizer | None = None,
        visual_embedder: VisualEmbedder | None = None,
        motion_extractor: MotionFeatureExtractor | None = None,
        audio_extractor: AudioFeatureExtractor | None = None,
        face_detector: FaceDetector | None = None,
        object_detector: ObjectDetector | None = None,
        memory_extractor: MemoryExtractor | None = None,
    ):
        self.shot_detector = shot_detector or _default_shot_detector()
        self.transcriber = transcriber or _default_transcriber()
        self.diarizer = diarizer or _default_diarizer()
        self.visual_embedder = visual_embedder or _default_visual_embedder()
        self.motion_extractor = motion_extractor or _default_motion_extractor()
        self.audio_extractor = audio_extractor or _default_audio_extractor()
        self.face_detector = face_detector or _default_face_detector()
        self.object_detector = object_detector or _default_object_detector()
        self.memory_extractor = memory_extractor or get_memory_extractor()


def _merge_speaker_labels(
    transcript: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[TranscriptSegment]:
    """Attach a speaker_label to each transcript line by matching its
    midpoint against the diarizer's turns. Transcriber and Diarizer are
    independent adapters (real faster-whisper/pyannote don't share
    boundaries either), so this overlap-match is deliberate glue code, not a
    shortcut specific to the mocks."""
    if not turns:
        return transcript

    merged: list[TranscriptSegment] = []
    for line in transcript:
        midpoint = (line.start_ts + line.end_ts) / 2.0
        label = next(
            (turn["speaker_label"] for turn in turns if turn["start_ts"] <= midpoint < turn["end_ts"]),
            None,
        )
        merged.append(line if label is None else line.model_copy(update={"speaker_label": label}))
    return merged


def build_segment_output(
    payload: SegmentTaskPayload, adapters: SegmentWorkerAdapters | None = None
) -> SegmentWorkerOutput:
    """Pure composition of the seven adapters into a SegmentWorkerOutput — no
    Celery, no logging side effects, so it is directly unit-testable
    (tests/test_segment_worker.py) with injected adapters.

    Runs the adapters sequentially, not concurrently. A thread-pool version
    was tried and measured on real GPU hardware to be ~40% SLOWER (229s vs
    165s on a 90s clip, single A40): all GPU-resident models share one GPU's
    single compute engine, so concurrent submission only adds VRAM
    contention and context-switch overhead — it does not yield real
    parallelism. Genuine parallelism for this pipeline must come from
    running segments on separate GPUs (horizontal fan-out), not from
    concurrency within one segment on one GPU. See CLAUDE.md.

    FaceDetector + VisualEmbedder.embed_faces (character-ID foundation) run
    right after VisualEmbedder.embed, reusing the same keyframe_refs and the
    same already-loaded CLIP model for face-crop embedding — no new visual
    model is loaded for this.

    Whisper+Gemma4-only mode (config.Settings.enable_secondary_adapters =
    False, the current default): shot_detector, diarizer, visual_embedder,
    face_detector, motion_extractor, audio_extractor, object_detector are
    skipped ENTIRELY — not even their Mock stand-ins run. Only Transcriber
    (real Whisper) does GPU work here; Gemma4Scorer (downstream, in the
    scorer stage) samples its own keyframe images directly from each span's
    cropped clip, so none of the skipped adapters' output is needed for its
    real multimodal scoring call. See CLAUDE.md for the measured cost this
    removes (Face Detector+Embed alone was 54% of total segment time)."""
    adapters = adapters or SegmentWorkerAdapters()
    secondary_enabled = get_settings().enable_secondary_adapters

    t0 = time.monotonic()
    if secondary_enabled:
        shot_boundaries = adapters.shot_detector.detect(payload.source_video_url, payload.start_ts, payload.end_ts)
    else:
        shot_boundaries = []
    t1 = time.monotonic()

    # Transcriber, Diarizer, and AudioFeatureExtractor all independently
    # re-extracted the same audio window from the (network-mounted, on real
    # GPU hardware) source video — measured as redundant I/O contributing to
    # run-to-run timing variance. Extract it ONCE here and hand the shared
    # WAV path to all three (see each adapter's audio_path parameter).
    #
    # Only do this pre-extraction when at least one of the three is a REAL
    # adapter — Mock* adapters never touch audio_path at all, and Step 1's
    # explicit contract is zero ffmpeg/network calls when every adapter is
    # mocked (see module docstring); unconditionally shelling out to ffmpeg
    # here would silently break that for the all-mock (Step 1/test) path.
    needs_real_audio = isinstance(adapters.transcriber, FasterWhisperTranscriber) or (
        secondary_enabled
        and (
            isinstance(adapters.diarizer, PyannoteDiarizer)
            or isinstance(adapters.audio_extractor, OpenSmileAudioFeatureExtractor)
        )
    )
    audio_context = (
        extracted_audio_wav(payload.source_video_url, payload.start_ts, payload.end_ts)
        if needs_real_audio
        else nullcontext(None)
    )
    with audio_context as audio_path:
        transcript = adapters.transcriber.transcribe(
            payload.source_video_url, payload.start_ts, payload.end_ts, audio_path=audio_path
        )
        t2 = time.monotonic()
        if secondary_enabled:
            speaker_turns = adapters.diarizer.diarize(
                payload.source_video_url, payload.start_ts, payload.end_ts, audio_path=audio_path
            )
        else:
            speaker_turns = []
        t3 = time.monotonic()
        if secondary_enabled:
            audio_features = adapters.audio_extractor.extract(
                payload.source_video_url, payload.start_ts, payload.end_ts, audio_path=audio_path
            )
        else:
            audio_features = {}
        t3b = time.monotonic()

    transcript = _merge_speaker_labels(transcript, speaker_turns)

    if secondary_enabled:
        # VisualEmbedder runs only on keyframes ShotDetector actually emitted
        # for this segment — never against raw/all frames.
        keyframe_refs = [shot.keyframe_ref for shot in shot_boundaries if shot.keyframe_ref]
        visual_embeddings = adapters.visual_embedder.embed(payload.source_video_url, keyframe_refs)
        t4 = time.monotonic()

        # Character-ID foundation: detect faces on the same keyframes, then
        # embed each detected face crop with the SAME already-loaded CLIP
        # model (see ClipVisualEmbedder.embed_faces) — no second visual
        # model. Cross-segment clustering/naming happens later, in the
        # reducer.
        face_detections = adapters.face_detector.detect(payload.source_video_url, keyframe_refs)
        face_embeddings = adapters.visual_embedder.embed_faces(payload.source_video_url, face_detections)
        t4b = time.monotonic()

        # Object-identification foundation (see CLAUDE.md's commerce
        # roadmap): detection only, on the same keyframes — no catalog
        # matching/ranking yet, just "is there a recognizable shoppable
        # object here, and where."
        shoppable_objects = adapters.object_detector.detect(payload.source_video_url, keyframe_refs)
        t4c = time.monotonic()

        motion_features = adapters.motion_extractor.extract(
            payload.source_video_url, payload.start_ts, payload.end_ts
        )
        t5 = time.monotonic()
    else:
        visual_embeddings = {}
        face_detections = {}
        face_embeddings = {}
        shoppable_objects = {}
        motion_features = {}
        t4 = t4b = t4c = t5 = t3b

    use_global_memory_pipeline = get_settings().use_global_memory_pipeline
    local_memory: dict | None = None
    if use_global_memory_pipeline:
        try:
            local_memory = adapters.memory_extractor.extract(str(payload.segment_id), transcript)
        except Exception as exc:  # noqa: BLE001 - fail-soft: missing memory must not fail the segment
            logger.warning(
                "segment_worker.memory_extraction_failed",
                segment_id=str(payload.segment_id),
                error=str(exc),
            )
    t6 = time.monotonic()

    logger.info(
        "segment_worker.adapter_timings",
        segment_id=str(payload.segment_id),
        secondary_adapters_enabled=secondary_enabled,
        use_global_memory_pipeline=use_global_memory_pipeline,
        shot_detector_seconds=round(t1 - t0, 2),
        transcriber_seconds=round(t2 - t1, 2),
        diarizer_seconds=round(t3 - t2, 2),
        audio_extractor_seconds=round(t3b - t3, 2),
        visual_embedder_seconds=round(t4 - t3b, 2),
        face_detector_and_embed_seconds=round(t4b - t4, 2),
        object_detector_seconds=round(t4c - t4b, 2),
        motion_extractor_seconds=round(t5 - t4c, 2),
        memory_extractor_seconds=round(t6 - t5, 2),
    )

    return SegmentWorkerOutput(
        segment_id=payload.segment_id,
        sequence_index=payload.sequence_index,
        status=SegmentStatus.completed,
        shot_boundaries=shot_boundaries,
        transcript=transcript,
        visual_embeddings=visual_embeddings,
        face_detections=face_detections,
        face_embeddings=face_embeddings,
        shoppable_objects=shoppable_objects,
        motion_features=motion_features,
        audio_features=audio_features,
        local_memory=local_memory,
        error=None,
    )


def _failure_output(
    segment_id: Any, sequence_index: Any, error_message: str, status: SegmentStatus = SegmentStatus.failed
) -> dict[str, Any]:
    """Best-effort JSON-safe SegmentWorkerOutput-shaped dict for the failure
    path. Tries to build a real (validated) SegmentWorkerOutput first; if
    segment_id/sequence_index are themselves too malformed for that (e.g.
    segment_id isn't a valid UUID), falls back to a raw dict with the same
    keys — the reducer must always get a soft failure it can skip, never a
    pydantic ValidationError that kills the whole chord.

    `status` defaults to SegmentStatus.failed but callers pass `.timeout` for
    a caught SoftTimeLimitExceeded (see run_segment_worker) — the reducer
    treats any non-completed status as equally degraded (see
    workers/reducer/logic.py's classify_segments), but the DB record should
    still say WHY: a timeout isn't the same failure mode as an adapter crash."""
    try:
        resolved_sequence_index = sequence_index if isinstance(sequence_index, int) else int(sequence_index)
    except (TypeError, ValueError):
        resolved_sequence_index = -1

    try:
        return SegmentWorkerOutput(
            segment_id=segment_id,
            sequence_index=resolved_sequence_index,
            status=status,
            error=error_message,
        ).model_dump(mode="json")
    except Exception:
        return {
            "segment_id": str(segment_id) if segment_id is not None else str(uuid.uuid4()),
            "sequence_index": resolved_sequence_index,
            "status": status.value,
            "shot_boundaries": [],
            "transcript": [],
            "visual_embeddings": {},
            "motion_features": {},
            "audio_features": {},
            "error": error_message,
        }


@celery_app.task(
    name="workers.segment_worker.tasks.run_segment_worker",
    soft_time_limit=_settings.segment_worker_soft_time_limit_seconds,
    time_limit=_settings.segment_worker_hard_time_limit_seconds,
)
def run_segment_worker(payload: dict[str, Any], adapters: SegmentWorkerAdapters | None = None) -> dict[str, Any]:
    """`adapters` is an optional test/DI seam (defaults to the Step 1 mocks
    via build_segment_output) — Celery/other callers only ever pass
    `payload`, matching the fixed contract exactly.

    Step 4: soft_time_limit/time_limit are provisional (config.Settings.
    segment_worker_soft_time_limit_seconds/_hard_time_limit_seconds — not yet
    measured against real GPU workloads). Hitting the SOFT limit raises
    SoftTimeLimitExceeded inside this function — caught explicitly below and
    turned into a SegmentStatus.timeout soft-failure, same fail-soft
    contract as any other exception. Hitting the HARD limit instead kills
    this task's worker child process outright (SIGKILL) — there is no
    Python code left running to catch that; see orchestrator/celery_app.py's
    task_acks_late/task_reject_on_worker_lost and orchestrator/watchdog.py
    for how that case is handled instead.
    """
    job_id = payload.get("job_id")
    segment_id = payload.get("segment_id")
    sequence_index = payload.get("sequence_index")
    log = logger.bind(job_id=job_id, segment_id=segment_id, stage="segment_worker")
    log.info("stage.started")
    start = time.monotonic()

    try:
        task_payload = SegmentTaskPayload.model_validate(payload)
        _apply_failure_injection(task_payload.source_video_url, task_payload.sequence_index)
        output = build_segment_output(task_payload, adapters)
        elapsed = time.monotonic() - start
        log.info(
            "stage.completed",
            sequence_index=output.sequence_index,
            elapsed_seconds=elapsed,
            shot_count=len(output.shot_boundaries),
            transcript_line_count=len(output.transcript),
        )
        return output.model_dump(mode="json")
    except SoftTimeLimitExceeded as exc:
        elapsed = time.monotonic() - start
        error_message = (
            f"segment_worker soft time limit ({_settings.segment_worker_soft_time_limit_seconds}s) "
            f"exceeded for segment_id={segment_id}: {exc}"
        )
        log.error("stage.failed", elapsed_seconds=elapsed, error=error_message, reason="soft_time_limit_exceeded")
        return _failure_output(segment_id, sequence_index, error_message, status=SegmentStatus.timeout)
    except Exception as exc:  # noqa: BLE001 - a segment crashing must degrade
        # to a soft failure the reducer can see, never kill the whole chord.
        elapsed = time.monotonic() - start
        error_message = f"segment_worker failed for segment_id={segment_id}: {exc}"
        log.error("stage.failed", elapsed_seconds=elapsed, error=error_message)
        return _failure_output(segment_id, sequence_index, error_message)
