"""Pydantic schemas.

Two use cases share this module:
  1. FastAPI request/response bodies (api/).
  2. Inter-stage message payloads passed through Celery (orchestrator/, workers/).
     Celery serializes task args/results as JSON, so these models are built
     with plain JSON-safe types (str timestamps via isoformat, UUID as str)
     rather than passing ORM objects across the wire.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from models.enums import JobStatus, SegmentStatus

# ---------------------------------------------------------------------------
# API: job submission / status / results
# ---------------------------------------------------------------------------


class JobCreateRequest(BaseModel):
    source_video_url: str
    sla_target_seconds: int | None = None
    # Optional: if omitted, api.main.submit_job falls back to
    # orchestrator.splitter.get_duration_probe() (real ffprobe-based in Step
    # 2 when config.Settings.use_real_duration_probe is set, otherwise
    # MockDurationProbe) to infer it from source_video_url. Pass it explicitly
    # to skip probing (e.g. in Step 1 tests/dev, or when the exact duration
    # is already known).
    total_duration_seconds: float | None = None
    # Explicit only. No inline hardcoded 8/12 default anywhere in this codebase —
    # if omitted, the orchestrator falls back to
    # config.Settings.provisional_dev_segment_count, which is documented as an
    # unconfirmed dev placeholder, not a production decision. Ignored if
    # segment_duration_seconds is also set (see below) — that takes priority.
    segment_count: int | None = Field(
        default=None,
        description="Number of parallel segments to shard the video into. "
        "Not yet decided for production (8 vs 12 pending Phase 3 latency "
        "measurements) — pass explicitly for tests/dev.",
    )
    # Alternative to segment_count: say how long each segment should be
    # instead of how many there should be. api.main.submit_job derives
    # segment_count from this via
    # orchestrator.splitter.compute_segment_count_from_duration(), which
    # needs total_duration_seconds resolved first (explicit or probed) — takes
    # priority over segment_count when both are set.
    segment_duration_seconds: float | None = Field(
        default=None,
        description="Desired duration per segment, in seconds. If set, "
        "segment_count is auto-derived from this and the video's total "
        "duration instead of being read directly.",
    )


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_video_url: str
    status: JobStatus
    created_at: datetime
    total_segments: int
    sla_target_seconds: int
    completed_at: datetime | None = None


class HighlightResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    span_id: uuid.UUID
    start_ts: float
    end_ts: float
    transcript_excerpt: str | None
    final_score: float
    justification: str | None
    clip_url: str | None = None
    rich_data: dict | None = None


class JobResultsResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    results: list[HighlightResultOut]


class UploadResponse(BaseModel):
    """Returned by POST /uploads — source_video_url is a plain local
    filesystem path (not a file:// URI), matching what JobCreateRequest.
    source_video_url expects: ffmpeg's -i flag takes a bare path directly."""

    source_video_url: str
    total_duration_seconds: float


# ---------------------------------------------------------------------------
# Inter-stage payloads
# ---------------------------------------------------------------------------


class SegmentTaskPayload(BaseModel):
    """Input to workers.segment_worker — one per parallel chord member."""

    job_id: uuid.UUID
    segment_id: uuid.UUID
    sequence_index: int
    source_video_url: str
    start_ts: float
    end_ts: float
    overlap_start: float
    overlap_end: float


class ShotBoundary(BaseModel):
    ts: float
    keyframe_ref: str | None = None


class TranscriptSegment(BaseModel):
    start_ts: float
    end_ts: float
    text: str
    speaker_label: str | None = None


class SegmentWorkerOutput(BaseModel):
    """Raw Stage-2 output for one segment: shots, transcript, embeddings,
    motion/audio features. Consumed by span_builder (Stage 3)."""

    segment_id: uuid.UUID
    sequence_index: int
    status: SegmentStatus
    shot_boundaries: list[ShotBoundary] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    # keyframe_ref -> embedding vector
    visual_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    # keyframe_ref -> [{face_id, bbox: [x0,y0,x1,y1] normalized, confidence}]
    # — see workers/segment_worker/adapters/face_detector.py. Foundation for
    # character identification: cross-segment face clustering + name-binding
    # happens later, in the reducer — this is just raw per-keyframe detections.
    face_detections: dict[str, list[dict]] = Field(default_factory=dict)
    # face_id -> embedding vector (from ClipVisualEmbedder.embed_faces —
    # reuses the same CLIP model as visual_embeddings, not a second model).
    face_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    # keyframe_ref -> [{label, bbox: [x0,y0,x1,y1] normalized, confidence}]
    # — see workers/segment_worker/adapters/object_detector.py. First step
    # of the commerce/object-identification roadmap in CLAUDE.md: detection
    # only, no catalog matching/ranking yet.
    shoppable_objects: dict[str, list[dict]] = Field(default_factory=dict)
    motion_features: dict[str, float] = Field(default_factory=dict)
    audio_features: dict[str, float] = Field(default_factory=dict)
    # Compact per-chunk memory (characters, events, objects, locations, open/
    # resolved story threads) produced by a lightweight TEXT-ONLY Gemma4 call
    # over this segment's transcript — see workers/segment_worker/adapters/
    # memory_extractor.py. Only populated when config.Settings.
    # use_global_memory_pipeline is True; None otherwise (including for every
    # existing test fixture/consumer, which never set this field). Consumed
    # by workers.global_selector.tasks.select_and_score_job, which merges
    # every segment's local_memory into one global movie memory before its
    # single job-wide selection call.
    local_memory: dict | None = None
    error: str | None = None


class CandidateSpanPayload(BaseModel):
    """Stage 3 (span_builder) output / Stage 4 (scorer) input."""

    segment_id: uuid.UUID
    start_ts: float
    end_ts: float
    transcript_excerpt: str | None = None
    feature_vector: dict = Field(default_factory=dict)
    touches_boundary: bool = False
    # Threaded through from orchestrator.pipeline (which already has it from
    # the Job row) via build_spans' source_video_url kwarg, the same way
    # overlap_start/overlap_end are bound onto that same task signature.
    # Needed by Gemma4Scorer to crop/save this span's actual video+audio clip
    # and attach real keyframe images/audio to the multimodal scoring call —
    # the aggregate feature_vector alone isn't enough for a genuinely
    # multimodal LLM judge. Optional (defaults None) so existing tests/
    # fixtures that don't set it keep working; Gemma4Scorer falls back to
    # text-only scoring if it's missing.
    source_video_url: str | None = None


class ScoredSpanPayload(CandidateSpanPayload):
    """Stage 4 (scorer) output / Stage 5 (reducer) input."""

    raw_score: float
    justification: str | None = None
    llm_model_version: str
    # URL of this span's actual cropped video+audio clip, saved via
    # storage.object_storage by Gemma4Scorer (see workers/scorer/adapters.py)
    # — None for MockScorer or if clip extraction/save failed for this span.
    clip_url: str | None = None
    # Full rich moment-analysis JSON (moment_title, dialogue_by_character,
    # characters_present, sound_design, cinematography, ad_placement,
    # risk_hints, inversion_score, etc.) — see Gemma4Scorer._SCORING_INSTRUCTION
    # for the exact shape. Stored as a flexible dict (not typed fields) since
    # it's deeply nested and the schema is still evolving — mirrors how
    # feature_vector is already stored.
    rich_data: dict | None = None


class SegmentPipelineResult(BaseModel):
    """What one full segment chain (segment_worker -> span_builder -> scorer)
    contributes to the chord callback (Stage 5 reducer input)."""

    segment_id: uuid.UUID
    sequence_index: int
    status: SegmentStatus
    scored_spans: list[ScoredSpanPayload] = Field(default_factory=list)
    error: str | None = None


class ReducedSpan(ScoredSpanPayload):
    normalized_score: float


class ReducerOutput(BaseModel):
    job_id: uuid.UUID
    spans: list[ReducedSpan]
    degraded_segment_ids: list[uuid.UUID] = Field(default_factory=list)
    dropped_duplicate_count: int = 0


class RankedHighlight(BaseModel):
    rank: int
    span: ReducedSpan
    final_score: float
