import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from models.db import Base
from models.enums import JobStatus, SegmentStatus


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_video_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        String(32), nullable=False, default=JobStatus.created
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Segment count for this job. Set explicitly at submission time (or by the
    # orchestrator's split step) — never silently defaulted to a hardcoded
    # global constant. See config.Settings.provisional_dev_segment_count.
    total_segments: Mapped[int] = mapped_column(nullable=False)
    sla_target_seconds: Mapped[int] = mapped_column(nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    segments: Mapped[list["Segment"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    highlight_results: Mapped[list["HighlightResult"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    sequence_index: Mapped[int] = mapped_column(nullable=False)

    # Global timestamps (seconds from the start of the full film) — NOT
    # segment-local. Stage 5 (reducer) dedupes across segments using these,
    # so segment-local timestamps would silently corrupt dedup logic.
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    overlap_start: Mapped[float] = mapped_column(Float, nullable=False)
    overlap_end: Mapped[float] = mapped_column(Float, nullable=False)

    status: Mapped[SegmentStatus] = mapped_column(
        String(32), nullable=False, default=SegmentStatus.pending
    )
    worker_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped["Job"] = relationship(back_populates="segments")
    candidate_spans: Mapped[list["CandidateSpan"]] = relationship(
        back_populates="segment", cascade="all, delete-orphan"
    )


class CandidateSpan(Base):
    __tablename__ = "candidate_spans"

    id: Mapped[uuid.UUID] = _uuid_pk()
    segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )

    # Global timestamps — see Segment note above.
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    transcript_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_vector: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # True if this span overlaps a segment boundary/overlap window — the
    # reducer needs this to know which spans are dedup candidates against a
    # neighboring segment's output.
    touches_boundary: Mapped[bool] = mapped_column(default=False, nullable=False)

    segment: Mapped["Segment"] = relationship(back_populates="candidate_spans")
    scored_span: Mapped["ScoredSpan | None"] = relationship(
        back_populates="candidate_span", uselist=False, cascade="all, delete-orphan"
    )


class ScoredSpan(Base):
    """Extends CandidateSpan (Stage 4 output). Denormalized copy of the
    candidate's fields plus scoring metadata, so downstream stages (reducer,
    ranker) can read a scored span without joining back to candidate_spans."""

    __tablename__ = "scored_spans"

    id: Mapped[uuid.UUID] = _uuid_pk()
    candidate_span_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("candidate_spans.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    segment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )

    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    transcript_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_vector: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    touches_boundary: Mapped[bool] = mapped_column(default=False, nullable=False)

    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    normalized_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    # URL of this span's actual cropped video+audio clip, saved via
    # storage.object_storage by Gemma4Scorer at scoring time (see
    # workers/scorer/adapters.py) — None for MockScorer (Step 1, no real
    # video available) or if clip extraction/save failed for this span.
    clip_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full rich moment-analysis JSON — see
    # models.schemas.ScoredSpanPayload.rich_data's docstring for the shape.
    rich_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    candidate_span: Mapped["CandidateSpan"] = relationship(back_populates="scored_span")
    highlight_result: Mapped["HighlightResult | None"] = relationship(
        back_populates="span", uselist=False
    )


class HighlightResult(Base):
    __tablename__ = "highlight_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    rank: Mapped[int] = mapped_column(nullable=False)
    span_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("scored_spans.id", ondelete="CASCADE"), nullable=False
    )
    final_score: Mapped[float] = mapped_column(Float, nullable=False)

    job: Mapped["Job"] = relationship(back_populates="highlight_results")
    span: Mapped["ScoredSpan"] = relationship(back_populates="highlight_result")
