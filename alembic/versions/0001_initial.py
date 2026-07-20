"""initial schema: jobs, segments, candidate_spans, scored_spans, highlight_results

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-11

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table( 
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_video_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("total_segments", sa.Integer(), nullable=False),
        sa.Column("sla_target_seconds", sa.Integer(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("start_ts", sa.Float(), nullable=False),
        sa.Column("end_ts", sa.Float(), nullable=False),
        sa.Column("overlap_start", sa.Float(), nullable=False),
        sa.Column("overlap_end", sa.Float(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("worker_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_segments_job_id", "segments", ["job_id"])

    op.create_table(
        "candidate_spans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("segment_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("segments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_ts", sa.Float(), nullable=False),
        sa.Column("end_ts", sa.Float(), nullable=False),
        sa.Column("transcript_excerpt", sa.Text(), nullable=True),
        sa.Column("feature_vector", postgresql.JSONB(), nullable=True),
        sa.Column("touches_boundary", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_candidate_spans_segment_id", "candidate_spans", ["segment_id"])

    op.create_table(
        "scored_spans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "candidate_span_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("candidate_spans.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("segment_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("segments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_ts", sa.Float(), nullable=False),
        sa.Column("end_ts", sa.Float(), nullable=False),
        sa.Column("transcript_excerpt", sa.Text(), nullable=True),
        sa.Column("feature_vector", postgresql.JSONB(), nullable=True),
        sa.Column("touches_boundary", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("raw_score", sa.Float(), nullable=False),
        sa.Column("normalized_score", sa.Float(), nullable=True),
        sa.Column("justification", sa.Text(), nullable=True),
        sa.Column("llm_model_version", sa.String(128), nullable=False),
    )
    op.create_index("ix_scored_spans_segment_id", "scored_spans", ["segment_id"])

    op.create_table(
        "highlight_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("span_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("scored_spans.id", ondelete="CASCADE"), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
    )
    op.create_index("ix_highlight_results_job_id", "highlight_results", ["job_id"])


def downgrade() -> None:
    op.drop_table("highlight_results")
    op.drop_table("scored_spans")
    op.drop_table("candidate_spans")
    op.drop_table("segments")
    op.drop_table("jobs")
