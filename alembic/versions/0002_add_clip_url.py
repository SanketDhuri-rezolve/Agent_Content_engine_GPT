"""add clip_url to scored_spans

Revision ID: 0002_add_clip_url
Revises: 0001_initial
Create Date: 2026-07-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_clip_url"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("scored_spans", sa.Column("clip_url", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("scored_spans", "clip_url")
