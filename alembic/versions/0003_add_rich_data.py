"""add rich_data to scored_spans

Revision ID: 0003_add_rich_data
Revises: 0002_add_clip_url
Create Date: 2026-07-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_add_rich_data"
down_revision: Union[str, None] = "0002_add_clip_url"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("scored_spans", sa.Column("rich_data", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("scored_spans", "rich_data")
