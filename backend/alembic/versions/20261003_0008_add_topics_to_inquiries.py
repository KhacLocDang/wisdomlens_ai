"""add topics to inquiries

Revision ID: 0c43be4d0b3a
Revises: 0c43be4d0b39
Create Date: 2026-10-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0c43be4d0b3a"
down_revision: Union[str, None] = "0c43be4d0b39"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inquiries",
        sa.Column(
            "topics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("inquiries", "topics")