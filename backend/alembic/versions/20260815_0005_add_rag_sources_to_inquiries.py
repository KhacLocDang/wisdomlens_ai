"""add rag_sources column to inquiries

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-15
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inquiries",
        sa.Column("rag_sources", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.alter_column("inquiries", "rag_sources", server_default=None)


def downgrade() -> None:
    op.drop_column("inquiries", "rag_sources")
