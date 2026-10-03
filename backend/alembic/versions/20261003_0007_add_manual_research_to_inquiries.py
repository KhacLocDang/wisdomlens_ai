"""add manual research fields to inquiries

Revision ID: 0c43be4d0b39
Revises: 0c43be4d0b38
Create Date: 2026-10-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0c43be4d0b39"
down_revision: Union[str, None] = "0c43be4d0b38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inquiries",
        sa.Column("answer_type", sa.String(length=30), server_default="generated", nullable=False),
    )
    op.add_column(
        "inquiries",
        sa.Column("manual_fields", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
    )
    op.add_column("inquiries", sa.Column("ai_source", sa.String(length=100), nullable=True))
    op.add_column("inquiries", sa.Column("manual_system_prompt", sa.Text(), nullable=True))
    op.add_column(
        "inquiries",
        sa.Column("manual_sections", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("inquiries", "manual_sections")
    op.drop_column("inquiries", "manual_system_prompt")
    op.drop_column("inquiries", "ai_source")
    op.drop_column("inquiries", "manual_fields")
    op.drop_column("inquiries", "answer_type")