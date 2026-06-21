"""create inquiries table

Revision ID: 0001
Revises:
Create Date: 2026-06-17

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "inquiries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("buddhism", sa.Text(), nullable=False),
        sa.Column("western_philosophy", sa.Text(), nullable=False),
        sa.Column("psychology", sa.Text(), nullable=False),
        sa.Column("similarities", sa.Text(), nullable=False),
        sa.Column("differences", sa.Text(), nullable=False),
        sa.Column("references", JSONB(), nullable=False, server_default="[]"),
        sa.Column("language", sa.String(5), nullable=False, server_default="vi"),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("inquiries")
