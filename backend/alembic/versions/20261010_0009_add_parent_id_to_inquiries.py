"""add parent_id to inquiries

Revision ID: 0c43be4d0b3b
Revises: 0c43be4d0b3a
Create Date: 2026-10-10 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0c43be4d0b3b"
down_revision: Union[str, None] = "0c43be4d0b3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "inquiries",
        sa.Column(
            "parent_id",
            sa.Integer(),
            sa.ForeignKey("inquiries.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("inquiries", "parent_id")

