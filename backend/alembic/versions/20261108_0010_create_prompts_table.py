"""create prompts table

Revision ID: 0c43be4d0b3c
Revises: 0c43be4d0b3b
Create Date: 2026-11-08 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0c43be4d0b3c"
down_revision: Union[str, None] = "0c43be4d0b3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("custom_prompt", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(op.f("ix_prompts_key"), "prompts", ["key"], unique=True)
    op.create_index(op.f("ix_prompts_id"), "prompts", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_prompts_id"), table_name="prompts")
    op.drop_index(op.f("ix_prompts_key"), table_name="prompts")
    op.drop_table("prompts")
