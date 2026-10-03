"""add audio columns to inquiries

Revision ID: 0c43be4d0b38
Revises: 274aa5d3654e
Create Date: 2026-10-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0c43be4d0b38"
down_revision: Union[str, None] = "274aa5d3654e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inquiries", sa.Column("audio_filename", sa.String(length=255), nullable=True))
    op.add_column("inquiries", sa.Column("audio_mime_type", sa.String(length=50), nullable=True))
    op.add_column("inquiries", sa.Column("audio_voice", sa.String(length=100), nullable=True))
    op.add_column("inquiries", sa.Column("audio_model", sa.String(length=100), nullable=True))
    op.add_column("inquiries", sa.Column("audio_created_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("inquiries", "audio_created_at")
    op.drop_column("inquiries", "audio_model")
    op.drop_column("inquiries", "audio_voice")
    op.drop_column("inquiries", "audio_mime_type")
    op.drop_column("inquiries", "audio_filename")
