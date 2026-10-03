from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB

from app.database import Base


class Inquiry(Base):
    __tablename__ = "inquiries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    buddhism = Column(Text, nullable=True)
    western_philosophy = Column(Text, nullable=True)
    psychology = Column(Text, nullable=True)
    perspectives = Column(JSONB, nullable=False, server_default="{}")
    similarities = Column(Text, nullable=False)
    differences = Column(Text, nullable=False)
    references = Column(JSONB, nullable=False, default=list)
    rag_sources = Column(JSONB, nullable=False, default=list)
    language = Column(String(5), nullable=False, server_default="vi")
    source = Column(String(20), nullable=False)
    model = Column(String(100), nullable=True)
    answer_type = Column(String(30), nullable=False, server_default="generated")
    topics = Column(JSONB, nullable=False, server_default="[]")
    manual_fields = Column(JSONB, nullable=False, server_default="[]")
    ai_source = Column(String(100), nullable=True)
    manual_system_prompt = Column(Text, nullable=True)
    manual_sections = Column(JSONB, nullable=False, server_default="[]")
    audio_filename = Column(String(255), nullable=True)
    audio_mime_type = Column(String(50), nullable=True)
    audio_voice = Column(String(100), nullable=True)
    audio_model = Column(String(100), nullable=True)
    audio_created_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
