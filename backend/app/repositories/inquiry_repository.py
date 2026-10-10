from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import get_audio_storage_root
from app.models.inquiry import Inquiry


def escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def save_inquiry(
    db: Session,
    answer: dict,
    *,
    language: str,
    source: str,
    model: str | None = None,
    rag_sources: list[dict] | None = None,
    answer_type: str = "generated",
    topics: list[str] | None = None,
    parent_id: int | None = None,
) -> Inquiry:
    perspectives = answer.get("perspectives") or {}
    inquiry = Inquiry(
        parent_id=parent_id,
        question=answer["question"],
        summary=answer["summary"],
        buddhism=perspectives.get("buddhism"),
        western_philosophy=perspectives.get("western_philosophy"),
        psychology=perspectives.get("psychology"),
        perspectives=perspectives,
        similarities=answer["similarities"],
        differences=answer["differences"],
        references=answer.get("references") or [],
        rag_sources=rag_sources if rag_sources is not None else answer.get("rag_sources") or [],
        language=language,
        source=source,
        model=model,
        answer_type=answer_type,
        topics=topics or [],
    )
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry


def update_inquiry_answer(
    db: Session,
    inquiry: Inquiry,
    *,
    summary: str,
    perspectives: dict[str, str],
    similarities: str,
    differences: str,
    references: list[str],
) -> Inquiry:
    inquiry.summary = summary
    inquiry.perspectives = perspectives
    inquiry.buddhism = perspectives.get("buddhism")
    inquiry.western_philosophy = perspectives.get("western_philosophy")
    inquiry.psychology = perspectives.get("psychology")
    inquiry.similarities = similarities
    inquiry.differences = differences
    inquiry.references = references
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry


def fork_inquiry(
    db: Session,
    original: Inquiry,
    *,
    summary: str,
    perspectives: dict[str, str],
    similarities: str,
    differences: str,
    references: list[str],
) -> Inquiry:
    forked = Inquiry(
        parent_id=original.id,
        question=original.question,
        summary=summary,
        buddhism=perspectives.get("buddhism"),
        western_philosophy=perspectives.get("western_philosophy"),
        psychology=perspectives.get("psychology"),
        perspectives=perspectives,
        similarities=similarities,
        differences=differences,
        references=references,
        rag_sources=original.rag_sources or [],
        language=original.language,
        source=original.source,
        model=original.model,
        answer_type=original.answer_type or "generated",
        topics=original.topics or [],
        manual_fields=original.manual_fields or [],
        ai_source=original.ai_source,
        manual_system_prompt=original.manual_system_prompt,
        manual_sections=original.manual_sections or [],
    )
    db.add(forked)
    db.commit()
    db.refresh(forked)
    return forked



def list_inquiries(db: Session, limit: int = 20, q: str | None = None) -> list[Inquiry]:
    query = db.query(Inquiry)
    if q is not None:
        q = q.strip()
        if len(q) >= 2:
            term = f"%{escape_like(q)}%"
            query = query.filter(
                or_(
                    Inquiry.question.ilike(term, escape="\\"),
                    Inquiry.summary.ilike(term, escape="\\"),
                )
            )
    return query.order_by(Inquiry.created_at.desc()).limit(limit).all()


def get_inquiry(db: Session, inquiry_id: int) -> Inquiry | None:
    return db.query(Inquiry).filter(Inquiry.id == inquiry_id).first()


def get_inquiry_audio_file_path(inquiry_id: int, *, storage_root=None):
    root = storage_root or get_audio_storage_root()
    return root / f"inquiry_{inquiry_id}.wav"


def update_inquiry_audio(
    db: Session,
    inquiry: Inquiry,
    *,
    audio_filename: str,
    audio_mime_type: str,
    audio_voice: str | None,
    audio_model: str | None,
) -> Inquiry:
    inquiry.audio_filename = audio_filename
    inquiry.audio_mime_type = audio_mime_type
    inquiry.audio_voice = audio_voice
    inquiry.audio_model = audio_model
    inquiry.audio_created_at = datetime.now(timezone.utc)
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry


def clear_inquiry_audio(db: Session, inquiry: Inquiry) -> Inquiry:
    inquiry.audio_filename = None
    inquiry.audio_mime_type = None
    inquiry.audio_voice = None
    inquiry.audio_model = None
    inquiry.audio_created_at = None
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry
