from sqlalchemy import or_
from sqlalchemy.orm import Session

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
) -> Inquiry:
    perspectives = answer.get("perspectives") or {}
    inquiry = Inquiry(
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
    )
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry


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
