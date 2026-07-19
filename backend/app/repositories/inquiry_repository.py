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
) -> Inquiry:
    inquiry = Inquiry(
        question=answer["question"],
        summary=answer["summary"],
        buddhism=answer["buddhism"],
        western_philosophy=answer["western_philosophy"],
        psychology=answer["psychology"],
        similarities=answer["similarities"],
        differences=answer["differences"],
        references=answer.get("references") or [],
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
