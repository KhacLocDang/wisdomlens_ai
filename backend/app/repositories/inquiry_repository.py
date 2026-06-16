from sqlalchemy.orm import Session

from app.models.inquiry import Inquiry


def save_inquiry(
    db: Session,
    answer: dict,
    *,
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
        source=source,
        model=model,
    )
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    return inquiry


def list_inquiries(db: Session, limit: int = 20) -> list[Inquiry]:
    return (
        db.query(Inquiry)
        .order_by(Inquiry.created_at.desc())
        .limit(limit)
        .all()
    )


def get_inquiry(db: Session, inquiry_id: int) -> Inquiry | None:
    return db.query(Inquiry).filter(Inquiry.id == inquiry_id).first()
