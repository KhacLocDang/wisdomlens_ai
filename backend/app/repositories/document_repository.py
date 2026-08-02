from sqlalchemy.orm import Session

from app.models.document import Document


def create_document(db: Session, *, title: str, category: str, author: str | None = None, source_url: str | None = None, metadata: dict | None = None) -> Document:
    document = Document(
        title=title,
        category=category,
        author=author,
        source_url=source_url,
        metadata_json=metadata or {},
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def list_documents(db: Session, limit: int = 50) -> list[Document]:
    return db.query(Document).order_by(Document.created_at.desc()).limit(limit).all()


def get_document(db: Session, document_id: int) -> Document | None:
    return db.query(Document).filter(Document.id == document_id).first()
