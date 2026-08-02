from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def create_document_chunk(db: Session, *, document_id: int, content: str, metadata: dict | None = None) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        content=content,
        metadata_json=metadata or {},
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def list_document_chunks(db: Session, limit: int = 100) -> list[DocumentChunk]:
    return db.query(DocumentChunk).order_by(DocumentChunk.created_at.desc()).limit(limit).all()


def get_document_chunks_for_document(db: Session, document_id: int) -> list[DocumentChunk]:
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.id)
        .all()
    )
