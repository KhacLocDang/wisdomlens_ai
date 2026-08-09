from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def create_document_chunk(
    db: Session,
    *,
    document_id: int,
    content: str,
    metadata: dict | None = None,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
    commit: bool = True,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        content=content,
        metadata_json=metadata or {},
        embedding_json=embedding,
        embedding_model=embedding_model,
    )
    db.add(chunk)
    if commit:
        db.commit()
        db.refresh(chunk)
    else:
        db.flush()
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


def get_document_chunks_with_embeddings(db: Session) -> list[DocumentChunk]:
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.embedding_json.isnot(None))
        .all()
    )


def get_document_chunks_without_embeddings(
    db: Session,
    limit: int | None = None,
    document_id: int | None = None,
) -> list[DocumentChunk]:
    query = db.query(DocumentChunk).filter(DocumentChunk.embedding_json.is_(None))
    if document_id is not None:
        query = query.filter(DocumentChunk.document_id == document_id)
    query = query.order_by(DocumentChunk.id)
    return query.limit(limit).all() if limit is not None else query.all()


def update_document_chunk_embedding(
    db: Session,
    chunk_id: int,
    embedding: list[float],
    embedding_model: str,
) -> DocumentChunk:
    chunk = db.get(DocumentChunk, chunk_id)
    if chunk is None:
        raise ValueError(f"Chunk {chunk_id} not found")
    chunk.embedding_json = embedding
    chunk.embedding_model = embedding_model
    db.commit()
    db.refresh(chunk)
    return chunk
