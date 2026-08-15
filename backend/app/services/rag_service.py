from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.config import get_rag_min_score
from app.repositories.document_repository import get_document
from app.rag.retriever import retrieve_similar_chunks


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def build_rag_context(db: Session, query: str, limit: int = 5) -> dict[str, Any]:
    min_score = get_rag_min_score()
    chunks = [
        chunk
        for chunk in retrieve_similar_chunks(db, query=query, limit=limit)
        if float(getattr(chunk, "score", 0.0)) >= min_score
    ]
    sources: list[dict[str, Any]] = []
    context_parts: list[str] = []

    for rank, chunk in enumerate(chunks, start=1):
        metadata = _normalize_metadata(chunk.metadata_json)
        document = get_document(db, chunk.document_id)

        title = metadata.get("source_title") or (document.title if document else None)
        category = metadata.get("category") or (document.category if document else None)
        author = metadata.get("author") or (document.author if document else None)
        source_url = metadata.get("source_url") or (document.source_url if document else None)
        chunk_index = metadata.get("chunk_index")

        sources.append(
            {
                "rank": rank,
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "score": float(getattr(chunk, "score", 0.0)),
                "title": title,
                "category": category,
                "author": author,
                "source_url": source_url,
                "chunk_index": chunk_index,
                "embedding_model": chunk.embedding_model,
                "metadata": metadata,
            }
        )

        context_parts.append(
            "\n".join(
                [
                    f"[Source {rank}]",
                    f"Document: {title or 'Unknown'}",
                    f"Category: {category or 'Unknown'}",
                    f"Chunk: {chunk_index if chunk_index is not None else 'Unknown'}",
                    f"Score: {getattr(chunk, 'score', 0.0):.4f}",
                    "Content:",
                    chunk.content.strip(),
                ]
            )
        )

    return {
        "query": query,
        "sources": sources,
        "context": "\n\n".join(context_parts).strip(),
        "has_sources": bool(sources),
        "min_score": min_score,
    }
