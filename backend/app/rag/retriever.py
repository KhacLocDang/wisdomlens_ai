import math
from typing import Any

from sqlalchemy.orm import Session

from app.repositories.chunk_repository import get_document_chunks_with_embeddings
from app.repositories.document_repository import get_document


class RetrievalError(Exception):
    pass


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        raise RetrievalError("Vector lengths must match for cosine similarity.")
    denom = _vector_norm(a) * _vector_norm(b)
    if denom == 0:
        return 0.0
    return _dot_product(a, b) / denom


def retrieve_similar_chunks(db: Session, query: str, limit: int = 5) -> list[Any]:
    from app.rag.embedding import generate_embedding

    if not query or not query.strip():
        raise RetrievalError("Query text must be provided.")

    query_embedding = generate_embedding(query)
    chunks = get_document_chunks_with_embeddings(db)
    scored = []
    for chunk in chunks:
        chunk_embedding = chunk.embedding_json
        if not isinstance(chunk_embedding, list):
            continue
        try:
            score = _cosine_similarity(query_embedding, chunk_embedding)
        except RetrievalError:
            continue
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, chunk in scored[:limit]:
        chunk.score = score
        results.append(chunk)
    return results
