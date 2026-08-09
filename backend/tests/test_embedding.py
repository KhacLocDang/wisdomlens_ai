import pytest

from app.rag.embedding import EmbeddingError, generate_embedding


def test_generate_embedding_missing_api_key(monkeypatch):
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="Embedding API key is not configured"):
        generate_embedding("test text")
