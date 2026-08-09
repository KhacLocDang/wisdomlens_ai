from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_retrieve_endpoint_requires_query():
    response = client.get("/rag/retrieve")
    assert response.status_code == 422


def test_retrieve_endpoint_invalid_limit():
    response = client.get("/rag/retrieve", params={"q": "hello", "limit": 0})
    assert response.status_code == 422


def test_refresh_embeddings_endpoint(monkeypatch):
    class Chunk:
        def __init__(self, id, content):
            self.id = id
            self.content = content

    def fake_get_document_chunks_without_embeddings(db, limit=None, document_id=None):
        return [Chunk(1, "chunk one"), Chunk(2, "chunk two")]

    def fake_generate_embedding(content):
        return [0.1, 0.2, 0.3]

    refreshed = []

    def fake_update_document_chunk_embedding(db, chunk_id, embedding, embedding_model):
        refreshed.append((chunk_id, tuple(embedding), embedding_model))

    monkeypatch.setattr("app.main.get_document_chunks_without_embeddings", fake_get_document_chunks_without_embeddings)
    monkeypatch.setattr("app.main.generate_embedding", fake_generate_embedding)
    monkeypatch.setattr("app.main.update_document_chunk_embedding", fake_update_document_chunk_embedding)
    monkeypatch.setattr("app.main.get_embedding_model", lambda: "gemini-embedding-2-preview")

    response = client.post("/rag/embeddings/refresh")
    assert response.status_code == 200
    body = response.json()
    assert body["refreshed_count"] == 2
    assert body["failed_count"] == 0
    assert body["refreshed_chunk_ids"] == [1, 2]
    assert refreshed == [
        (1, (0.1, 0.2, 0.3), "gemini-embedding-2-preview"),
        (2, (0.1, 0.2, 0.3), "gemini-embedding-2-preview"),
    ]
