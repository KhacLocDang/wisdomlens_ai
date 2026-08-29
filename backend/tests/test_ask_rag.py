from fastapi.testclient import TestClient

from app.main import app
from app.services.rag_service import build_rag_context


client = TestClient(app)


def test_ask_without_rag_uses_original_flow(monkeypatch):
    captured = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr("app.main.use_rag", lambda: False)
    monkeypatch.setattr("app.main.resolve_model", lambda requested: "gemini-2.5-flash")

    def fake_generate_gemini_answer(question, language, model=None, rag_context=None, **kwargs):
        captured["rag_context"] = rag_context
        return {
            "question": question,
            "summary": "summary",
            "perspectives": {
                "buddhism": "buddhism",
                "western_philosophy": "western philosophy",
                "psychology": "psychology",
            },
            "similarities": "similarities",
            "differences": "differences",
            "references": ["ref"],
            "rag_sources": [],
        }

    saved = {}

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["source"] = source
        saved["rag_sources"] = rag_sources
        return None

    monkeypatch.setattr("app.main.generate_gemini_answer", fake_generate_gemini_answer)
    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post("/ask", json={"question": "Why are humans afraid of failure?", "language": "en"})

    assert response.status_code == 200
    body = response.json()
    assert body["rag_sources"] == []
    assert captured["rag_context"] is None
    assert saved["source"] == "gemini"
    assert saved["rag_sources"] == []


def test_ask_with_rag_passes_retrieved_context(monkeypatch):
    captured = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr("app.main.resolve_model", lambda requested: "gemini-2.5-flash")
    monkeypatch.setattr("app.main.use_rag", lambda: True)

    def fake_build_rag_context(db, query, limit=5):
        return {
            "query": query,
            "has_sources": True,
            "context": (
                "[Source 1]\n"
                "Document: Sample Sutra\n"
                "Category: Buddhism\n"
                "Chunk: 2\n"
                "Score: 0.9910\n"
                "Content:\n"
                "Attachment causes suffering."
            ),
            "sources": [
                {
                    "rank": 1,
                    "chunk_id": 10,
                    "document_id": 3,
                    "score": 0.991,
                    "title": "Sample Sutra",
                    "category": "Buddhism",
                    "author": "Teacher",
                    "source_url": "https://example.com/sutra",
                    "chunk_index": 2,
                    "embedding_model": "gemini-embedding-2-preview",
                    "metadata": {"source_title": "Sample Sutra", "chunk_index": 2},
                }
            ],
        }

    def fake_generate_gemini_answer(question, language, model=None, rag_context=None, **kwargs):
        captured["rag_context"] = rag_context
        return {
            "question": question,
            "summary": "summary",
            "perspectives": {
                "buddhism": "buddhism",
                "western_philosophy": "western philosophy",
                "psychology": "psychology",
            },
            "similarities": "similarities",
            "differences": "differences",
            "references": ["ref"],
            "rag_sources": rag_context["sources"],
        }

    saved = {}

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["rag_sources"] = rag_sources
        return None

    monkeypatch.setattr("app.main.build_rag_context", fake_build_rag_context)
    monkeypatch.setattr("app.main.generate_gemini_answer", fake_generate_gemini_answer)
    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post(
        "/ask",
        json={"question": "Why do humans suffer?", "language": "en", "use_rag": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rag_sources"][0]["rank"] == 1
    assert body["rag_sources"][0]["chunk_id"] == 10
    assert captured["rag_context"]["has_sources"] is True
    assert saved["rag_sources"][0]["document_id"] == 3


def test_ask_with_rag_handles_no_relevant_chunks(monkeypatch):
    captured = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr("app.main.use_rag", lambda: False)
    monkeypatch.setattr("app.main.resolve_model", lambda requested: "gemini-2.5-flash")
    monkeypatch.setattr(
        "app.main.build_rag_context",
        lambda db, query, limit=5: {"query": query, "has_sources": False, "context": "", "sources": []},
    )

    def fake_generate_gemini_answer(question, language, model=None, rag_context=None, **kwargs):
        captured["rag_context"] = rag_context
        return {
            "question": question,
            "summary": "summary",
            "perspectives": {
                "buddhism": "buddhism",
                "western_philosophy": "western philosophy",
                "psychology": "psychology",
            },
            "similarities": "similarities",
            "differences": "differences",
            "references": [],
            "rag_sources": rag_context["sources"],
        }

    monkeypatch.setattr("app.main.generate_gemini_answer", fake_generate_gemini_answer)
    monkeypatch.setattr("app.main.save_inquiry", lambda *args, **kwargs: None)

    response = client.post(
        "/ask",
        json={"question": "An unrelated question", "language": "en", "use_rag": True},
    )

    assert response.status_code == 200
    assert response.json()["rag_sources"] == []
    assert captured["rag_context"]["has_sources"] is False


def test_rag_context_excludes_chunks_below_minimum_score(monkeypatch):
    class Chunk:
        def __init__(self, chunk_id, score):
            self.id = chunk_id
            self.document_id = 1
            self.score = score
            self.content = "Relevant document content"
            self.metadata_json = {"chunk_index": chunk_id}
            self.embedding_model = "gemini-embedding-2-preview"

    monkeypatch.setattr(
        "app.services.rag_service.retrieve_similar_chunks",
        lambda db, query, limit: [Chunk(1, 0.82), Chunk(2, 0.12)],
    )
    monkeypatch.setattr("app.services.rag_service.get_rag_min_score", lambda: 0.35)
    monkeypatch.setattr("app.services.rag_service.get_document", lambda db, document_id: None)

    context = build_rag_context(db=object(), query="question")

    assert context["has_sources"] is True
    assert context["min_score"] == 0.35
    assert [source["chunk_id"] for source in context["sources"]] == [1]


def test_inquiry_detail_returns_saved_rag_sources(monkeypatch):
    class SavedInquiry:
        id = 1
        question = "Why do humans suffer?"
        summary = "summary"
        buddhism = "buddhism"
        western_philosophy = "western philosophy"
        psychology = "psychology"
        perspectives = {
            "buddhism": "buddhism",
            "western_philosophy": "western philosophy",
            "psychology": "psychology",
        }
        similarities = "similarities"
        differences = "differences"
        references = ["ref"]
        rag_sources = [
            {
                "rank": 1,
                "chunk_id": 10,
                "document_id": 3,
                "score": 0.991,
                "title": "Sample Sutra",
            }
        ]
        language = "en"
        source = "gemini"
        model = "gemini-2.5-flash"
        created_at = "2026-08-15T00:00:00+00:00"

    monkeypatch.setattr("app.main.get_inquiry", lambda db, inquiry_id: SavedInquiry())

    response = client.get("/inquiries/1")

    assert response.status_code == 200
    assert response.json()["rag_sources"][0]["rank"] == 1
    assert response.json()["perspectives"]["buddhism"] == "buddhism"


def test_ask_with_selected_perspectives(monkeypatch):
    captured = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr("app.main.resolve_model", lambda requested: "gemini-2.5-flash")

    def fake_generate_gemini_answer(question, language, model=None, rag_context=None, perspectives=None, **kwargs):
        captured["perspectives"] = perspectives
        return {
            "question": question,
            "summary": "summary",
            "perspectives": {
                "buddhism": "buddhism",
                "psychology": "psychology",
            },
            "similarities": "similarities",
            "differences": "differences",
            "references": ["ref"],
            "rag_sources": [],
        }

    monkeypatch.setattr("app.main.generate_gemini_answer", fake_generate_gemini_answer)
    monkeypatch.setattr("app.main.save_inquiry", lambda *args, **kwargs: None)

    # Valid perspectives selection
    response = client.post(
        "/ask",
        json={
            "question": "Selected perspectives test",
            "language": "en",
            "perspectives": ["buddhism", "psychology"],
        },
    )

    assert response.status_code == 200
    assert captured["perspectives"] == ["buddhism", "psychology"]
    body = response.json()
    assert "buddhism" in body["perspectives"]
    assert "psychology" in body["perspectives"]
    assert "western_philosophy" not in body["perspectives"]

    # Invalid perspective validation test
    invalid_response = client.post(
        "/ask",
        json={
            "question": "Invalid perspective",
            "language": "en",
            "perspectives": ["invalid_name"],
        },
    )
    assert invalid_response.status_code == 400
    assert "Unsupported perspective" in invalid_response.json()["detail"]
