from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.services.manual_research import get_manual_research_config

client = TestClient(app)


def _request_payload() -> dict:
    return {
        "question": "How do people understand suffering?",
        "answer": "Claude's complete answer with multiple perspectives.",
        "language": "en",
        "ai_source": "Claude Web",
        "model": "Claude Sonnet",
        "topics": ["buddhism", "psychology"],
    }


def test_manual_research_config_returns_system_prompt_only():
    response = client.get("/manual-research/config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["system_prompt"]
    assert {topic["key"] for topic in payload["topics"]} == {
        "buddhism",
        "psychology",
        "western_philosophy",
        "christianity",
        "eastern_philosophy",
        "natural_science",
        "other",
    }


def test_create_manual_research_saves_question_and_complete_answer(monkeypatch):
    saved = {}

    def fake_save_inquiry(
        db,
        answer,
        *,
        language,
        source,
        model=None,
        rag_sources=None,
        answer_type="generated",
        topics=None,
    ):
        saved.update(
            answer=answer,
            language=language,
            source=source,
            model=model,
            rag_sources=rag_sources,
            answer_type=answer_type,
            topics=topics,
        )
        return SimpleNamespace(
            id=404,
            question=answer["question"],
            summary=answer["summary"],
            perspectives={},
            similarities="",
            differences="",
            references=[],
            rag_sources=[],
            language=language,
            source=source,
            model=model,
            answer_type=answer_type,
            topics=topics or [],
            audio_filename=None,
            audio_mime_type=None,
            audio_voice=None,
            audio_model=None,
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post("/manual-research", json=_request_payload())

    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["id"] == 404
    assert answer["answer_type"] == "manual_answer"
    assert answer["question"] == "How do people understand suffering?"
    assert answer["summary"] == "Claude's complete answer with multiple perspectives."
    assert answer["source"] == "Claude Web"
    assert answer["model"] == "Claude Sonnet"
    assert answer["topics"] == ["buddhism", "psychology"]
    assert saved["answer"]["summary"] == answer["summary"]
    assert saved["answer_type"] == "manual_answer"
    assert saved["topics"] == ["buddhism", "psychology"]


def test_create_manual_research_allows_no_topics(monkeypatch):
    saved = {}

    def fake_save_inquiry(db, answer, **kwargs):
        saved.update(kwargs)
        return SimpleNamespace(
            id=405,
            question=answer["question"],
            summary=answer["summary"],
            perspectives={},
            similarities="",
            differences="",
            references=[],
            rag_sources=[],
            language=kwargs["language"],
            source=kwargs["source"],
            model=kwargs["model"],
            answer_type=kwargs["answer_type"],
            topics=kwargs["topics"],
            audio_filename=None,
            audio_mime_type=None,
            audio_voice=None,
            audio_model=None,
            created_at=datetime.now(timezone.utc),
        )

    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)
    payload = _request_payload()
    payload["topics"] = []

    response = client.post("/manual-research", json=payload)

    assert response.status_code == 200
    assert response.json()["topics"] == []
    assert saved["topics"] == []


def test_create_manual_research_rejects_blank_answer_after_trim(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_inquiry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not save")),
    )

    payload = _request_payload()
    payload["answer"] = "  "
    response = client.post("/manual-research", json=payload)

    assert response.status_code == 400
    assert "answer" in response.json()["detail"].lower()


def test_create_manual_research_rejects_unknown_topic(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_inquiry",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not save")),
    )
    payload = _request_payload()
    payload["topics"] = ["made_up_topic"]

    response = client.post("/manual-research", json=payload)

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()
