from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _sample_inquiry(inquiry_id=1, parent_id=None):
    return SimpleNamespace(
        id=inquiry_id,
        parent_id=parent_id,
        question="What is the meaning of life?",
        summary="A summary about meaning.",
        buddhism="Buddhism perspective",
        western_philosophy="Western philosophy perspective",
        psychology=None,
        perspectives={
            "buddhism": "Buddhism perspective",
            "western_philosophy": "Western philosophy perspective",
        },
        similarities="Both seek purpose",
        differences="One focuses on detachment, one on reason",
        references=["Dhammapada", "Nicomachean Ethics"],
        rag_sources=[],
        language="en",
        source="gemini",
        model="gemini-2.5-flash",
        answer_type="generated",
        topics=[],
        manual_fields=[],
        ai_source=None,
        manual_system_prompt=None,
        manual_sections=[],
        audio_filename=None,
        audio_mime_type=None,
        audio_voice=None,
        audio_model=None,
        audio_created_at=None,
        created_at=datetime.now(timezone.utc),
    )


def test_update_inquiry_endpoint_success(monkeypatch):
    mock_inquiry = _sample_inquiry(inquiry_id=1)

    monkeypatch.setattr("app.main.get_inquiry", lambda db, iid: mock_inquiry if iid == 1 else None)

    def fake_update(db, inquiry, *, summary, perspectives, similarities, differences, references):
        inquiry.summary = summary
        inquiry.perspectives = perspectives
        inquiry.similarities = similarities
        inquiry.differences = differences
        inquiry.references = references
        return inquiry

    monkeypatch.setattr("app.main.update_inquiry_answer", fake_update)

    update_payload = {
        "summary": "Updated summary text",
        "perspectives": {"buddhism": "Updated Buddhism view"},
        "similarities": "Updated similarities",
        "differences": "Updated differences",
        "references": ["Updated source"],
    }

    res = client.put("/inquiries/1", json=update_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == 1
    assert data["parent_id"] is None
    assert data["summary"] == "Updated summary text"
    assert data["perspectives"] == {"buddhism": "Updated Buddhism view"}
    assert data["references"] == ["Updated source"]


def test_update_inquiry_endpoint_not_found(monkeypatch):
    monkeypatch.setattr("app.main.get_inquiry", lambda db, iid: None)
    res = client.put("/inquiries/999", json={"summary": "abc"})
    assert res.status_code == 404


def test_fork_inquiry_endpoint_success(monkeypatch):
    original_inquiry = _sample_inquiry(inquiry_id=1, parent_id=None)

    monkeypatch.setattr("app.main.get_inquiry", lambda db, iid: original_inquiry if iid == 1 else None)

    def fake_fork(db, original, *, summary, perspectives, similarities, differences, references):
        forked = _sample_inquiry(inquiry_id=2, parent_id=original.id)
        forked.summary = summary
        forked.perspectives = perspectives
        forked.similarities = similarities
        forked.differences = differences
        forked.references = references
        return forked

    monkeypatch.setattr("app.main.fork_inquiry", fake_fork)

    fork_payload = {
        "summary": "Forked new summary",
        "perspectives": {"psychology": "Psychology insight"},
        "similarities": "Forked similarities",
        "differences": "Forked differences",
        "references": ["Carl Jung"],
    }

    res = client.post("/inquiries/1/fork", json=fork_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == 2
    assert data["parent_id"] == 1
    assert data["question"] == original_inquiry.question
    assert data["summary"] == "Forked new summary"
    assert data["perspectives"] == {"psychology": "Psychology insight"}
    assert data["references"] == ["Carl Jung"]


def test_fork_inquiry_endpoint_not_found(monkeypatch):
    monkeypatch.setattr("app.main.get_inquiry", lambda db, iid: None)
    res = client.post("/inquiries/999/fork", json={"summary": "abc"})
    assert res.status_code == 404

