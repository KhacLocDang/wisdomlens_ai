from fastapi.testclient import TestClient
from types import SimpleNamespace
import sys

from app.main import app
from app.services.providers.claude import ClaudeProvider

client = TestClient(app)


def test_list_providers():
    response = client.get("/providers")
    assert response.status_code == 200
    providers = response.json()
    assert "gemini" in providers
    assert "claude" in providers


def test_list_models_with_provider_param():
    # Gemini models
    res_gemini = client.get("/models?provider=gemini")
    assert res_gemini.status_code == 200
    assert len(res_gemini.json()) > 0
    assert any("gemini" in m["id"].lower() for m in res_gemini.json())

    # Claude models
    res_claude = client.get("/models?provider=claude")
    assert res_claude.status_code == 200
    claude_models = res_claude.json()
    assert len(claude_models) > 0
    assert any("claude" in m["id"].lower() for m in claude_models)

    # Invalid provider
    res_invalid = client.get("/models?provider=unknown_provider")
    assert res_invalid.status_code == 400


def test_ask_with_claude_provider(monkeypatch):
    captured = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)

    def fake_generate_provider_answer(
        question,
        provider=None,
        language="vi",
        model=None,
        rag_context=None,
        perspectives=None,
        **kwargs,
    ):
        captured["provider"] = provider
        captured["model"] = model
        return {
            "question": question,
            "summary": "Claude summary",
            "perspectives": {
                "buddhism": "Claude buddhism perspective",
            },
            "similarities": "",
            "differences": "",
            "references": ["Claude reference"],
            "rag_sources": [],
        }

    saved = {}

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["source"] = source
        saved["model"] = model
        return None

    monkeypatch.setattr("app.main.generate_provider_answer", fake_generate_provider_answer)
    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post(
        "/ask",
        json={
            "question": "What is the nature of change?",
            "language": "en",
            "provider": "claude",
            "model": "claude-3-5-haiku-20241022",
            "perspectives": ["buddhism"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Claude summary"
    assert captured["provider"] == "claude"
    assert captured["model"] == "claude-3-5-haiku-20241022"
    assert saved["source"] == "claude"
    assert saved["model"] == "claude-3-5-haiku-20241022"


def test_claude_preserves_unparseable_response(monkeypatch):
    raw_text = '{"summary": "Partial Claude answer'
    calls = {"count": 0}

    class FakeMessages:
        def create(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=raw_text)],
                stop_reason="max_tokens",
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("app.config.get_claude_api_key", lambda: "test-key")
    monkeypatch.setattr("app.config.get_claude_model", lambda: "claude-test")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))

    answer = ClaudeProvider().generate_answer("Test question")

    assert calls["count"] == 1
    assert answer["summary"] == raw_text
    assert "token limit" in answer["generation_warning"]


def test_ask_saves_unparseable_claude_response(monkeypatch):
    raw_text = '{"summary": "Partial Claude answer'
    saved = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr(
        "app.main.generate_provider_answer",
        lambda **kwargs: {
            "question": kwargs["question"],
            "summary": raw_text,
            "generation_warning": "Claude output is incomplete.",
            "perspectives": {},
            "similarities": "",
            "differences": "",
            "references": [],
            "rag_sources": [],
        },
    )

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["summary"] = answer["summary"]
        saved["source"] = source

    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post(
        "/ask",
        json={"question": "Test question", "provider": "claude"},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == raw_text
    assert response.json()["generation_warning"] == "Claude output is incomplete."
    assert saved["summary"].startswith("[Response could not be parsed; raw model output follows]")
    assert raw_text in saved["summary"]
    assert saved["source"] == "claude"
