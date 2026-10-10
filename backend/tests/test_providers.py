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


def test_list_models_with_provider_param(monkeypatch):
    from app.services.providers.claude import CLAUDE_MODEL_CATALOG, _with_output_limits
    monkeypatch.setattr(ClaudeProvider, "list_models", lambda self: _with_output_limits(CLAUDE_MODEL_CATALOG))

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
    sonnet_37 = next(m for m in claude_models if "claude-3-7-sonnet" in m["id"])
    haiku_35 = next(m for m in claude_models if "claude-3-5-haiku" in m["id"])
    assert sonnet_37["max_output_tokens"] == 64000
    assert haiku_35["max_output_tokens"] == 8192

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
        captured["sentences_per_section"] = kwargs.get("sentences_per_section")
        captured["claude_max_tokens"] = kwargs.get("claude_max_tokens")
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
            "sentences_per_section": 5,
            "claude_max_tokens": 6000,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Claude summary"
    assert captured["provider"] == "claude"
    assert captured["model"] == "claude-3-5-haiku-20241022"
    assert captured["sentences_per_section"] == 5
    assert captured["claude_max_tokens"] == 6000
    assert saved["source"] == "claude"
    assert saved["model"] == "claude-3-5-haiku-20241022"


def test_ask_rejects_claude_tokens_above_selected_model_limit(monkeypatch):
    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr(
        "app.main.generate_provider_answer",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("provider must not be called")),
    )

    response = client.post(
        "/ask",
        json={
            "question": "Test question",
            "provider": " CLAUDE ",
            "model": "claude-3-5-sonnet-20241022",
            "claude_max_tokens": 8193,
        },
    )

    assert response.status_code == 400
    assert "cannot exceed 8192" in response.json()["detail"]


def test_claude_preserves_unparseable_response(monkeypatch):
    raw_text = '{"summary": "Partial Claude answer'
    calls = {"count": 0, "max_tokens": None}

    class FakeMessages:
        def create(self, **kwargs):
            calls["count"] += 1
            calls["max_tokens"] = kwargs["max_tokens"]
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=raw_text)],
                stop_reason="max_tokens",
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("app.config.get_claude_api_key", lambda: "test-key")
    monkeypatch.setattr("app.config.get_claude_model", lambda: "claude-test")
    monkeypatch.setattr("app.config.get_claude_max_tokens", lambda: 8192)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))

    answer = ClaudeProvider().generate_answer("Test question")

    assert calls["count"] == 1
    assert calls["max_tokens"] == 8192
    assert answer["summary"] == raw_text
    assert "token limit" in answer["generation_warning"]


def test_claude_uses_requested_tokens_and_sentence_count(monkeypatch):
    captured = {}
    valid_json = '{"summary":"Answer","perspectives":{},"similarities":"","differences":"","references":[]}'

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=valid_json)],
                stop_reason="end_turn",
            )

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("app.config.get_claude_api_key", lambda: "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))

    answer = ClaudeProvider().generate_answer(
        "Test question",
        model="claude-3-5-sonnet-20241022",
        sentences_per_section=5,
        claude_max_tokens=7000,
    )

    assert answer["summary"] == "Answer"
    assert captured["max_tokens"] == 7000
    assert "mỗi phần văn bản gồm 5 câu" in captured["system"]


def test_claude_rejects_token_override_above_model_cap(monkeypatch):
    monkeypatch.setattr("app.config.get_claude_api_key", lambda: "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kwargs: None))

    try:
        ClaudeProvider().generate_answer(
            "Test question",
            model="claude-3-5-haiku-20241022",
            claude_max_tokens=8193,
        )
    except ValueError as exc:
        assert "output limit of 8192" in str(exc)
    else:
        raise AssertionError("Expected a token limit validation error")


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
