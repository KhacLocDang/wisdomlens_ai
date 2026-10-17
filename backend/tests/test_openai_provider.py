"""Unit tests for the OpenAI provider implementation."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.main import app
from app.services.providers import get_provider, list_models_for_provider, list_supported_providers
from app.services.providers.openai import OPENAI_MODEL_CATALOG, OpenAIProvider

client = TestClient(app)


def test_openai_provider_registry():
    assert "openai" in list_supported_providers()
    prov = get_provider("openai")
    assert isinstance(prov, OpenAIProvider)
    models = list_models_for_provider("openai")
    assert len(models) >= len(OPENAI_MODEL_CATALOG)


def test_openai_list_models_fallback(monkeypatch):
    monkeypatch.setattr("app.config.get_openai_api_key", lambda: None)
    prov = OpenAIProvider()
    models = prov.list_models()
    assert models == OPENAI_MODEL_CATALOG


def test_openai_list_models_dynamic(monkeypatch):
    class FakeModel:
        def __init__(self, model_id):
            self.id = model_id

    class FakeModelsApi:
        def list(self):
            return [
                FakeModel("gpt-4o-mini"),
                FakeModel("gpt-4o-2024-08-06"),
                FakeModel("o3-mini"),
                FakeModel("text-embedding-3-small"),  # Should be excluded
                FakeModel("dall-e-3"),                 # Should be excluded
            ]

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.models = FakeModelsApi()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "fake-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    models = prov.list_models()
    model_ids = [m["id"] for m in models]
    assert "gpt-4o-mini" in model_ids
    assert "gpt-4o-2024-08-06" in model_ids
    assert "o3-mini" in model_ids
    assert "text-embedding-3-small" not in model_ids
    assert "dall-e-3" not in model_ids


def test_openai_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr("app.config.get_openai_api_key", lambda: None)
    prov = OpenAIProvider()
    try:
        prov.generate_answer(question="Test question")
    except ValueError as exc:
        assert "OPENAI_API_KEY is not configured" in str(exc)
    else:
        raise AssertionError("Expected ValueError when OPENAI_API_KEY is not configured")


def test_openai_generate_answer_success(monkeypatch):
    valid_json = (
        '{"summary": "OpenAI summary", '
        '"perspectives": {"buddhism": "OpenAI buddhism perspective"}, '
        '"similarities": "OpenAI similarities", '
        '"differences": "OpenAI differences", '
        '"references": ["OpenAI ref 1", "OpenAI ref 2"]}'
    )

    captured_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=valid_json),
                    )
                ]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    answer = prov.generate_answer(
        question="Why do people fear change?",
        language="vi",
        model="gpt-4o-mini",
        perspectives=["buddhism"],
        conciseness="balanced",
    )

    assert answer["summary"] == "OpenAI summary"
    assert answer["perspectives"]["buddhism"] == "OpenAI buddhism perspective"
    assert answer["similarities"] == "OpenAI similarities"
    assert answer["differences"] == "OpenAI differences"
    assert answer["references"] == ["OpenAI ref 1", "OpenAI ref 2"]
    assert captured_kwargs["model"] == "gpt-4o-mini"
    assert captured_kwargs["response_format"] == {"type": "json_object"}
    assert captured_kwargs["temperature"] == 0.7


def test_openai_generate_answer_reasoning_model_no_temperature(monkeypatch):
    valid_json = '{"summary": "o3 summary", "perspectives": {}, "similarities": "", "differences": "", "references": []}'
    captured_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=valid_json))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    prov.generate_answer(question="Question", model="o3-mini")

    assert captured_kwargs["model"] == "o3-mini"
    assert "temperature" not in captured_kwargs


def test_openai_generate_answer_unparseable_json_preserves_raw_text(monkeypatch):
    raw_text = '{"summary": "Incomplete OpenAI answer'

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=raw_text))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    answer = prov.generate_answer(question="Test question", model="gpt-4o-mini")

    assert answer["summary"] == raw_text
    assert "could not be parsed" in answer["generation_warning"]


def test_openai_generate_answer_length_finish_reason(monkeypatch):
    raw_text = '{"summary": "Truncated answer'

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=raw_text))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    answer = prov.generate_answer(question="Test question", model="gpt-4o-mini")

    assert answer["summary"] == raw_text
    assert "reached its output token limit" in answer["generation_warning"]


def test_openai_empty_response_raises(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=""))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-openai-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAIClient))

    prov = OpenAIProvider()
    try:
        prov.generate_answer(question="Test question")
    except ValueError as exc:
        assert "OpenAI returned an empty response" in str(exc)
    else:
        raise AssertionError("Expected ValueError on empty OpenAI response")


def test_ask_endpoint_with_openai_provider(monkeypatch):
    captured = {}
    saved = {}

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
            "summary": "GPT summary",
            "perspectives": {"buddhism": "Buddhism perspective"},
            "similarities": "Same",
            "differences": "Diff",
            "references": ["Ref"],
            "rag_sources": [],
        }

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["source"] = source
        saved["model"] = model
        return None

    monkeypatch.setattr("app.main.generate_provider_answer", fake_generate_provider_answer)
    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post(
        "/ask",
        json={
            "question": "What is the meaning of mindfulness?",
            "language": "en",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "perspectives": ["buddhism"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "GPT summary"
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"
    assert saved["source"] == "openai"
    assert saved["model"] == "gpt-4o-mini"


def test_ask_with_openai_and_rag(monkeypatch):
    captured = {}
    saved = {}

    monkeypatch.setattr("app.main.use_fake_answers", lambda: False)
    monkeypatch.setattr(
        "app.main.build_rag_context",
        lambda db, query: {
            "query": query,
            "has_sources": True,
            "context": "Sample RAG context",
            "sources": [{"rank": 1, "chunk_id": 1, "document_id": 1, "score": 0.85}],
        },
    )

    def fake_generate_provider_answer(
        question,
        provider=None,
        language="vi",
        model=None,
        rag_context=None,
        perspectives=None,
        **kwargs,
    ):
        captured["rag_context"] = rag_context
        return {
            "question": question,
            "summary": "GPT RAG summary",
            "perspectives": {},
            "similarities": "",
            "differences": "",
            "references": [],
            "rag_sources": rag_context.get("sources") if rag_context else [],
        }

    def fake_save_inquiry(db, answer, *, language, source, model=None, rag_sources=None):
        saved["rag_sources"] = rag_sources
        saved["source"] = source

    monkeypatch.setattr("app.main.generate_provider_answer", fake_generate_provider_answer)
    monkeypatch.setattr("app.main.save_inquiry", fake_save_inquiry)

    response = client.post(
        "/ask",
        json={
            "question": "What does Buddhism say about impermanence?",
            "provider": "openai",
            "model": "gpt-4o",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "GPT RAG summary"
    assert len(data["rag_sources"]) == 1
    assert data["rag_sources"][0]["score"] == 0.85
    assert saved["source"] == "openai"
    assert len(saved["rag_sources"]) == 1


def test_openai_generate_answer_temperature_retry(monkeypatch):
    """When a model rejects custom temperature, the provider should
    automatically retry the request without the temperature parameter."""
    valid_json = (
        '{"summary": "sol summary", "perspectives": {}, '
        '"similarities": "", "differences": "", "references": []}'
    )

    call_log: list[dict] = []

    class FakeBadRequestError(Exception):
        """Stand-in for openai.BadRequestError."""
        def __init__(self, message, body):
            super().__init__(message)
            self.body = body

    class FakeCompletions:
        def create(self, **kwargs):
            call_log.append(dict(kwargs))
            if "temperature" in kwargs:
                raise FakeBadRequestError(
                    "temperature unsupported",
                    {"message": "Unsupported value: 'temperature' does not support 0.7 with this model."},
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=valid_json))]
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAIClient, BadRequestError=FakeBadRequestError),
    )

    prov = OpenAIProvider()
    answer = prov.generate_answer(question="Test", model="gpt-6.1-sol")

    # First call should have included temperature; second should not
    assert len(call_log) == 2
    assert "temperature" in call_log[0]
    assert "temperature" not in call_log[1]
    assert answer["summary"] == "sol summary"


def test_openai_generate_answer_non_temperature_error_propagates(monkeypatch):
    """BadRequestError that is NOT about temperature should not be caught."""

    class FakeBadRequestError(Exception):
        def __init__(self, message, body):
            super().__init__(message)
            self.body = body

    class FakeCompletions:
        def create(self, **kwargs):
            raise FakeBadRequestError(
                "invalid model",
                {"message": "The model 'gpt-nonexistent' does not exist."},
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr("app.config.get_openai_api_key", lambda: "test-key")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=FakeOpenAIClient, BadRequestError=FakeBadRequestError),
    )

    prov = OpenAIProvider()
    try:
        prov.generate_answer(question="Test", model="gpt-nonexistent")
    except FakeBadRequestError as exc:
        assert "invalid model" in str(exc)
    else:
        raise AssertionError("Expected FakeBadRequestError to propagate")
