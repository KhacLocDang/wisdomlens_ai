from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.services import tts_service

client = TestClient(app)


def test_list_tts_models_uses_v1beta(monkeypatch):
    observed = {}

    class FakeModels:
        def list(self):
            return [
                SimpleNamespace(
                    name="models/gemini-3.8-flash-tts",
                    display_name="Gemini 3.8 Flash TTS",
                    supported_actions=["generateContent"],
                )
            ]

    def fake_client(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(tts_service, "get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(tts_service.genai, "Client", fake_client)
    monkeypatch.setattr(tts_service, "_tts_models_cache", None)
    monkeypatch.setattr(tts_service, "_tts_models_cache_at", 0.0)

    models = tts_service.list_tts_models()

    assert [model["id"] for model in models] == ["gemini-3.8-flash-tts"]
    assert observed["http_options"].api_version == "v1beta"


def test_synthesize_speech_uses_v1beta(monkeypatch):
    observed = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            observed["generate_content"] = kwargs
            return SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(
                                        data=b"RIFFtest-audio",
                                        mime_type="audio/wav",
                                    )
                                )
                            ]
                        )
                    )
                ]
            )

    def fake_client(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(tts_service, "get_gemini_api_key", lambda: "test-key")
    monkeypatch.setattr(tts_service.genai, "Client", fake_client)

    audio, mime_type = tts_service.synthesize_speech(
        "Test speech",
        model="gemini-2.5-flash-preview-tts",
    )

    assert audio == b"RIFFtest-audio"
    assert mime_type == "audio/wav"
    assert observed["http_options"].api_version == "v1beta"
    assert observed["generate_content"]["model"] == "gemini-2.5-flash-preview-tts"


def test_list_tts_voices():
    response = client.get("/tts/voices")
    assert response.status_code == 200
    voices = response.json()
    assert len(voices) > 0
    voice_ids = [v["id"] for v in voices]
    assert "Aoede" in voice_ids
    assert "Puck" in voice_ids


def test_tts_synthesize_validation_empty_text():
    response = client.post(
        "/tts/synthesize",
        json={"text": "   ", "language": "vi"},
    )
    assert response.status_code == 400


def test_tts_synthesize_success(monkeypatch):
    import io
    import wave

    # Create dummy 0.1s WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00" * 4800)
    dummy_wav_bytes = wav_buffer.getvalue()

    def fake_synthesize_speech(text, voice=None, model=None, language="vi"):
        assert len(text) > 0
        return dummy_wav_bytes, "audio/wav"

    monkeypatch.setattr("app.main.synthesize_speech", fake_synthesize_speech)

    response = client.post(
        "/tts/synthesize",
        json={
            "text": "Cuộc sống là một chuỗi những trải nghiệm và học hỏi.",
            "voice": "Aoede",
            "model": "gemini-2.0-flash",
            "language": "vi",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) == len(dummy_wav_bytes)
