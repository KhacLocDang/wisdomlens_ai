from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_list_tts_models():
    response = client.get("/tts/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) > 0
    assert any("gemini" in m["id"].lower() for m in models)


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
