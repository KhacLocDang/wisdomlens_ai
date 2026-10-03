from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_save_inquiry_audio_persists_metadata(monkeypatch, tmp_path):
    inquiry = SimpleNamespace(
        id=42,
        question="What is suffering?",
        summary="Summary",
        perspectives={"buddhism": "..."},
        similarities="",
        differences="",
        references=[],
        rag_sources=[],
        language="vi",
        source="gemini",
        model="gemini-2.5-flash",
        audio_filename=None,
        audio_mime_type=None,
        audio_voice=None,
        audio_model=None,
    )

    monkeypatch.setattr("app.main.get_inquiry", lambda db, inquiry_id: inquiry)
    monkeypatch.setattr("app.main.synthesize_speech", lambda **kwargs: (b"fake-audio", "audio/wav"))
    monkeypatch.setattr("app.main.get_audio_storage_root", lambda: tmp_path)

    def fake_update_inquiry_audio(db, target, *, audio_filename, audio_mime_type, audio_voice, audio_model):
        target.audio_filename = audio_filename
        target.audio_mime_type = audio_mime_type
        target.audio_voice = audio_voice
        target.audio_model = audio_model
        return target

    monkeypatch.setattr("app.main.update_inquiry_audio", fake_update_inquiry_audio)

    response = client.post(
        "/inquiries/42/audio",
        json={
            "text": "Cuộc sống là một chuỗi những trải nghiệm.",
            "voice": "Aoede",
            "model": "gemini-2.5-flash-preview-tts",
            "language": "vi",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_available"] is True
    assert payload["audio_filename"].endswith(".wav")
    assert payload["audio_voice"] == "Aoede"

    saved_file = tmp_path / "inquiry_42.wav"
    assert saved_file.exists()
    assert saved_file.read_bytes() == b"fake-audio"


def test_delete_inquiry_audio_removes_file(monkeypatch, tmp_path):
    inquiry = SimpleNamespace(
        id=99,
        question="Question",
        summary="Summary",
        perspectives={},
        similarities="",
        differences="",
        references=[],
        rag_sources=[],
        language="vi",
        source="gemini",
        model="gemini-2.5-flash",
        audio_filename="inquiry_99.wav",
        audio_mime_type="audio/wav",
        audio_voice="Aoede",
        audio_model="gemini-2.5-flash-preview-tts",
    )

    audio_file = tmp_path / "inquiry_99.wav"
    audio_file.write_bytes(b"existing-audio")

    monkeypatch.setattr("app.main.get_inquiry", lambda db, inquiry_id: inquiry)
    monkeypatch.setattr("app.main.get_audio_storage_root", lambda: tmp_path)

    def fake_clear_inquiry_audio(db, target):
        target.audio_filename = None
        target.audio_mime_type = None
        target.audio_voice = None
        target.audio_model = None
        return target

    monkeypatch.setattr("app.main.clear_inquiry_audio", fake_clear_inquiry_audio)

    response = client.delete("/inquiries/99/audio")

    assert response.status_code == 200
    assert response.json()["audio_available"] is False
    assert not audio_file.exists()
