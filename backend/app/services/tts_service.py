"""Text-to-Speech (TTS) service using Google Gemini API."""

from __future__ import annotations

import base64
import io
import logging
import time
import wave
from typing import Any

from google import genai
from google.genai import types

from app.config import (
    FALLBACK_GEMINI_TTS_MODELS,
    GEMINI_TTS_VOICES,
    get_gemini_api_key,
    get_gemini_tts_model,
    get_gemini_tts_voice,
    normalize_model_id,
)

logger = logging.getLogger(__name__)

_tts_models_cache: list[dict] | None = None
_tts_models_cache_at: float = 0.0
_TTS_MODELS_CACHE_TTL_SECONDS = 600  # 10 minutes

_TTS_EXCLUDE_FRAGMENTS = (
    "embedding",
    "image",
    "imagen",
    "live",
    "veo",
    "robotics",
    "computer-use",
    "deep-research",
    "aquavision",
    "nano-banana",
    "transcribe",
    "customtools",
)


def _is_tts_gemini_model(model_id: str, supported_actions: list | None = None) -> bool:
    lower = model_id.lower()
    if "gemini" not in lower:
        return False
    if any(fragment in lower for fragment in _TTS_EXCLUDE_FRAGMENTS):
        return False
    if supported_actions:
        actions = {str(a).lower() for a in supported_actions}
        if actions and "generatecontent" not in actions and "generate_content" not in actions:
            if not any("generate" in a and "content" in a for a in actions):
                return False
    # Dedicated TTS & native-audio speech synthesis models in Google Gemini
    if "tts" in lower or "native-audio" in lower or model_id in FALLBACK_GEMINI_TTS_MODELS:
        return True
    return False


def _fallback_tts_model_list() -> list[dict]:
    return [
        {"id": model_id, "display_name": f"Gemini ({model_id})"}
        for model_id in FALLBACK_GEMINI_TTS_MODELS
    ]


def list_tts_models() -> list[dict]:
    """List Gemini TTS models via Google AI API, with short cache and fallback."""
    global _tts_models_cache, _tts_models_cache_at

    now = time.time()
    if _tts_models_cache is not None and (now - _tts_models_cache_at) < _TTS_MODELS_CACHE_TTL_SECONDS:
        return _tts_models_cache

    api_key = get_gemini_api_key()
    if not api_key:
        return _fallback_tts_model_list()

    try:
        client = genai.Client(api_key=api_key)
        models: list[dict] = []
        seen: set[str] = set()

        for item in client.models.list():
            raw_name = getattr(item, "name", None) or ""
            model_id = normalize_model_id(raw_name)
            if not model_id or model_id in seen:
                continue

            supported = getattr(item, "supported_actions", None) or getattr(
                item, "supported_generation_methods", None
            )
            if isinstance(supported, str):
                supported = [supported]

            if not _is_tts_gemini_model(model_id, list(supported) if supported else None):
                continue

            display_name = getattr(item, "display_name", None) or f"Gemini ({model_id})"
            models.append({"id": model_id, "display_name": display_name})
            seen.add(model_id)

        models.sort(key=lambda m: m["id"])
        if not models:
            models = _fallback_tts_model_list()

        _tts_models_cache = models
        _tts_models_cache_at = now
        return models
    except Exception as exc:
        logger.warning("Failed to list Gemini TTS models from API: %s; using fallback", exc)
        return _fallback_tts_model_list()


def list_tts_voices() -> list[dict]:
    """Return list of supported Gemini TTS voices."""
    return GEMINI_TTS_VOICES


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    bit_depth: int = 16,
) -> bytes:
    """Wrap raw PCM audio bytes in a standard WAV container."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


def synthesize_speech(
    text: str,
    voice: str | None = None,
    model: str | None = None,
    language: str = "vi",
) -> tuple[bytes, str]:
    """Generate audio speech from text using Google Gemini API.

    Returns:
        tuple[bytes, str]: (audio_bytes, mime_type)
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Text content cannot be empty for speech synthesis")

    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured for Text-to-Speech")

    model_id = normalize_model_id(model or get_gemini_tts_model())
    voice_name = (voice or get_gemini_tts_voice()).strip()

    # Validate voice name against supported prebuilt voices
    supported_voices = {v["id"].lower(): v["id"] for v in GEMINI_TTS_VOICES}
    if voice_name.lower() in supported_voices:
        voice_name = supported_voices[voice_name.lower()]
    else:
        voice_name = "Aoede"

    if language == "vi":
        prompt = (
            "Hãy đọc to đoạn văn sau bằng tiếng Việt với giọng điệu tự nhiên, "
            "truyền cảm, rõ ràng và đúng ngữ điệu. KHÔNG thêm bất kỳ lời dẫn hay bình luận nào:\n\n"
            f"{clean_text}"
        )
    else:
        prompt = (
            "Please read the following text aloud clearly and naturally with "
            "expressive pacing. Do NOT include any introductory or concluding remarks:\n\n"
            f"{clean_text}"
        )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )
    except Exception as exc:
        logger.exception("Gemini TTS API request failed: %s", exc)
        raise RuntimeError(f"Gemini TTS request failed: {exc}") from exc

    audio_bytes: bytes | None = None
    mime_type: str = "audio/wav"

    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data is not None:
                        raw_data = getattr(inline_data, "data", None)
                        if isinstance(raw_data, str):
                            audio_bytes = base64.b64decode(raw_data)
                        elif isinstance(raw_data, bytes):
                            audio_bytes = raw_data

                        part_mime = getattr(inline_data, "mime_type", None) or ""
                        if part_mime:
                            mime_type = part_mime
                        break
            if audio_bytes:
                break

    if not audio_bytes:
        raise RuntimeError("Gemini TTS returned an empty audio response")

    # If format is raw PCM or L16, convert to standard playable WAV container
    if (
        "pcm" in mime_type.lower()
        or "l16" in mime_type.lower()
        or "raw" in mime_type.lower()
        or not audio_bytes.startswith(b"RIFF")
    ):
        if not audio_bytes.startswith(b"ID3") and not audio_bytes.startswith(b"\xff\xfb"):  # not MP3
            audio_bytes = pcm_to_wav(audio_bytes, sample_rate=24000)
            mime_type = "audio/wav"

    return audio_bytes, mime_type
