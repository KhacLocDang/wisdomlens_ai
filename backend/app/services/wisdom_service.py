import time

from google import genai
from google.genai import types

from app.config import (
    FALLBACK_GEMINI_MODELS,
    get_gemini_api_key,
    get_gemini_model,
    normalize_model_id,
)
from app.schemas import AskResponse, GeminiWisdomFields

# Exclude non-text / specialized model name fragments.
_EXCLUDE_FRAGMENTS = (
    "embedding",
    "tts",
    "image",
    "imagen",
    "live",
    "veo",
    "robotics",
    "computer-use",
    "deep-research",
    "aquavision",
    "nano-banana",
    "native-audio",
    "audio-preview",
    "audio-latest",
)

_models_cache: list[dict] | None = None
_models_cache_at: float = 0.0
_MODELS_CACHE_TTL_SECONDS = 600  # 10 minutes

SYSTEM_PROMPTS = {
    "en": (
        "You are WisdomLens AI. Answer the user's life question by synthesizing "
        "three perspectives: Buddhism, Western philosophy, and psychology.\n\n"
        "Rules:\n"
        "- Provide structured informational perspectives, NOT personal advice or therapy.\n"
        "- Be thoughtful, accessible, and concise (2-4 sentences per section).\n"
        "- Cite real sources in references where possible (texts, thinkers, research areas).\n"
        "- Return ONLY valid JSON with these keys:\n"
        "  summary, buddhism, western_philosophy, psychology, similarities, differences, references\n"
        "- references must be a JSON array of strings.\n"
        "- Answer entirely in English."
    ),
    "vi": (
        "Bạn là WisdomLens AI. Hãy trả lời câu hỏi cuộc sống của người dùng "
        "bằng cách tổng hợp ba góc nhìn: Phật giáo, Triết học phương Tây, và Tâm lý học.\n\n"
        "Quy tắc:\n"
        "- Cung cấp góc nhìn có cấu trúc, KHÔNG đưa lời khuyên cá nhân hay trị liệu.\n"
        "- Trình bày rõ ràng, dễ hiểu, ngắn gọn (2-4 câu mỗi phần).\n"
        "- Trích dẫn nguồn thật nếu có thể (kinh sách, nhà tư tưởng, lĩnh vực nghiên cứu).\n"
        "- Trả về CHỈ JSON hợp lệ với các key:\n"
        "  summary, buddhism, western_philosophy, psychology, similarities, differences, references\n"
        "- references phải là mảng JSON các chuỗi.\n"
        "- Trả lời toàn bộ bằng tiếng Việt."
    ),
}

FAKE_ANSWERS = {
    "en": {
        "summary": (
            "This is a placeholder summary. "
            "WisdomLens AI will synthesize perspectives from "
            "Buddhism, Western philosophy, and psychology."
        ),
        "buddhism": (
            "From a Buddhist perspective (placeholder): suffering often arises "
            "from attachment to outcomes and fear of losing a desired self-image."
        ),
        "western_philosophy": (
            "From Western philosophy (placeholder): thinkers from Stoicism to "
            "existentialism discuss failure as part of human finitude and authentic choice."
        ),
        "psychology": (
            "From psychology (placeholder): fear of failure links to threat "
            "responses, perfectionism, and social evaluation concerns."
        ),
        "similarities": (
            "All three lenses (placeholder) connect fear of failure to how "
            "humans relate to uncertainty, identity, and the need for meaning."
        ),
        "differences": (
            "They differ (placeholder) in emphasis: Buddhism on attachment, "
            "Western philosophy on ethics and meaning, psychology on cognition and behavior."
        ),
        "references": [
            "Placeholder — Dhammapada (Buddhist teachings)",
            "Placeholder — Epictetus, Enchiridion (Stoic perspective)",
            "Placeholder — APA overview on fear of failure (psychology)",
        ],
    },
    "vi": {
        "summary": (
            "Đây là bản tóm tắt tạm. "
            "WisdomLens AI sẽ tổng hợp góc nhìn từ "
            "Phật giáo, Triết học phương Tây, và Tâm lý học."
        ),
        "buddhism": (
            "Từ góc nhìn Phật giáo (tạm): khổ đau thường phát sinh "
            "từ sự chấp thủ vào kết quả và nỗi sợ mất đi hình ảnh bản thân."
        ),
        "western_philosophy": (
            "Từ góc nhìn Triết học phương Tây (tạm): các nhà tư tưởng từ Khắc kỷ "
            "đến Hiện sinh xem thất bại là phần tự nhiên của sự hữu hạn con người."
        ),
        "psychology": (
            "Từ góc nhìn Tâm lý học (tạm): nỗi sợ thất bại liên quan đến "
            "phản ứng phòng vệ, chủ nghĩa hoàn hảo, và áp lực đánh giá xã hội."
        ),
        "similarities": (
            "Cả ba góc nhìn (tạm) đều liên kết nỗi sợ thất bại với cách "
            "con người đối mặt sự bất định, bản sắc, và nhu cầu tìm ý nghĩa."
        ),
        "differences": (
            "Chúng khác nhau (tạm) ở trọng tâm: Phật giáo về chấp thủ, "
            "Triết học phương Tây về đạo đức và ý nghĩa, Tâm lý học về nhận thức và hành vi."
        ),
        "references": [
            "Tạm — Kinh Pháp Cú (giáo lý Phật giáo)",
            "Tạm — Epictetus, Enchiridion (góc nhìn Khắc kỷ)",
            "Tạm — APA tổng quan về nỗi sợ thất bại (tâm lý học)",
        ],
    },
}


def _is_text_gemini_model(model_id: str, supported_actions: list | None = None) -> bool:
    lower = model_id.lower()
    if "gemini" not in lower:
        return False
    if any(fragment in lower for fragment in _EXCLUDE_FRAGMENTS):
        return False
    if supported_actions:
        actions = {str(a).lower() for a in supported_actions}
        if actions and "generatecontent" not in actions and "generate_content" not in actions:
            if not any("generate" in a and "content" in a for a in actions):
                return False
    return True


def _fallback_model_list() -> list[dict]:
    return [
        {"id": model_id, "display_name": model_id}
        for model_id in FALLBACK_GEMINI_MODELS
    ]


def list_gemini_models() -> list[dict]:
    """List Gemini text models via API, with short cache and fallback."""
    global _models_cache, _models_cache_at

    now = time.time()
    if _models_cache is not None and (now - _models_cache_at) < _MODELS_CACHE_TTL_SECONDS:
        return _models_cache

    api_key = get_gemini_api_key()
    if not api_key:
        return _fallback_model_list()

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

            if not _is_text_gemini_model(model_id, list(supported) if supported else None):
                continue

            display_name = getattr(item, "display_name", None) or model_id
            models.append({"id": model_id, "display_name": display_name})
            seen.add(model_id)

        models.sort(key=lambda m: m["id"])
        if not models:
            models = _fallback_model_list()

        _models_cache = models
        _models_cache_at = now
        return models
    except Exception:
        return _fallback_model_list()


def resolve_model(requested: str | None) -> str:
    """Pick requested model if available, else default from env."""
    default = get_gemini_model()
    if not requested:
        return default

    model_id = normalize_model_id(requested)
    available_ids = {m["id"] for m in list_gemini_models()}
    if model_id in available_ids:
        return model_id
    if _is_text_gemini_model(model_id):
        return model_id
    raise ValueError(f"Unsupported model: {requested}")


def generate_fake_answer(question: str, language: str = "vi") -> dict:
    """Return a static structured answer for the MVP skeleton."""
    fake = FAKE_ANSWERS.get(language, FAKE_ANSWERS["vi"])
    return {"question": question, **fake}


def generate_gemini_answer(
    question: str,
    language: str = "vi",
    model: str | None = None,
) -> dict:
    """Call Gemini and return a structured answer matching AskResponse."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    model_id = resolve_model(model)
    prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["vi"])

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=GeminiWisdomFields,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")

    fields = GeminiWisdomFields.model_validate_json(response.text)
    return AskResponse(question=question, **fields.model_dump()).model_dump()
