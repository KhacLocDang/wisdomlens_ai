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
        "multiple perspectives.\n\n"
        "Rules:\n"
        "- Provide structured informational perspectives, NOT personal advice or therapy.\n"
        "- Be thoughtful, accessible, and concise (2-4 sentences per section).\n"
        "- Cite real sources in references where possible (texts, thinkers, research areas).\n"
        "- Return ONLY valid JSON with these keys:\n"
        "  summary, perspectives, similarities, differences, references\n"
        "- perspectives must be a JSON object containing the perspective keys as strings and answers as strings.\n"
        "- references must be a JSON array of strings.\n"
        "- Answer entirely in English."
    ),
    "vi": (
        "Bạn là WisdomLens AI. Hãy trả lời câu hỏi cuộc sống của người dùng "
        "bằng cách tổng hợp nhiều góc nhìn.\n\n"
        "Quy tắc:\n"
        "- Cung cấp góc nhìn có cấu trúc, KHÔNG đưa lời khuyên cá nhân hay trị liệu.\n"
        "- Trình bày rõ ràng, dễ hiểu, ngắn gọn (2-4 câu mỗi phần).\n"
        "- Trích dẫn nguồn thật nếu có thể (kinh sách, nhà tư tưởng, lĩnh vực nghiên cứu).\n"
        "- Trả về CHỈ JSON hợp lệ với các key:\n"
        "  summary, perspectives, similarities, differences, references\n"
        "- perspectives phải là một đối tượng JSON chứa các khóa của góc nhìn dưới dạng chuỗi và nội dung trả lời dưới dạng chuỗi.\n"
        "- references phải là mảng JSON các chuỗi.\n"
        "- Trả lời toàn bộ bằng tiếng Việt."
    ),
}

RAG_INSTRUCTIONS = (
    "RAG mode:\n"
    "- Use the retrieved context as the primary source of evidence.\n"
    "- If the context is missing or conflicts, say so clearly.\n"
    "- Do not invent facts that are not supported by the retrieved context.\n"
    "- If no relevant chunks were retrieved, state that explicitly and answer cautiously using general WisdomLens knowledge."
)

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
        "christianity": (
            "From a Christian perspective (placeholder): failure and suffering can be "
            "understood as part of a redemptive journey, calling for humility and faith."
        ),
        "eastern_philosophy": (
            "From Eastern philosophy (placeholder): thinkers like Confucius and the Tao "
            "tradition see failure as a teacher that reveals the path of virtue and balance."
        ),
        "natural_science": (
            "From natural science (placeholder): fear of failure is rooted in evolutionary "
            "threat-detection mechanisms that protect survival and social standing."
        ),
        "similarities": (
            "All lenses (placeholder) connect fear of failure to how "
            "humans relate to uncertainty, identity, and the need for meaning."
        ),
        "differences": (
            "They differ (placeholder) in emphasis: Buddhism on attachment, "
            "Western philosophy on ethics and meaning, psychology on cognition and behavior, "
            "Christianity on redemption and faith, Eastern philosophy on virtue and balance, "
            "natural science on evolutionary adaptation."
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
            "Phật giáo, Triết học phương Tây, Tâm lý học, "
            "Thiên Chúa giáo, Triết học phương Đông, và Khoa học tự nhiên."
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
        "christianity": (
            "Từ góc nhìn Thiên Chúa giáo (tạm): thất bại và khổ đau có thể được hiểu "
            "là một phần trong hành trình cứu chuộc, mời gọi sự khiêm nhường và đức tin."
        ),
        "eastern_philosophy": (
            "Từ góc nhìn Triết học phương Đông (tạm): các nhà tư tưởng như Khổng Tử và "
            "truyền thống Đạo gia xem thất bại là người thầy dẫn dắt ta trên con đường đức hạnh và cân bằng."
        ),
        "natural_science": (
            "Từ góc nhìn Khoa học tự nhiên (tạm): nỗi sợ thất bại bắt nguồn từ các cơ chế "
            "phát hiện mối nguy tiến hóa giúp bảo vệ sự sống còn và vị thế xã hội."
        ),
        "similarities": (
            "Tất cả các góc nhìn (tạm) đều liên kết nỗi sợ thất bại với cách "
            "con người đối mặt sự bất định, bản sắc, và nhu cầu tìm ý nghĩa."
        ),
        "differences": (
            "Chúng khác nhau (tạm) ở trọng tâm: Phật giáo về chấp thủ, "
            "Triết học phương Tây về đạo đức và ý nghĩa, Tâm lý học về nhận thức và hành vi, "
            "Thiên Chúa giáo về sự cứu chuộc và đức tin, Triết học phương Đông về đức hạnh và cân bằng, "
            "Khoa học tự nhiên về sự thích nghi tiến hóa."
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


def generate_fake_answer(
    question: str,
    language: str = "vi",
    perspectives: list[str] | None = None,
) -> dict:
    """Return a static structured answer for the MVP skeleton."""
    fake = FAKE_ANSWERS.get(language, FAKE_ANSWERS["vi"]).copy()

    _all_perspectives = [
        "buddhism", "western_philosophy", "psychology",
        "christianity", "eastern_philosophy", "natural_science",
    ]

    if perspectives is None:
        perspectives = _all_perspectives[:]
    else:
        perspectives = [p.lower() for p in perspectives]

    perspectives_dict = {}
    for p in _all_perspectives:
        if p in perspectives:
            perspectives_dict[p] = fake.get(p, "")

    for p in _all_perspectives:
        fake.pop(p, None)

    fake["perspectives"] = perspectives_dict

    if len(perspectives) <= 1:
        fake["similarities"] = ""
        fake["differences"] = ""

    return {"question": question, **fake}


def generate_gemini_answer(
    question: str,
    language: str = "vi",
    model: str | None = None,
    rag_context: dict | None = None,
    perspectives: list[str] | None = None,
) -> dict:
    """Call Gemini and return a structured answer matching AskResponse."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    model_id = resolve_model(model)
    prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["vi"])
    contents = f"Question: {question}"

    if perspectives is None:
        perspectives = ["buddhism", "western_philosophy", "psychology", "christianity", "eastern_philosophy", "natural_science"]
    else:
        perspectives = [p.lower() for p in perspectives]

    # Map perspective IDs to human readable names for prompt instruction
    perspective_names = {
        "en": {
            "buddhism": "Buddhism",
            "western_philosophy": "Western philosophy",
            "psychology": "Psychology",
            "christianity": "Christianity",
            "eastern_philosophy": "Eastern philosophy",
            "natural_science": "Natural science",
        },
        "vi": {
            "buddhism": "Phật giáo",
            "western_philosophy": "Triết học phương Tây",
            "psychology": "Tâm lý học",
            "christianity": "Thiên Chúa giáo",
            "eastern_philosophy": "Triết học phương Đông",
            "natural_science": "Khoa học tự nhiên",
        }
    }
    lang_names = perspective_names.get(language, perspective_names["vi"])
    selected_names = [lang_names.get(p, p) for p in perspectives]

    if language == "en":
        perspective_list_str = ", ".join(selected_names[:-1]) + " and " + selected_names[-1] if len(selected_names) > 1 else selected_names[0]
        perspective_instruction = (
            f"\n\nActive Perspectives:\n"
            f"You MUST only analyze the question and populate the 'perspectives' JSON object for these keys: {perspectives}.\n"
            f"Do NOT include any other keys in the 'perspectives' JSON object.\n"
            f"If only one perspective is selected, set similarities and differences to empty strings \"\"."
        )
    else:
        perspective_list_str = ", ".join(selected_names[:-1]) + " và " + selected_names[-1] if len(selected_names) > 1 else selected_names[0]
        perspective_instruction = (
            f"\n\nGóc nhìn hoạt động:\n"
            f"Bạn BẮT BUỘC chỉ được phân tích câu hỏi và điền thông tin vào đối tượng JSON 'perspectives' cho các khóa sau: {perspectives}.\n"
            f"KHÔNG được bao gồm bất kỳ khóa nào khác trong đối tượng JSON 'perspectives'.\n"
            f"Nếu chỉ có một góc nhìn được chọn, hãy đặt similarities và differences thành chuỗi rỗng \"\"."
        )
    prompt = f"{prompt}\n{perspective_instruction}"

    rag_sources = []
    if rag_context is not None:
        prompt = f"{prompt}\n\n{RAG_INSTRUCTIONS}"
        rag_sources = rag_context.get("sources") or []
        retrieved_context = (rag_context.get("context") or "").strip()
        if retrieved_context:
            contents = f"Retrieved context:\n{retrieved_context}\n\nUser question: {question}"
        else:
            contents = (
                "Retrieved context: (none)\n\n"
                "No relevant document chunks were found for this question.\n\n"
                f"User question: {question}"
            )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    # Clean up possible trailing commas that break strict JSON parsing
    import re, json
    def _clean_json(raw: str) -> str:
        # Remove any commas that appear just before a closing brace or bracket
        # This handles trailing commas in objects and arrays.
        return re.sub(r",\s*(?=[}\]])", "", raw)
    cleaned = _clean_json(response.text)
    fields = GeminiWisdomFields.model_validate_json(cleaned)
    return AskResponse(question=question, rag_sources=rag_sources, **fields.model_dump()).model_dump()
