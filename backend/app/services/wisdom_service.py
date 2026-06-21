from google import genai
from google.genai import types

from app.config import get_gemini_api_key, get_gemini_model
from app.schemas import AskResponse, GeminiWisdomFields

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


def generate_fake_answer(question: str, language: str = "vi") -> dict:
    """Return a static structured answer for the MVP skeleton."""
    fake = FAKE_ANSWERS.get(language, FAKE_ANSWERS["vi"])
    return {"question": question, **fake}


def generate_gemini_answer(question: str, language: str = "vi") -> dict:
    """Call Gemini and return a structured answer matching AskResponse."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["vi"])

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_gemini_model(),
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
