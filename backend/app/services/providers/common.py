import re
from typing import Any

PERSPECTIVE_NAMES = {
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
    },
}

ALL_DEFAULT_PERSPECTIVES = [
    "buddhism",
    "western_philosophy",
    "psychology",
    "christianity",
    "eastern_philosophy",
    "natural_science",
]

SYSTEM_PROMPTS = {
    "en": (
        "You are WisdomLens AI. Answer the user's life question by synthesizing "
        "multiple perspectives.\n\n"
        "Rules:\n"
        "- Provide structured informational perspectives, NOT personal advice or therapy.\n"
        "- Be thoughtful and accessible; keep each text section to {sentence_count} sentences.\n"
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
        "- Trình bày rõ ràng, dễ hiểu; mỗi phần văn bản gồm {sentence_count} câu.\n"
        "- Trích dẫn nguồn thật nếu có thể (kinh sách, nhà tư tưởng, lĩnh vực nghiên cứu).\n"
        "- Trả về CHỈ JSON hợp lệ với các key:\n"
        "  summary, perspectives, similarities, differences, references\n"
        "- perspectives phải là một đối tượng JSON chứa các khóa của góc nhìn dưới dạng chuỗi và nội dung trả lời dưới dạng chuỗi.\n"
        "- references phải là mảng JSON các chuỗi.\n"
        "- Trả lời toàn bộ bằng tiếng Việt."
    ),
}

CONCISENESS_SENTENCE_COUNT = {
    "brief": "1-2",
    "balanced": "2-4",
    "detailed": "4-7",
}


def resolve_sentence_count(
    conciseness: str = "balanced",
    sentences_per_section: int | None = None,
) -> str:
    if sentences_per_section is not None:
        return str(sentences_per_section)
    return CONCISENESS_SENTENCE_COUNT.get(
        conciseness, CONCISENESS_SENTENCE_COUNT["balanced"]
    )


RAG_INSTRUCTIONS = (
    "RAG mode:\n"
    "- Use the retrieved context as the primary source of evidence.\n"
    "- If the context is missing or conflicts, say so clearly.\n"
    "- Do not invent facts that are not supported by the retrieved context.\n"
    "- If no relevant chunks were retrieved, state that explicitly and answer cautiously using general WisdomLens knowledge."
)


def build_system_prompt(
    language: str = "vi",
    perspectives: list[str] | None = None,
    has_rag: bool = False,
    conciseness: str = "balanced",
    sentences_per_section: int | None = None,
) -> str:
    """Build standardized system prompt with perspectives and optional RAG rules."""
    prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["vi"])
    sentence_count = resolve_sentence_count(conciseness, sentences_per_section)
    prompt = prompt.replace("{sentence_count}", sentence_count)

    if perspectives is None:
        active_perspectives = ALL_DEFAULT_PERSPECTIVES[:]
    else:
        active_perspectives = [p.lower() for p in perspectives]

    lang_names = PERSPECTIVE_NAMES.get(language, PERSPECTIVE_NAMES["vi"])
    selected_names = [lang_names.get(p, p) for p in active_perspectives]

    if language == "en":
        perspective_instruction = (
            f"\n\nActive Perspectives:\n"
            f"You MUST only analyze the question and populate the 'perspectives' JSON object for these keys: {active_perspectives}.\n"
            f"Do NOT include any other keys in the 'perspectives' JSON object.\n"
            f"If only one perspective is selected, set similarities and differences to empty strings \"\"."
        )
    else:
        perspective_instruction = (
            f"\n\nGóc nhìn hoạt động:\n"
            f"Bạn BẮT BUỘC chỉ được phân tích câu hỏi và điền thông tin vào đối tượng JSON 'perspectives' cho các khóa sau: {active_perspectives}.\n"
            f"KHÔNG được bao gồm bất kỳ khóa nào khác trong đối tượng JSON 'perspectives'.\n"
            f"Nếu chỉ có một góc nhìn được chọn, hãy đặt similarities và differences thành chuỗi rỗng \"\"."
        )
    prompt = f"{prompt}\n{perspective_instruction}"

    if has_rag:
        prompt = f"{prompt}\n\n{RAG_INSTRUCTIONS}"

    return prompt


def build_user_content(
    question: str,
    rag_context: dict[str, Any] | None = None,
) -> tuple[str, list[dict]]:
    """Build standardized user content string and extract rag_sources."""
    rag_sources: list[dict] = []
    if rag_context is not None:
        rag_sources = rag_context.get("sources") or []
        retrieved_context = (rag_context.get("context") or "").strip()
        if retrieved_context:
            content = f"Retrieved context:\n{retrieved_context}\n\nUser question: {question}"
        else:
            content = (
                "Retrieved context: (none)\n\n"
                "No relevant document chunks were found for this question.\n\n"
                f"User question: {question}"
            )
    else:
        content = f"Question: {question}"

    return content, rag_sources


def clean_json_string(raw: str) -> str:
    """Strip markdown code block fences and trailing commas that break strict JSON parsing."""
    text = raw.strip()
    # Remove markdown code block if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Remove any trailing commas before closing braces/brackets
    return re.sub(r",\s*(?=[}\]])", "", text.strip())
