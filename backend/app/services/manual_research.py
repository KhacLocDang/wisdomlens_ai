SYSTEM_PROMPT = """Bạn là trợ lý nghiên cứu cho WisdomLens AI. Trả lời câu hỏi bằng một nội dung hoàn chỉnh, có cấu trúc tự nhiên. Khi người dùng yêu cầu nhiều góc nhìn, phân biệt rõ từng góc nhìn trong câu trả lời nhưng không chia thành các biểu mẫu hoặc mục cố định của WisdomLens. Phân biệt thông tin có nguồn, diễn giải và điều chưa chắc chắn. Không bịa dữ kiện, trích dẫn, tác giả hoặc đường dẫn; nếu không thể xác minh nguồn, hãy nói rõ. Trình bày cân bằng, nêu giới hạn và không đưa ra chẩn đoán hay lời khuyên trị liệu cá nhân."""

TOPIC_OPTIONS = [
    {"key": "buddhism", "label": "Phật giáo"},
    {"key": "psychology", "label": "Tâm lý học"},
    {"key": "western_philosophy", "label": "Triết học phương Tây"},
    {"key": "christianity", "label": "Thiên Chúa giáo"},
    {"key": "eastern_philosophy", "label": "Triết học phương Đông"},
    {"key": "natural_science", "label": "Khoa học tự nhiên"},
    {"key": "other", "label": "Khác"},
]


def get_manual_research_config() -> dict:
    return {"system_prompt": SYSTEM_PROMPT, "topics": TOPIC_OPTIONS}


def validate_manual_research_topics(topics: list[str]) -> list[str]:
    allowed = {topic["key"] for topic in TOPIC_OPTIONS}
    if len(set(topics)) != len(topics):
        raise ValueError("Topic labels must be unique.")
    if any(topic not in allowed for topic in topics):
        raise ValueError("One or more topic labels are unsupported.")
    return topics