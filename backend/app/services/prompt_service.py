"""Prompt service managing shipped defaults, persistence overrides, and safe formatting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Shipped default prompts: Single Source of Truth for defaults
DEFAULT_PROMPTS: dict[str, dict[str, str]] = {
    "ask_system_prompt_vi": {
        "name": "System Prompt Hỏi đáp (Tiếng Việt)",
        "category": "Hỏi đáp (Ask)",
        "description": "Prompt hệ thống tiếng Việt cho luồng tổng hợp tri thức đa góc nhìn (/ask). Hỗ trợ biến {sentence_count} để điều chỉnh độ dài câu trả lời.",
        "default": (
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
    },
    "ask_system_prompt_en": {
        "name": "System Prompt Hỏi đáp (English)",
        "category": "Hỏi đáp (Ask)",
        "description": "English system prompt for synthesizing multiple perspectives (/ask). Supports {sentence_count} placeholder for answer length.",
        "default": (
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
    },
    "rag_instructions": {
        "name": "Hướng dẫn RAG (RAG Instructions)",
        "category": "RAG",
        "description": "Hướng dẫn bổ sung được đính kèm vào System Prompt khi có context truy xuất từ tài liệu RAG.",
        "default": (
            "RAG mode:\n"
            "- Use the retrieved context as the primary source of evidence.\n"
            "- If the context is missing or conflicts, say so clearly.\n"
            "- Do not invent facts that are not supported by the retrieved context.\n"
            "- If no relevant chunks were retrieved, state that explicitly and answer cautiously using general WisdomLens knowledge."
        ),
    },
    "manual_research_system_prompt": {
        "name": "System Prompt Nghiên cứu thủ công",
        "category": "Nghiên cứu thủ công (Manual Research)",
        "description": "System prompt hiển thị trong tab Nghiên cứu AI thủ công để người dùng sao chép sang AI bên ngoài.",
        "default": (
            "Bạn là trợ lý nghiên cứu cho WisdomLens AI. Trả lời câu hỏi bằng một nội dung hoàn chỉnh, "
            "có cấu trúc tự nhiên. Khi người dùng yêu cầu nhiều góc nhìn, phân biệt rõ từng góc nhìn trong "
            "câu trả lời nhưng không chia thành các biểu mẫu hoặc mục cố định của WisdomLens. Phân biệt "
            "thông tin có nguồn, diễn giải và điều chưa chắc chắn. Không bịa dữ kiện, trích dẫn, tác giả "
            "hoặc đường dẫn; nếu không thể xác minh nguồn, hãy nói rõ. Trình bày cân bằng, nêu giới hạn "
            "và không đưa ra chẩn đoán hay lời khuyên trị liệu cá nhân."
        ),
    },
    "critique_template": {
        "name": "Template Phản biện câu trả lời AI (AI Answer Critique Template)",
        "category": "Phản biện (Critique)",
        "description": "Khung prompt độc lập với nhà cung cấp/mô hình dùng để sao chép sang AI bên ngoài (hoặc cùng AI) nhằm thẩm định và phản biện câu trả lời của WisdomLens.",
        "default": (
            "Bạn là chuyên gia thẩm định và phản biện tri thức (WisdomLens AI Answer Evaluator).\n"
            "Nhiệm vụ của bạn là đánh giá, kiểm chứng và phản biện một câu trả lời do AI tạo ra dựa trên câu hỏi ban đầu của người dùng.\n\n"
            "[CÂU HỎI BAN ĐẦU CỦA NGƯỜI DÙNG]:\n"
            "{{QUESTION}}\n\n"
            "[CÂU TRẢ LỜI CỦA AI CẦN PHẢN BIỆN]:\n"
            "{{AI_ANSWER}}\n\n"
            "---\n"
            "HÃY PHÂN TÍCH VÀ PHẢN BIỆN THEO CÁC TIÊU CHÍ SAU:\n"
            "1. Tính chính xác & Nguồn gốc (Accuracy & Grounding):\n"
            "   - Các góc nhìn, luận điểm hoặc trích dẫn có chuẩn xác theo các trường phái/học thuyết được đề cập không? Có dấu hiệu bịa đặt thông tin (hallucination) không?\n"
            "2. Tính khách quan & Cân bằng (Objectivity & Neutrality):\n"
            "   - Câu trả lời có giữ thái độ trung lập, mang tính cung cấp thông tin thay vì áp đặt hay đưa ra lời khuyên trị liệu/cá nhân hóa không?\n"
            "3. Điểm mạnh (Strengths):\n"
            "   - Những điểm tổng hợp tốt, sâu sắc, có giá trị khai sáng hoặc liên kết ý nghĩa giữa các góc nhìn.\n"
            "4. Điểm thiếu sót hoặc Giới hạn (Blind Spots & Weaknesses):\n"
            "   - Những khía cạnh quan trọng bị bỏ qua, hiểu lầm khái niệm, hoặc điểm khác biệt chưa được làm rõ.\n"
            "5. Gợi ý hoàn thiện hoặc Góc nhìn mở rộng (Recommendations):\n"
            "   - Các nguồn tham khảo, tài liệu triết học/tâm lý chuẩn mực hoặc góc nhìn phản đề cần bổ sung để câu trả lời hoàn thiện hơn."
        ),
    },
}

SUPPORTED_PROMPT_KEYS = list(DEFAULT_PROMPTS.keys())


def format_prompt_safely(template: str, **kwargs: str | int | None) -> str:
    """Safely replace known placeholders without throwing KeyError/ValueError on other braces."""
    result = template
    for key, val in kwargs.items():
        if val is not None:
            result = result.replace(f"{{{key}}}", str(val))
    return result


def get_effective_prompt(db: Session | None, key: str) -> str:
    """Return active prompt text: custom_prompt if saved, otherwise shipped default."""
    if key not in DEFAULT_PROMPTS:
        raise ValueError(f"Unsupported prompt key: {key}")

    shipped_default = DEFAULT_PROMPTS[key]["default"]
    if db is None:
        return shipped_default

    try:
        from app.repositories.prompt_repository import get_prompt
        record = get_prompt(db, key)
        if record and record.custom_prompt and record.custom_prompt.strip():
            return record.custom_prompt.strip()
    except Exception as exc:
        logger.warning("Failed to load prompt '%s' from database, using shipped default: %s", key, exc)

    return shipped_default


def list_prompts_info(db: Session | None) -> list[dict]:
    """List all prompts with customization status and effective text."""
    db_records = {}
    if db is not None:
        try:
            from app.repositories.prompt_repository import list_prompts
            records = list_prompts(db)
            db_records = {r.key: r for r in records}
        except Exception as exc:
            logger.warning("Failed to list prompts from database: %s", exc)

    items = []
    for key, meta in DEFAULT_PROMPTS.items():
        record = db_records.get(key)
        custom_prompt = (
            record.custom_prompt.strip()
            if record and record.custom_prompt and record.custom_prompt.strip()
            else None
        )
        is_customized = custom_prompt is not None
        effective_prompt = custom_prompt if is_customized else meta["default"]
        updated_at = getattr(record, "updated_at", None) if record else None

        items.append({
            "key": key,
            "name": meta["name"],
            "category": meta["category"],
            "description": meta["description"],
            "default_prompt": meta["default"],
            "custom_prompt": custom_prompt,
            "effective_prompt": effective_prompt,
            "is_customized": is_customized,
            "updated_at": updated_at,
        })
    return items


def get_prompt_info(db: Session | None, key: str) -> dict:
    """Retrieve full details for a single prompt key."""
    if key not in DEFAULT_PROMPTS:
        raise ValueError(f"Unsupported prompt key: {key}")

    meta = DEFAULT_PROMPTS[key]
    custom_prompt = None
    updated_at = None

    if db is not None:
        try:
            from app.repositories.prompt_repository import get_prompt
            record = get_prompt(db, key)
            if record and record.custom_prompt and record.custom_prompt.strip():
                custom_prompt = record.custom_prompt.strip()
            updated_at = getattr(record, "updated_at", None) if record else None
        except Exception as exc:
            logger.warning("Failed to get prompt '%s' from database: %s", key, exc)

    is_customized = custom_prompt is not None
    return {
        "key": key,
        "name": meta["name"],
        "category": meta["category"],
        "description": meta["description"],
        "default_prompt": meta["default"],
        "custom_prompt": custom_prompt,
        "effective_prompt": custom_prompt if is_customized else meta["default"],
        "is_customized": is_customized,
        "updated_at": updated_at,
    }
