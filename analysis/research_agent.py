"""WisdomLens Research Agent v0.1.

Standalone experimental research tool. It reads existing inquiry data from
PostgreSQL, asks Gemini to look for dataset-grounded patterns, and writes a
Markdown research report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect, select

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_database_url, get_gemini_api_key, get_gemini_model  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models.inquiry import Inquiry  # noqa: E402


DEFAULT_LIMIT = 200
TEXT_PREVIEW_CHARS = 900
DEFAULT_MAX_PROMPT_CHARS = 200_000
DEFAULT_REPORT_PATH = REPO_ROOT / "analysis" / "research_report.md"

RESEARCH_SYSTEM_INSTRUCTION = """Bạn là WisdomLens Research Agent v0.1.

Nhiệm vụ của bạn là phân tích dataset WisdomLens được cung cấp và tạo một báo
cáo nghiên cứu cẩn trọng bằng Markdown, viết bằng tiếng Việt.

Quy tắc:
- Chỉ sử dụng dataset được cung cấp trong prompt.
- Không khẳng định các kết luận như chân lý phổ quát về con người, tôn giáo,
  triết học hoặc tâm lý học.
- Phân biệt rõ giữa Quan sát, Mẫu hình, và Giả thuyết.
- Quan sát phải được hỗ trợ trực tiếp bởi dữ liệu được cung cấp.
- Mẫu hình có thể mô tả khuynh hướng ngữ nghĩa lặp lại, nhưng phải thận trọng.
- Giả thuyết phải được trình bày như ý tưởng đáng điều tra tiếp, không phải kết luận.
- Nêu rõ hạn chế, đặc biệt là cỡ mẫu nhỏ, trường dữ liệu thiếu, câu hỏi trùng lặp,
  câu trả lời do model tạo, và khả năng thiên lệch chọn mẫu.
- Ưu tiên nhận định ngắn gọn, hữu ích hơn suy đoán rộng.
- Nếu dataset trống hoặc quá nhỏ, hãy nói rõ và tránh phân tích quá mức.
- Toàn bộ báo cáo phải viết bằng tiếng Việt, trừ khi cần giữ nguyên thuật ngữ kỹ thuật.

Cấu trúc Markdown bắt buộc:
1. # Báo cáo nghiên cứu WisdomLens
2. ## Ảnh chụp dataset
3. ## Quan sát
4. ## Mẫu hình lặp lại
5. ## Các câu hỏi tương đồng về ngữ nghĩa
6. ## Khác biệt giữa các góc nhìn
7. ## Tín hiệu bất ngờ hoặc thú vị
8. ## Giả thuyết cần điều tra thêm
9. ## Hạn chế
10. ## Bước nghiên cứu tiếp theo
"""

LEGACY_PERSPECTIVE_COLUMNS = {
    "buddhism": "buddhism",
    "western_philosophy": "western_philosophy",
    "psychology": "psychology",
}


def configure_utf8_output() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def truncate_text(value: Any, limit: int = TEXT_PREVIEW_CHARS) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def normalize_json_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if item is not None}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): item for key, item in parsed.items() if item is not None}
    return {}


def normalize_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def merge_perspectives(row: Any) -> dict[str, str]:
    perspectives = normalize_json_object(row.perspectives)
    normalized: dict[str, str] = {
        str(key): truncate_text(value)
        for key, value in perspectives.items()
        if str(value).strip()
    }

    for key, column_name in LEGACY_PERSPECTIVE_COLUMNS.items():
        value = getattr(row, column_name, None)
        if value and key not in normalized:
            normalized[key] = truncate_text(value)

    return normalized


def serialize_inquiry(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "question": truncate_text(row.question),
        "summary": truncate_text(row.summary),
        "perspectives": merge_perspectives(row),
        "similarities": truncate_text(row.similarities),
        "differences": truncate_text(row.differences),
        "references": [truncate_text(item, 300) for item in normalize_json_list(row.references)],
        "rag_sources_count": len(normalize_json_list(row.rag_sources)),
        "language": row.language,
        "source": row.source,
        "model": row.model,
        "created_at": isoformat(row.created_at),
    }


def load_inquiry_records(limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        result = db.execute(
            select(Inquiry)
            .order_by(Inquiry.created_at.desc(), Inquiry.id.desc())
            .limit(limit)
        )
        rows = result.scalars().all()

    return [serialize_inquiry(row) for row in rows]


def inspect_database() -> dict[str, Any]:
    with SessionLocal() as db:
        inspector = inspect(db.bind)
        table_names = inspector.get_table_names()
        inquiry_columns = []
        if "inquiries" in table_names:
            inquiry_columns = [
                {"name": column["name"], "type": str(column["type"])}
                for column in inspector.get_columns("inquiries")
            ]

    return {
        "database_url": get_database_url(),
        "tables": table_names,
        "inquiry_columns": inquiry_columns,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    perspective_keys: set[str] = set()
    languages: dict[str, int] = {}
    sources: dict[str, int] = {}
    models: dict[str, int] = {}

    for record in records:
        perspective_keys.update(record["perspectives"].keys())
        languages[record["language"] or "unknown"] = languages.get(record["language"] or "unknown", 0) + 1
        sources[record["source"] or "unknown"] = sources.get(record["source"] or "unknown", 0) + 1
        models[record["model"] or "unknown"] = models.get(record["model"] or "unknown", 0) + 1

    return {
        "loaded_records": len(records),
        "available_perspectives": sorted(perspective_keys),
        "languages": dict(sorted(languages.items())),
        "sources": dict(sorted(sources.items())),
        "models": dict(sorted(models.items())),
        "date_range": {
            "oldest": min((r["created_at"] for r in records if r["created_at"]), default=None),
            "newest": max((r["created_at"] for r in records if r["created_at"]), default=None),
        },
    }


def build_research_dataset(limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    records = load_inquiry_records(limit=limit)
    return {
        "metadata": {
            "purpose": "WisdomLens Research Agent v0.1 readonly data preparation",
            "limit": limit,
            "record_count": len(records),
        },
        "summary": summarize_records(records),
        "records": records,
    }


def build_research_prompt(dataset: dict[str, Any]) -> str:
    dataset_json = json.dumps(dataset, ensure_ascii=False, indent=2, default=str)
    return (
        "Hãy phân tích dataset WisdomLens này. Các records là lịch sử inquiry gần đây "
        "được đọc từ PostgreSQL database của ứng dụng.\n\n"
        "Tập trung vào các chủ đề lặp lại trong câu hỏi về đời sống, mối quan hệ giữa "
        "các chủ đề, các câu hỏi tương đồng về ngữ nghĩa, khác biệt giữa các góc nhìn, "
        "tín hiệu bất ngờ, và các giả thuyết đáng nghiên cứu tiếp.\n\n"
        "Hãy viết báo cáo cuối cùng bằng tiếng Việt.\n\n"
        "Dataset JSON:\n"
        "```json\n"
        f"{dataset_json}\n"
        "```"
    )


def estimate_prompt_chars(dataset: dict[str, Any]) -> int:
    return len(RESEARCH_SYSTEM_INSTRUCTION) + len(build_research_prompt(dataset))


def validate_dataset_for_llm(dataset: dict[str, Any], max_prompt_chars: int) -> None:
    if max_prompt_chars < 10_000:
        raise ValueError("--max-prompt-chars must be at least 10000.")

    prompt_chars = estimate_prompt_chars(dataset)
    if prompt_chars > max_prompt_chars:
        raise ValueError(
            "Prepared dataset is too large for the configured prompt budget. "
            f"Estimated prompt size: {prompt_chars} characters. "
            f"Budget: {max_prompt_chars} characters. "
            "Run again with a lower --limit or a higher --max-prompt-chars value."
        )


def build_report_preamble(dataset: dict[str, Any], model: str) -> str:
    summary = dataset["summary"]
    return (
        "<!--\n"
        "Generated by WisdomLens Research Agent v0.1\n"
        f"Generated at: {datetime.now(UTC).isoformat()}\n"
        f"Model: {model}\n"
        f"Loaded records: {summary['loaded_records']}\n"
        f"Date range: {summary['date_range']['oldest']} -> {summary['date_range']['newest']}\n"
        "Database access: readonly SELECT from inquiries\n"
        "-->\n\n"
    )


def generate_research_report(dataset: dict[str, Any], model: str | None = None) -> str:
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    model_id = model or get_gemini_model()

    if not dataset["records"]:
        return (
            build_report_preamble(dataset, model_id)
            + (
            "# Báo cáo nghiên cứu WisdomLens\n\n"
            "## Ảnh chụp dataset\n\n"
            "Không có inquiry record nào, nên chưa chạy phân tích nghiên cứu bằng LLM.\n\n"
            "## Hạn chế\n\n"
            "- Bảng `inquiries` trả về 0 record.\n"
            "- Cần thêm câu hỏi WisdomLens thật trước khi chạy lại research agent.\n"
            )
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=build_research_prompt(dataset),
        config=types.GenerateContentConfig(
            system_instruction=RESEARCH_SYSTEM_INSTRUCTION,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty research report.")
    return build_report_preamble(dataset, model_id) + response.text.strip() + "\n"


def write_report(report: str, output_path: Path = DEFAULT_REPORT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main() -> None:
    configure_utf8_output()

    parser = argparse.ArgumentParser(description="Run WisdomLens Research Agent v0.1.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum inquiry records to load.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print prepared data preview only; do not call Gemini or write the research report.",
    )
    parser.add_argument(
        "--preview-records",
        type=int,
        default=3,
        help="Number of prepared records to print when using --preview.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Gemini model override. Defaults to GEMINI_MODEL.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        help="Stop before calling Gemini if the prepared prompt is larger than this.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate DB access, data shape, prompt size, and Gemini configuration without calling Gemini.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.preview_records < 0:
        raise SystemExit("--preview-records must be 0 or greater")

    try:
        db_info = inspect_database()
        dataset = build_research_dataset(limit=args.limit)
    except SQLAlchemyError as exc:
        raise SystemExit(
            "Could not connect to the WisdomLens database. "
            "Start PostgreSQL with `docker compose up postgres` or the full stack with "
            "`docker compose up --build`, then run this script again.\n\n"
            f"Database URL: {get_database_url()}\n"
            f"Error: {exc}"
        ) from exc

    output = {
        "database": {
            "tables": db_info["tables"],
            "inquiry_columns": db_info["inquiry_columns"],
        },
        "summary": dataset["summary"],
        "checks": {
            "gemini_api_key_configured": bool(get_gemini_api_key()),
            "selected_model": args.model or get_gemini_model(),
            "estimated_prompt_chars": estimate_prompt_chars(dataset),
            "max_prompt_chars": args.max_prompt_chars,
            "would_write_report_to": str(args.output),
        },
        "preview_records": dataset["records"][: args.preview_records],
    }

    try:
        validate_dataset_for_llm(dataset, args.max_prompt_chars)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.preview or args.check_only:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
        return

    try:
        report = generate_research_report(dataset, model=args.model)
    except Exception as exc:
        raise SystemExit(f"Research report generation failed: {exc}") from exc

    report_path = write_report(report, args.output)
    print(f"Research report written to: {report_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
