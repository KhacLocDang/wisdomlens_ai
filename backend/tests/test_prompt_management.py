"""Unit and integration tests for Prompt Management and Critique Template."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.prompt import Prompt
from app.models.inquiry import Inquiry
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas import PromptUpdateRequest
from app.services.prompt_service import (
    DEFAULT_PROMPTS,
    SUPPORTED_PROMPT_KEYS,
    format_prompt_safely,
    get_effective_prompt,
    get_prompt_info,
    list_prompts_info,
)
from app.services.providers.common import build_system_prompt
from app.services.manual_research import get_manual_research_config
from app.main import (
    get_prompt_endpoint,
    list_prompts_endpoint,
    restore_prompt_endpoint,
    update_prompt_endpoint,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database for test execution."""
    engine = create_engine("sqlite:///:memory:")
    Prompt.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_prompt_inventory_contains_all_required_keys():
    """Verify all 5 required prompts are inventoried with metadata and default texts."""
    expected_keys = {
        "ask_system_prompt_vi",
        "ask_system_prompt_en",
        "rag_instructions",
        "manual_research_system_prompt",
        "critique_template",
    }
    assert set(SUPPORTED_PROMPT_KEYS) == expected_keys

    for key in expected_keys:
        meta = DEFAULT_PROMPTS[key]
        assert meta["name"]
        assert meta["category"]
        assert meta["description"]
        assert meta["default"]
        assert len(meta["default"].strip()) > 20


def test_format_prompt_safely():
    """Verify format_prompt_safely replaces known placeholders and ignores unknown/malformed braces."""
    template = "Summary has {sentence_count} sentences. Also {extra_var} and raw {json_key: value}."

    # Safe substitution
    formatted = format_prompt_safely(template, sentence_count="2-4")
    assert "Summary has 2-4 sentences." in formatted
    assert "{extra_var}" in formatted  # unchanged, no KeyError
    assert "{json_key: value}" in formatted  # raw braces unchanged, no ValueError

    # Missing placeholder in template should not raise error
    plain = "No placeholders here."
    assert format_prompt_safely(plain, sentence_count="1-2") == plain


def test_get_effective_prompt_shipped_default(db_session):
    """When DB has no customization, effective prompt must match shipped default exactly."""
    for key in SUPPORTED_PROMPT_KEYS:
        default_text = DEFAULT_PROMPTS[key]["default"]
        assert get_effective_prompt(db_session, key) == default_text
        assert get_effective_prompt(None, key) == default_text


def test_update_and_restore_prompt_lifecycle(db_session):
    """Verify custom prompt save, effective retrieval, and restore-to-default lifecycle."""
    key = "ask_system_prompt_vi"
    default_text = DEFAULT_PROMPTS[key]["default"]
    custom_text = "Bạn là WisdomLens AI tùy biến theo phong cách đặc biệt."

    # Initial state
    info_initial = get_prompt_info(db_session, key)
    assert info_initial["is_customized"] is False
    assert info_initial["custom_prompt"] is None
    assert info_initial["effective_prompt"] == default_text

    # Update prompt
    req = PromptUpdateRequest(custom_prompt=custom_text)
    updated_resp = update_prompt_endpoint(key, req, db=db_session)
    assert updated_resp["is_customized"] is True
    assert updated_resp["custom_prompt"] == custom_text
    assert updated_resp["effective_prompt"] == custom_text
    assert get_effective_prompt(db_session, key) == custom_text

    # Restore default
    restored_resp = restore_prompt_endpoint(key, db=db_session)
    assert restored_resp["is_customized"] is False
    assert restored_resp["custom_prompt"] is None
    assert restored_resp["effective_prompt"] == default_text
    assert get_effective_prompt(db_session, key) == default_text


def test_list_prompts_endpoint(db_session):
    """Verify listing all prompts includes all categories and customization flags."""
    prompts = list_prompts_endpoint(db=db_session)
    assert len(prompts) == len(SUPPORTED_PROMPT_KEYS)
    keys = [p["key"] for p in prompts]
    assert "critique_template" in keys
    assert "ask_system_prompt_vi" in keys

    categories = {p["category"] for p in prompts}
    assert "Hỏi đáp (Ask)" in categories
    assert "Phản biện (Critique)" in categories
    assert "RAG" in categories


def test_build_system_prompt_uses_effective_prompt(db_session):
    """Verify build_system_prompt honors customized prompt in DB and formats sentence_count."""
    key = "ask_system_prompt_vi"
    custom_template = (
        "CUSTOM SYSTEM PROMPT: Mỗi phần viết đúng {sentence_count} câu nhé. "
        "Return valid JSON with summary, perspectives, similarities, differences, references."
    )
    update_prompt_endpoint(key, PromptUpdateRequest(custom_prompt=custom_template), db=db_session)

    rendered = build_system_prompt(
        language="vi",
        conciseness="balanced",
        db=db_session,
    )
    assert "CUSTOM SYSTEM PROMPT" in rendered
    assert "Mỗi phần viết đúng 2-4 câu nhé." in rendered


def test_manual_research_uses_effective_prompt(db_session):
    """Verify manual research config endpoint retrieves customized prompt and restores properly."""
    custom_mr = "Hướng dẫn nghiên cứu thủ công tùy biến cho dự án."
    update_prompt_endpoint(
        "manual_research_system_prompt",
        PromptUpdateRequest(custom_prompt=custom_mr),
        db=db_session,
    )

    cfg = get_manual_research_config(db=db_session)
    assert cfg["system_prompt"] == custom_mr

    restore_prompt_endpoint("manual_research_system_prompt", db=db_session)
    cfg_restored = get_manual_research_config(db=db_session)
    assert cfg_restored["system_prompt"] == DEFAULT_PROMPTS["manual_research_system_prompt"]["default"]


def test_critique_template_is_purely_optional_and_manual(db_session):
    """Verify critique_template exists, is customizable, but is NOT part of standard ask prompt."""
    critique_info = get_prompt_info(db_session, "critique_template")
    assert "Thẩm định" in critique_info["default_prompt"] or "phản biện" in critique_info["default_prompt"]
    assert "{{QUESTION}}" in critique_info["default_prompt"]
    assert "{{AI_ANSWER}}" in critique_info["default_prompt"]

    # Ensure build_system_prompt NEVER includes critique template
    ask_prompt = build_system_prompt(language="vi", db=db_session)
    assert "{{QUESTION}}" not in ask_prompt
    assert "{{AI_ANSWER}}" not in ask_prompt


def test_database_safety_no_impact_on_existing_tables(db_session):
    """Verify that Prompt table definition and operations are isolated from inquiries and documents."""
    # Ensure Prompt table is distinct and separate
    assert Prompt.__tablename__ == "prompts"
    assert Inquiry.__tablename__ == "inquiries"
    assert Document.__tablename__ == "documents"
    assert DocumentChunk.__tablename__ == "document_chunks"

    # Verify Prompt table has no foreign keys or shared columns with inquiries
    assert len(Prompt.__table__.foreign_keys) == 0
    prompt_col_names = {c.name for c in Prompt.__table__.columns}
    assert "key" in prompt_col_names
    assert "custom_prompt" in prompt_col_names
    assert "parent_id" not in prompt_col_names
    assert "question" not in prompt_col_names

    # Perform prompt CRUD
    update_prompt_endpoint(
        "ask_system_prompt_en",
        PromptUpdateRequest(custom_prompt="New custom prompt text"),
        db=db_session,
    )
    saved = get_prompt_info(db_session, "ask_system_prompt_en")
    assert saved["custom_prompt"] == "New custom prompt text"
