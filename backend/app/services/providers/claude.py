"""Claude (Anthropic) provider implementation."""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.services.providers.base import BaseLLMProvider
from app.services.providers.common import (
    ALL_DEFAULT_PERSPECTIVES,
    build_system_prompt,
    build_user_content,
    clean_json_string,
)

# Static catalog — used as fallback when API key is absent or list fails.
CLAUDE_MODEL_CATALOG = [
    {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"},
    {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
    {"id": "claude-3-7-sonnet-20250219", "display_name": "Claude 3.7 Sonnet"},
]


class ClaudeProvider(BaseLLMProvider):
    """LLM provider backed by Anthropic Claude."""

    def list_models(self) -> list[dict]:
        from app.config import get_claude_api_key
        api_key = get_claude_api_key()
        if not api_key:
            return CLAUDE_MODEL_CATALOG

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            models = []
            for m in client.models.list():
                model_id = getattr(m, "id", None) or getattr(m, "name", None) or ""
                if not model_id or "claude" not in model_id.lower():
                    continue
                display_name = getattr(m, "display_name", None) or model_id
                models.append({"id": model_id, "display_name": display_name})
            return models or CLAUDE_MODEL_CATALOG
        except Exception:
            return CLAUDE_MODEL_CATALOG

    def generate_answer(
        self,
        question: str,
        language: str = "vi",
        model: str | None = None,
        rag_context: dict | None = None,
        perspectives: list[str] | None = None,
    ) -> dict:
        from app.config import get_claude_api_key, get_claude_model
        from app.schemas import AskResponse, WisdomFields

        import anthropic

        api_key = get_claude_api_key()
        if not api_key:
            raise ValueError("CLAUDE_API_KEY is not configured")

        model_id = model or get_claude_model()

        if perspectives is None:
            perspectives = ALL_DEFAULT_PERSPECTIVES[:]
        else:
            perspectives = [p.lower() for p in perspectives]

        system_prompt = build_system_prompt(
            language=language,
            perspectives=perspectives,
            has_rag=(rag_context is not None),
        )
        user_content, rag_sources = build_user_content(question, rag_context)

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw_text = block.text
                break

        if not raw_text:
            raise ValueError("Claude returned an empty response")

        cleaned = clean_json_string(raw_text)
        try:
            fields = WisdomFields.model_validate_json(cleaned)
        except ValidationError:
            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "max_tokens":
                warning = (
                    "Claude reached its output token limit. The response may be incomplete; "
                    "the raw output has been preserved in the summary."
                )
            else:
                warning = (
                    "Claude returned a response that could not be parsed into the expected "
                    "answer format. The raw output has been preserved in the summary."
                )
            return AskResponse(
                question=question,
                summary=raw_text,
                generation_warning=warning,
                rag_sources=rag_sources,
            ).model_dump()

        return AskResponse(
            question=question,
            rag_sources=rag_sources,
            **fields.model_dump(),
        ).model_dump()
