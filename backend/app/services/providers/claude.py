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
# Max output tokens per model (based on Anthropic docs, Nov 2026):
#   claude-3-5-haiku / claude-3-5-sonnet : 8,192
#   claude-3-7-sonnet                     : 64,000
#   claude-haiku-4-5 / claude-sonnet-4-5 / claude-opus-4-5 : 32,000
#   claude-sonnet-5-5 / claude-opus-5-5  : 128,000
CLAUDE_MODEL_CATALOG = [
    {"id": "claude-3-5-haiku-20241022", "display_name": "Claude 3.5 Haiku"},
    {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
    {"id": "claude-3-7-sonnet-20250219", "display_name": "Claude 3.7 Sonnet"},
    {"id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5"},
    {"id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5"},
    {"id": "claude-opus-4-5-20251101", "display_name": "Claude Opus 4.5"},
    {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5"},
    {"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5"},
]


def get_claude_model_max_output_tokens(model_id: str) -> int:
    """Return the maximum output token limit for a given Claude model ID."""
    normalized = model_id.strip().lower()
    # Claude 5.5 series — 128,000 output tokens
    if "claude-sonnet-5-5" in normalized or "claude-opus-5-5" in normalized:
        return 128000
    # Claude 4.5 series — 32,000 output tokens
    if any(tag in normalized for tag in ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5")):
        return 32000
    # Claude 3.7 Sonnet — 64,000 output tokens
    if "claude-3-7-sonnet" in normalized:
        return 64000
    # Claude 3.5 and older — 8,192 output tokens
    return 8192


def get_claude_default_max_tokens(model_id: str) -> int:
    from app.config import get_claude_max_tokens

    return min(get_claude_max_tokens(), get_claude_model_max_output_tokens(model_id))


def _with_output_limits(models: list[dict]) -> list[dict]:
    return [
        {
            **item,
            "max_output_tokens": get_claude_model_max_output_tokens(item["id"]),
            "default_max_tokens": get_claude_default_max_tokens(item["id"]),
        }
        for item in models
    ]


class ClaudeProvider(BaseLLMProvider):
    """LLM provider backed by Anthropic Claude."""

    def list_models(self) -> list[dict]:
        from app.config import get_claude_api_key
        api_key = get_claude_api_key()
        if not api_key:
            return _with_output_limits(CLAUDE_MODEL_CATALOG)

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
            return _with_output_limits(models or CLAUDE_MODEL_CATALOG)
        except Exception:
            return _with_output_limits(CLAUDE_MODEL_CATALOG)

    def generate_answer(
        self,
        question: str,
        language: str = "vi",
        model: str | None = None,
        rag_context: dict | None = None,
        perspectives: list[str] | None = None,
        conciseness: str = "balanced",
        sentences_per_section: int | None = None,
        claude_max_tokens: int | None = None,
        db: object | None = None,
        **kwargs: object,
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
            conciseness=conciseness,
            sentences_per_section=sentences_per_section,
            db=db,
        )
        user_content, rag_sources = build_user_content(question, rag_context)

        model_token_limit = get_claude_model_max_output_tokens(model_id)
        if claude_max_tokens is None:
            max_tokens = get_claude_default_max_tokens(model_id)
        elif claude_max_tokens > model_token_limit:
            raise ValueError(
                f"claude_max_tokens exceeds the {model_id} output limit of {model_token_limit}."
            )
        else:
            max_tokens = claude_max_tokens

        client = anthropic.Anthropic(api_key=api_key)
        # Use streaming to support models that require it for long requests
        # (Claude 4.5+ / 5.5 enforce streaming for responses that may take >10 min).
        # stream.get_final_message() blocks until complete and returns a standard Message.
        raw_text = ""
        stop_reason = None
        with client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            final_message = stream.get_final_message()
            stop_reason = getattr(final_message, "stop_reason", None)
            for block in final_message.content:
                if getattr(block, "type", None) == "text":
                    raw_text = block.text
                    break

        if not raw_text:
            raise ValueError("Claude returned an empty response")

        cleaned = clean_json_string(raw_text)
        try:
            fields = WisdomFields.model_validate_json(cleaned)
        except ValidationError:
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
