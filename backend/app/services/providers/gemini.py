"""Gemini provider implementation."""

from __future__ import annotations

from app.services.providers.base import BaseLLMProvider
from app.services.providers.common import build_system_prompt, build_user_content, clean_json_string


class GeminiProvider(BaseLLMProvider):
    """LLM provider backed by Google Gemini."""

    def list_models(self) -> list[dict]:
        from app.services.wisdom_service import list_gemini_models
        return list_gemini_models()

    def generate_answer(
        self,
        question: str,
        language: str = "vi",
        model: str | None = None,
        rag_context: dict | None = None,
        perspectives: list[str] | None = None,
    ) -> dict:
        from app.services.wisdom_service import generate_gemini_answer
        return generate_gemini_answer(
            question=question,
            language=language,
            model=model,
            rag_context=rag_context,
            perspectives=perspectives,
        )

