"""Base abstract class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Contract that every LLM provider must implement."""

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        language: str = "vi",
        model: str | None = None,
        rag_context: dict | None = None,
        perspectives: list[str] | None = None,
    ) -> dict:
        """Generate a structured wisdom answer and return it as a dict."""

    @abstractmethod
    def list_models(self) -> list[dict]:
        """Return a list of available models as [{"id": ..., "display_name": ...}]."""

