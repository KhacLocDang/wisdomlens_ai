"""OpenAI provider implementation."""

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
OPENAI_MODEL_CATALOG = [
    {"id": "gpt-4o-mini", "display_name": "GPT-4o Mini"},
    {"id": "gpt-4o", "display_name": "GPT-4o"},
    {"id": "gpt-4.1-mini", "display_name": "GPT-4.1 Mini"},
    {"id": "o3-mini", "display_name": "o3 Mini"},
    {"id": "gpt-4-turbo", "display_name": "GPT-4 Turbo"},
]


class OpenAIProvider(BaseLLMProvider):
    """LLM provider backed by OpenAI."""

    def list_models(self) -> list[dict]:
        from app.config import get_openai_api_key

        api_key = get_openai_api_key()
        if not api_key:
            return list(OPENAI_MODEL_CATALOG)

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)
            models = []
            for m in client.models.list():
                model_id = getattr(m, "id", None) or ""
                if not model_id:
                    continue
                lower_id = model_id.lower()
                # Include standard GPT and reasoning models, exclude non-chat or multimodal specialized models
                if (
                    any(lower_id.startswith(p) for p in ("gpt-", "o1", "o3"))
                    and not any(
                        x in lower_id
                        for x in ("realtime", "audio", "search", "transcription", "tts", "embedding")
                    )
                ):
                    display_name = model_id
                    for cat in OPENAI_MODEL_CATALOG:
                        if cat["id"] == model_id:
                            display_name = cat["display_name"]
                            break
                    models.append({"id": model_id, "display_name": display_name})

            if models:
                cat_ids = [c["id"] for c in OPENAI_MODEL_CATALOG]
                models.sort(key=lambda x: (0 if x["id"] in cat_ids else 1, x["id"]))
                return models
            return list(OPENAI_MODEL_CATALOG)
        except Exception:
            return list(OPENAI_MODEL_CATALOG)

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
        **kwargs,
    ) -> dict:
        from app.config import get_openai_api_key, get_openai_model
        from app.schemas import AskResponse, WisdomFields

        import openai

        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        model_id = model or get_openai_model()

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
        )
        user_content, rag_sources = build_user_content(question, rag_context)

        client = openai.OpenAI(api_key=api_key)

        request_kwargs = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }

        # Some models (reasoning models like o1/o3, newer models like
        # gpt-6.1-sol) reject custom temperature values.  Instead of
        # maintaining a fragile prefix list we attempt with temperature
        # first, then retry without it if the API rejects the value.
        _KNOWN_NO_TEMP_PREFIXES = ("o1", "o3")
        use_temperature = not any(
            model_id.startswith(p) for p in _KNOWN_NO_TEMP_PREFIXES
        )
        if use_temperature:
            request_kwargs["temperature"] = 0.7

        try:
            response = client.chat.completions.create(**request_kwargs)
        except openai.BadRequestError as exc:
            # Detect temperature-specific rejection and retry without it
            err_body = getattr(exc, "body", None) or {}
            err_msg = str(err_body.get("message", "")) if isinstance(err_body, dict) else str(exc)
            if "temperature" in err_msg.lower() and use_temperature:
                request_kwargs.pop("temperature", None)
                response = client.chat.completions.create(**request_kwargs)
            else:
                raise

        raw_text = ""
        finish_reason = None
        if response.choices:
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if hasattr(choice, "message") and getattr(choice.message, "content", None):
                raw_text = choice.message.content

        if not raw_text:
            raise ValueError("OpenAI returned an empty response")

        cleaned = clean_json_string(raw_text)
        try:
            fields = WisdomFields.model_validate_json(cleaned)
        except ValidationError:
            if finish_reason == "length":
                warning = (
                    "OpenAI reached its output token limit. The response may be incomplete; "
                    "the raw output has been preserved in the summary."
                )
            else:
                warning = (
                    "OpenAI returned a response that could not be parsed into the expected "
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

