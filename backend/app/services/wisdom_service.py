from google import genai
from google.genai import types

from app.config import get_gemini_api_key, get_gemini_model
from app.schemas import AskResponse, GeminiWisdomFields

WISDOM_SYSTEM_PROMPT = """You are WisdomLens AI. Answer the user's life question by synthesizing three perspectives: Buddhism, Western philosophy, and psychology.

Rules:
- Provide structured informational perspectives, NOT personal advice or therapy.
- Be thoughtful, accessible, and concise (2-4 sentences per section).
- Cite real sources in references where possible (texts, thinkers, research areas).
- Return ONLY valid JSON with these keys:
  summary, buddhism, western_philosophy, psychology, similarities, differences, references
- references must be a JSON array of strings."""


def generate_fake_answer(question: str) -> dict:
    """Return a static structured answer for the MVP skeleton."""
    return {
        "question": question,
        "summary": (
            f"This is a placeholder summary for: \"{question}\". "
            "WisdomLens AI will eventually synthesize perspectives from "
            "Buddhism, Western philosophy and psychology."
        ),
        "buddhism": (
            "From a Buddhist perspective (placeholder): suffering often arises "
            "from attachment to outcomes and fear of losing a desired self-image. "
            "Failure threatens the ego's story about who we are."
        ),
        "western_philosophy": (
            "From Western philosophy (placeholder): thinkers from Stoicism to "
            "existentialism discuss failure as part of human finitude, virtue, "
            "or authentic choice rather than as a final verdict on one's worth."
        ),
        "psychology": (
            "From psychology (placeholder): fear of failure links to threat "
            "responses, perfectionism, learned avoidance, and social evaluation "
            "concerns shaped by past experience and context."
        ),
        "similarities": (
            "All three lenses (placeholder) connect fear of failure to how "
            "humans relate to uncertainty, identity, and the need for safety "
            "or meaning."
        ),
        "differences": (
            "They differ (placeholder) in emphasis: Buddhism on attachment and "
            "suffering, Western philosophy on ethics and meaning, psychology on "
            "emotion, cognition, and behavior."
        ),
        "references": [
            "Placeholder — Dhammapada (general Buddhist teachings)",
            "Placeholder — Epictetus, Enchiridion (Stoic perspective)",
            "Placeholder — APA overview on fear of failure (psychology)",
        ],
    }


def generate_gemini_answer(question: str) -> dict:
    """Call Gemini and return a structured answer matching AskResponse."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=f"Question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=WISDOM_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GeminiWisdomFields,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")

    fields = GeminiWisdomFields.model_validate_json(response.text)
    return AskResponse(question=question, **fields.model_dump()).model_dump()
