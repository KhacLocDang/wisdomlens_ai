import os

from dotenv import load_dotenv

load_dotenv()


def use_fake_answers() -> bool:
    return os.getenv("USE_FAKE_ANSWERS", "false").lower() in ("1", "true", "yes")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None


# Used when Gemini models.list() fails or returns nothing useful.
FALLBACK_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]


def get_gemini_model() -> str:
    return normalize_model_id(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


def normalize_model_id(model_id: str) -> str:
    """Strip 'models/' prefix so IDs match generate_content."""
    name = (model_id or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return name


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://wisdomlens:wisdomlens@postgres:5432/wisdomlens",
    )
