import os

from dotenv import load_dotenv

load_dotenv()


def use_fake_answers() -> bool:
    return os.getenv("USE_FAKE_ANSWERS", "false").lower() in ("1", "true", "yes")


def get_gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def get_database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://wisdomlens:wisdomlens@postgres:5432/wisdomlens",
    )
