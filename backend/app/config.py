import os
import socket
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

load_dotenv()


def use_fake_answers() -> bool:
    return os.getenv("USE_FAKE_ANSWERS", "false").lower() in ("1", "true", "yes")


def use_rag() -> bool:
    return os.getenv("USE_RAG", "false").lower() in ("1", "true", "yes")


def get_rag_min_score() -> float:
    """Minimum cosine score a chunk needs before it is used as RAG context."""
    try:
        return float(os.getenv("RAG_MIN_SCORE", "0.35"))
    except ValueError:
        return 0.35


def get_ai_provider() -> str:
    """Default AI provider (gemini, claude). Defaults to gemini."""
    return os.getenv("AI_PROVIDER", "gemini").strip().lower()


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


def get_claude_api_key() -> str | None:
    return os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or None


FALLBACK_CLAUDE_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]


def get_claude_model() -> str:
    return os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022").strip()


def get_embedding_api_key() -> str | None:
    return os.getenv("EMBEDDING_API_KEY") or os.getenv("GEMINI_API_KEY") or None


def get_embedding_model() -> str:
    return normalize_model_id(os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview"))


def normalize_model_id(model_id: str) -> str:
    """Strip 'models/' prefix so IDs match generate_content."""
    name = (model_id or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return name


def get_database_url() -> str:
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://wisdomlens:wisdomlens@postgres:5432/wisdomlens",
    )

    # Ensure we always use psycopg2 dialect (SQLAlchemy 2.1+ defaults to psycopg3
    # when scheme is plain "postgresql://", but we only have psycopg2-binary installed).
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)

    parsed = urlparse(database_url)
    if parsed.hostname == "postgres":
        try:
            socket.getaddrinfo(parsed.hostname, None)
            return database_url
        except OSError:
            fallback_netloc = parsed.netloc.replace("postgres", "127.0.0.1", 1)
            return urlunparse(parsed._replace(netloc=fallback_netloc))

    return database_url
