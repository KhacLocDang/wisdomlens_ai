import os

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
    return os.getenv(
        "DATABASE_URL",
        "postgresql://wisdomlens:wisdomlens@postgres:5432/wisdomlens",
    )
