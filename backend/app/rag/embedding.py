from typing import Any

from google import genai
from google.genai import types

from app.config import get_embedding_api_key, get_embedding_model


class EmbeddingError(Exception):
    pass


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        raise EmbeddingError("Embedding response contained no values.")
    if isinstance(values, list):
        return [float(v) for v in values]
    raise EmbeddingError("Embedding response values are not a list of floats.")


def generate_embedding(text: str) -> list[float]:
    api_key = get_embedding_api_key()
    if not api_key:
        raise EmbeddingError("Embedding API key is not configured.")

    model = get_embedding_model()
    if not model:
        raise EmbeddingError("Embedding model is not configured.")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model=model,
            contents=[text],
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding request failed: {exc}") from exc

    if response.embeddings is None or len(response.embeddings) == 0:
        raise EmbeddingError("Embedding API returned no embeddings.")

    embedding_obj = response.embeddings[0]
    return _to_float_list(embedding_obj.values)
