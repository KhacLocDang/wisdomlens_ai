import pytest

from app.rag.retriever import _cosine_similarity, RetrievalError


def test_cosine_similarity_same_vector():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_length_mismatch():
    with pytest.raises(RetrievalError, match="Vector lengths must match"):
        _cosine_similarity([1.0], [1.0, 0.0])
