from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["vi", "en"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Why are humans afraid of failure?"])
    language: Language = "vi"
    use_rag: bool | None = Field(
        default=None,
        description="Override the USE_RAG env flag for this request.",
    )
    model: str | None = Field(
        default=None,
        examples=["gemini-2.5-flash"],
        description="Gemini model id from GET /models. Uses GEMINI_MODEL env default if omitted.",
    )
    perspectives: list[str] | None = Field(
        default=None,
        examples=[["buddhism", "psychology"]],
        description="List of perspectives to include. Defaults to all if not specified.",
    )


class ModelInfo(BaseModel):
    id: str
    display_name: str


class GeminiWisdomFields(BaseModel):
    summary: str
    perspectives: dict[str, str] = Field(default_factory=dict)
    similarities: str = ""
    differences: str = ""
    references: list[str] = Field(default_factory=list)


class RagSource(BaseModel):
    rank: int
    chunk_id: int
    document_id: int
    score: float
    title: str | None = None
    category: str | None = None
    author: str | None = None
    source_url: str | None = None
    chunk_index: int | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] | None = None


class AskResponse(BaseModel):
    question: str
    summary: str
    perspectives: dict[str, str] = Field(default_factory=dict)
    similarities: str = ""
    differences: str = ""
    references: list[str] = Field(default_factory=list)
    rag_sources: list[RagSource] = Field(default_factory=list)


class InquirySummary(BaseModel):
    id: int
    question: str
    language: str
    created_at: datetime
    source: str


class InquiryDetail(AskResponse):
    id: int
    language: str
    created_at: datetime
    source: str
    model: str | None = None


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    author: str | None = None
    source_url: str | None = None
    content: str = Field(..., min_length=1)
    metadata: dict[str, str] | None = None


class DocumentSummary(BaseModel):
    id: int
    title: str
    category: str
    author: str | None = None
    source_url: str | None = None
    created_at: datetime


class DocumentChunkDetail(BaseModel):
    id: int
    document_id: int
    content: str
    metadata: dict[str, Any] | None = None
    embedding_model: str | None = None
    created_at: datetime


class DocumentChunkRetrieval(DocumentChunkDetail):
    score: float


class EmbeddingRefreshError(BaseModel):
    chunk_id: int
    error: str


class EmbeddingRefreshResult(BaseModel):
    refreshed_count: int
    failed_count: int
    refreshed_chunk_ids: list[int]
    errors: list[EmbeddingRefreshError]
    quota_exhausted: bool = False
