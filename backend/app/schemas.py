from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["vi", "en"]
Conciseness = Literal["brief", "balanced", "detailed"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Why are humans afraid of failure?"])
    language: Language = "vi"
    use_rag: bool | None = Field(
        default=None,
        description="Override the USE_RAG env flag for this request.",
    )
    provider: str | None = Field(
        default=None,
        examples=["gemini", "claude"],
        description="LLM provider: 'gemini' or 'claude'. Uses AI_PROVIDER env default if omitted.",
    )
    model: str | None = Field(
        default=None,
        examples=["gemini-2.5-flash", "claude-3-5-haiku-20241022"],
        description="Model id. Uses provider default env if omitted.",
    )
    perspectives: list[str] | None = Field(
        default=None,
        examples=[["buddhism", "psychology"]],
        description="List of perspectives to include. Defaults to all if not specified.",
    )
    conciseness: Conciseness = Field(
        default="balanced",
        description="Answer length: brief (1-2), balanced (2-4), detailed (4-7) sentences per section.",
    )
    sentences_per_section: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Legacy override: exact sentence count per section. Prefer conciseness.",
    )
    claude_max_tokens: int | None = Field(default=None, ge=1, le=64000)


class ModelInfo(BaseModel):
    id: str
    display_name: str
    provider: str | None = None
    max_output_tokens: int | None = None
    default_max_tokens: int | None = None


class WisdomFields(BaseModel):
    summary: str
    perspectives: dict[str, str] = Field(default_factory=dict)
    similarities: str = ""
    differences: str = ""
    references: list[str] = Field(default_factory=list)


# Backward-compatible alias for existing code/tests
GeminiWisdomFields = WisdomFields


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
    generation_warning: str | None = None
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
    answer_type: str = "generated"
    ai_source: str | None = None
    topics: list[str] = Field(default_factory=list)


class InquiryDetail(AskResponse):
    id: int
    language: str
    created_at: datetime
    source: str
    model: str | None = None
    audio_available: bool = False
    audio_filename: str | None = None
    audio_mime_type: str | None = None
    audio_voice: str | None = None
    audio_model: str | None = None
    answer_type: str = "generated"
    manual_fields: list[str] = Field(default_factory=list)
    ai_source: str | None = None
    manual_system_prompt: str | None = None
    manual_sections: list[dict[str, str]] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class ManualResearchCreateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    answer: str = Field(..., min_length=1, max_length=100000)
    language: Language = "vi"
    ai_source: str = Field(..., min_length=1, max_length=20)
    model: str = Field(..., min_length=1, max_length=100)
    topics: list[str] = Field(default_factory=list, max_length=7)


class ManualResearchTopicOption(BaseModel):
    key: str
    label: str


class ManualResearchConfigResponse(BaseModel):
    system_prompt: str
    topics: list[ManualResearchTopicOption]


class InquiryAudioResponse(BaseModel):
    inquiry_id: int
    audio_available: bool
    audio_filename: str | None = None
    audio_mime_type: str | None = None
    audio_voice: str | None = None
    audio_model: str | None = None
    audio_created_at: datetime | None = None


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


class TTSVoiceInfo(BaseModel):
    id: str
    name: str
    description: str


class TTSModelInfo(BaseModel):
    id: str
    display_name: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Cuộc sống là một hành trình liên tục thay đổi."])
    voice: str | None = Field(
        default=None,
        examples=["Aoede", "Puck"],
        description="Gemini prebuilt voice name. Defaults to GEMINI_TTS_VOICE if omitted.",
    )
    model: str | None = Field(
        default=None,
        examples=["gemini-2.0-flash", "gemini-2.5-flash"],
        description="Gemini TTS model. Defaults to GEMINI_TTS_MODEL if omitted.",
    )
    language: Language = "vi"

