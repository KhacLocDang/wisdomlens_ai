from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["vi", "en"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Why are humans afraid of failure?"])
    language: Language = "vi"
    model: str | None = Field(
        default=None,
        examples=["gemini-2.5-flash"],
        description="Gemini model id from GET /models. Uses GEMINI_MODEL env default if omitted.",
    )


class ModelInfo(BaseModel):
    id: str
    display_name: str


class GeminiWisdomFields(BaseModel):
    summary: str
    buddhism: str
    western_philosophy: str
    psychology: str
    similarities: str
    differences: str
    references: list[str]


class AskResponse(BaseModel):
    question: str
    summary: str
    buddhism: str
    western_philosophy: str
    psychology: str
    similarities: str
    differences: str
    references: list[str]


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
