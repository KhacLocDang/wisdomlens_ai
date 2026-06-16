from datetime import datetime

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Why are humans afraid of failure?"])


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
    created_at: datetime
    source: str


class InquiryDetail(AskResponse):
    id: int
    created_at: datetime
    source: str
    model: str | None = None
