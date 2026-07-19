import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import use_fake_answers
from app.database import check_db_connection, get_db
from app.repositories.inquiry_repository import get_inquiry, list_inquiries, save_inquiry
from app.schemas import AskRequest, AskResponse, InquiryDetail, InquirySummary, ModelInfo
from app.services.wisdom_service import (
    generate_fake_answer,
    generate_gemini_answer,
    list_gemini_models,
    resolve_model,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="WisdomLens AI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "database": "ok" if check_db_connection() else "error",
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models_endpoint():
    """List Gemini text models available for this API key (cached)."""
    models = list_gemini_models()
    return [ModelInfo(id=m["id"], display_name=m["display_name"]) for m in models]


@app.post("/ask", response_model=AskResponse)
def ask_wisdom(request: AskRequest, db: Session = Depends(get_db)):
    question = request.question.strip()
    language = request.language

    if use_fake_answers():
        answer = generate_fake_answer(question, language)
        source = "fake"
        model = None
    else:
        try:
            model = resolve_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            answer = generate_gemini_answer(question, language, model=model)
            source = "gemini"
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini request failed: {exc}",
            ) from exc

    try:
        save_inquiry(db, answer, language=language, source=source, model=model)
    except Exception:
        logger.exception("Failed to save inquiry to database")

    return answer


@app.get("/inquiries", response_model=list[InquirySummary])
def list_inquiries_endpoint(
    q: Optional[str] = Query(None, description="Search keyword for question or summary"),
    limit: int = 20,
    db: Session = Depends(get_db),
):
    limit = min(max(limit, 1), 50)
    inquiries = list_inquiries(db, limit=limit, q=q)
    return [
        InquirySummary(
            id=inquiry.id,
            question=inquiry.question,
            language=inquiry.language,
            created_at=inquiry.created_at,
            source=inquiry.source,
        )
        for inquiry in inquiries
    ]


@app.get("/inquiries/{inquiry_id}", response_model=InquiryDetail)
def get_inquiry_endpoint(inquiry_id: int, db: Session = Depends(get_db)):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    return InquiryDetail(
        id=inquiry.id,
        question=inquiry.question,
        summary=inquiry.summary,
        buddhism=inquiry.buddhism,
        western_philosophy=inquiry.western_philosophy,
        psychology=inquiry.psychology,
        similarities=inquiry.similarities,
        differences=inquiry.differences,
        references=inquiry.references or [],
        language=inquiry.language,
        created_at=inquiry.created_at,
        source=inquiry.source,
        model=inquiry.model,
    )
