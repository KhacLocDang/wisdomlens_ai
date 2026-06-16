import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_gemini_model, use_fake_answers
from app.database import check_db_connection, get_db, init_db
from app.repositories.inquiry_repository import get_inquiry, list_inquiries, save_inquiry
from app.schemas import AskRequest, AskResponse, InquiryDetail, InquirySummary
from app.services.wisdom_service import generate_fake_answer, generate_gemini_answer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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


@app.post("/ask", response_model=AskResponse)
def ask_wisdom(request: AskRequest, db: Session = Depends(get_db)):
    question = request.question.strip()

    if use_fake_answers():
        answer = generate_fake_answer(question)
        source = "fake"
        model = None
    else:
        try:
            answer = generate_gemini_answer(question)
            source = "gemini"
            model = get_gemini_model()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini request failed: {exc}",
            ) from exc

    try:
        save_inquiry(db, answer, source=source, model=model)
    except Exception:
        logger.exception("Failed to save inquiry to database")

    return answer


@app.get("/inquiries", response_model=list[InquirySummary])
def list_inquiries_endpoint(limit: int = 20, db: Session = Depends(get_db)):
    limit = min(max(limit, 1), 50)
    inquiries = list_inquiries(db, limit=limit)
    return [
        InquirySummary(
            id=inquiry.id,
            question=inquiry.question,
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
        created_at=inquiry.created_at,
        source=inquiry.source,
        model=inquiry.model,
    )
