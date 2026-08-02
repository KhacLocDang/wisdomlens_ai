import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import use_fake_answers
from app.database import check_db_connection, get_db
from app.repositories.inquiry_repository import get_inquiry, list_inquiries, save_inquiry
from app.repositories.document_repository import get_document, list_documents
from app.repositories.chunk_repository import get_document_chunks_for_document
from app.rag.document_loader import extract_text_from_bytes, ingest_document
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentChunkDetail,
    DocumentCreate,
    DocumentSummary,
    InquiryDetail,
    InquirySummary,
    ModelInfo,
)
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


@app.post("/rag/documents", response_model=DocumentSummary)
def create_document_endpoint(doc: DocumentCreate, db: Session = Depends(get_db)):
    document = ingest_document(
        db,
        title=doc.title,
        category=doc.category,
        author=doc.author,
        source_url=doc.source_url,
        content=doc.content,
        metadata=doc.metadata,
    )
    return DocumentSummary(
        id=document.id,
        title=document.title,
        category=document.category,
        author=document.author,
        source_url=document.source_url,
        created_at=document.created_at,
    )


@app.post("/rag/documents/upload", response_model=DocumentSummary)
def upload_document_endpoint(
    title: str = Form(...),
    category: str = Form(...),
    author: str | None = Form(None),
    source_url: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    try:
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        content = extract_text_from_bytes(file_bytes, filename=file.filename, content_type=file.content_type)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded file: {exc}") from exc

    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no extractable text.")

    document = ingest_document(
        db,
        title=title,
        category=category,
        author=author,
        source_url=source_url,
        content=content,
        metadata={"source_file": file.filename},
    )
    return DocumentSummary(
        id=document.id,
        title=document.title,
        category=document.category,
        author=document.author,
        source_url=document.source_url,
        created_at=document.created_at,
    )


@app.get("/rag/documents", response_model=list[DocumentSummary])
def list_documents_endpoint(limit: int = 50, db: Session = Depends(get_db)):
    limit = min(max(limit, 1), 100)
    documents = list_documents(db, limit=limit)
    return [
        DocumentSummary(
            id=document.id,
            title=document.title,
            category=document.category,
            author=document.author,
            source_url=document.source_url,
            created_at=document.created_at,
        )
        for document in documents
    ]


@app.get("/rag/documents/{document_id}/chunks", response_model=list[DocumentChunkDetail])
def list_document_chunks_endpoint(document_id: int, db: Session = Depends(get_db)):
    document = get_document(db, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = get_document_chunks_for_document(db, document_id=document_id)
    return [
        DocumentChunkDetail(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata_json,
            created_at=chunk.created_at,
        )
        for chunk in chunks
    ]


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
