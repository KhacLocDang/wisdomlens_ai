import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_embedding_model, use_fake_answers, use_rag
from app.database import check_db_connection, get_db
from app.repositories.inquiry_repository import get_inquiry, list_inquiries, save_inquiry
from app.repositories.document_repository import get_document, list_documents
from app.repositories.chunk_repository import (
    get_document_chunks_for_document,
    get_document_chunks_without_embeddings,
    update_document_chunk_embedding,
)
from app.rag.document_loader import extract_text_from_bytes, ingest_document
from app.rag.embedding import generate_embedding
from app.rag.retriever import retrieve_similar_chunks
from app.services.rag_service import build_rag_context
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentChunkDetail,
    DocumentChunkRetrieval,
    DocumentCreate,
    DocumentSummary,
    InquiryDetail,
    InquirySummary,
    ModelInfo,
    EmbeddingRefreshResult,
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
    rag_enabled = request.use_rag if request.use_rag is not None else use_rag()
    perspectives = request.perspectives

    if perspectives is not None:
        supported = {"buddhism", "western_philosophy", "psychology", "christianity", "eastern_philosophy", "natural_science"}
        normalized = [p.strip().lower() for p in perspectives]
        if not normalized:
            raise HTTPException(status_code=400, detail="At least one perspective must be selected.")
        for p in normalized:
            if p not in supported:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported perspective: {p}. Supported perspectives are: {list(supported)}",
                )
        perspectives = normalized

    if use_fake_answers():
        answer = generate_fake_answer(question, language, perspectives=perspectives)
        source = "fake"
        model = None
        rag_sources: list[dict] = []
    else:
        try:
            model = resolve_model(request.model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            rag_context = None
            if rag_enabled:
                try:
                    rag_context = build_rag_context(db, query=question)
                except Exception:
                    logger.exception("RAG retrieval failed; falling back to non-RAG ask flow")
                    rag_context = None

            answer = generate_gemini_answer(
                question,
                language,
                model=model,
                rag_context=rag_context,
                perspectives=perspectives,
            )
            source = "gemini"
            rag_sources = answer.get("rag_sources") or []
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini request failed: {exc}",
            ) from exc

    try:
        save_inquiry(db, answer, language=language, source=source, model=model, rag_sources=rag_sources)
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
            embedding_model=chunk.embedding_model,
            created_at=chunk.created_at,
        )
        for chunk in chunks
    ]


@app.get("/rag/retrieve", response_model=list[DocumentChunkRetrieval])
def retrieve_document_chunks_endpoint(
    q: str = Query(..., min_length=1, description="Query text for semantic retrieval"),
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    try:
        chunks = retrieve_similar_chunks(db, query=q, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding or retrieval failed: {exc}") from exc

    return [
        DocumentChunkRetrieval(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata_json,
            score=chunk.score,
            embedding_model=chunk.embedding_model,
            created_at=chunk.created_at,
        )
        for chunk in chunks
    ]


@app.post(
    "/rag/embeddings/refresh",
    response_model=EmbeddingRefreshResult,
)
def refresh_embeddings_endpoint(
    document_id: int | None = Query(
        None,
        description="Optional document id to backfill embeddings for.",
    ),
    limit: int | None = Query(
        None,
        ge=1,
        le=500,
        description="Maximum number of missing chunk embeddings to refresh.",
    ),
    db: Session = Depends(get_db),
):
    try:
        chunks = get_document_chunks_without_embeddings(db, limit=limit, document_id=document_id)
        if not chunks:
            return {
                "refreshed_count": 0,
                "failed_count": 0,
                "refreshed_chunk_ids": [],
                "errors": [],
            }

        embedding_model = get_embedding_model()
        refreshed_chunk_ids: list[int] = []
        errors: list[dict[str, str]] = []
        quota_exhausted = False

        for chunk in chunks:
            try:
                embedding = generate_embedding(chunk.content)
                update_document_chunk_embedding(db, chunk.id, embedding, embedding_model)
                refreshed_chunk_ids.append(chunk.id)
            except Exception as exc:
                error_message = str(exc)
                logger.exception("Failed to refresh embedding for chunk %s", chunk.id)
                errors.append({"chunk_id": chunk.id, "error": error_message})

                if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message or "quota" in error_message.lower():
                    quota_exhausted = True
                    break

        return {
            "refreshed_count": len(refreshed_chunk_ids),
            "failed_count": len(errors),
            "refreshed_chunk_ids": refreshed_chunk_ids,
            "errors": errors,
            "quota_exhausted": quota_exhausted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during embedding refresh")
        raise HTTPException(
            status_code=502,
            detail=f"Embedding refresh failed: {exc}",
        ) from exc


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
        perspectives=inquiry.perspectives or {},
        similarities=inquiry.similarities,
        differences=inquiry.differences,
        references=inquiry.references or [],
        rag_sources=inquiry.rag_sources or [],
        language=inquiry.language,
        created_at=inquiry.created_at,
        source=inquiry.source,
        model=inquiry.model,
    )
