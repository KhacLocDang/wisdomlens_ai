import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_claude_model, get_embedding_model, use_fake_answers, use_rag
from app.config import get_audio_storage_root
from app.database import check_db_connection, get_db
from app.repositories.inquiry_repository import (
    clear_inquiry_audio,
    fork_inquiry,
    get_inquiry,
    get_inquiry_audio_file_path,
    list_inquiries,
    save_inquiry,
    update_inquiry_answer,
    update_inquiry_audio,
)
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
from app.services.manual_research import get_manual_research_config, validate_manual_research_topics
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentChunkDetail,
    DocumentChunkRetrieval,
    DocumentCreate,
    DocumentSummary,
    InquiryAudioResponse,
    InquiryDetail,
    InquirySummary,
    InquiryUpdateRequest,
    ManualResearchConfigResponse,
    ManualResearchCreateRequest,
    ModelInfo,
    EmbeddingRefreshResult,
    TTSModelInfo,
    TTSRequest,
    TTSVoiceInfo,
)
from app.services.wisdom_service import (
    generate_fake_answer,
    generate_gemini_answer,
    generate_provider_answer,
    list_gemini_models,
    resolve_model,
)
from app.services.providers import (
    list_models_for_provider,
    list_supported_providers,
)
from app.services.providers.claude import get_claude_model_max_output_tokens
from app.services.tts_service import (
    list_tts_models,
    list_tts_voices,
    synthesize_speech,
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


@app.get("/providers", response_model=list[str])
def list_providers_endpoint():
    """List supported AI providers."""
    return list_supported_providers()


@app.get("/models", response_model=list[ModelInfo])
def list_models_endpoint(provider: Optional[str] = Query(None, description="Optional provider filter (gemini, claude)")):
    """List available text models for specified provider or default provider."""
    try:
        return list_models_for_provider(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tts/models", response_model=list[TTSModelInfo])
def list_tts_models_endpoint():
    """List available Gemini Text-to-Speech models."""
    return list_tts_models()


@app.get("/tts/voices", response_model=list[TTSVoiceInfo])
def list_tts_voices_endpoint():
    """List available Gemini Text-to-Speech voices."""
    return list_tts_voices()


@app.post("/tts/synthesize")
def synthesize_speech_endpoint(request: TTSRequest):
    """Synthesize text into speech audio and return audio stream."""
    try:
        audio_bytes, mime_type = synthesize_speech(
            text=request.text,
            voice=request.voice,
            model=request.model,
            language=request.language,
        )
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={"Content-Disposition": "inline; filename=wisdom_speech.wav"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in TTS synthesis")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask_wisdom(request: AskRequest, db: Session = Depends(get_db)):
    question = request.question.strip()
    language = request.language
    rag_enabled = request.use_rag if request.use_rag is not None else use_rag()
    perspectives = request.perspectives
    requested_provider = request.provider.strip().lower() if request.provider else None

    if requested_provider == "claude" and request.claude_max_tokens is not None:
        model_id_for_limit = request.model or get_claude_model()
        max_allowed_tokens = get_claude_model_max_output_tokens(model_id_for_limit)
        if request.claude_max_tokens > max_allowed_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"claude_max_tokens cannot exceed {max_allowed_tokens} for model "
                    f"{model_id_for_limit}."
                ),
            )

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
            rag_context = None
            if rag_enabled:
                try:
                    rag_context = build_rag_context(db, query=question)
                except Exception:
                    logger.exception("RAG retrieval failed; falling back to non-RAG ask flow")
                    rag_context = None

            if requested_provider in (None, "gemini"):
                model = resolve_model(request.model)
                answer = generate_gemini_answer(
                    question,
                    language,
                    model=model,
                    rag_context=rag_context,
                    perspectives=perspectives,
                    conciseness=request.conciseness,
                    sentences_per_section=request.sentences_per_section,
                )
                source = "gemini"
            else:
                answer = generate_provider_answer(
                    question=question,
                    provider=requested_provider,
                    language=language,
                    model=request.model,
                    rag_context=rag_context,
                    perspectives=perspectives,
                    conciseness=request.conciseness,
                    sentences_per_section=request.sentences_per_section,
                    claude_max_tokens=request.claude_max_tokens,
                )
                source = requested_provider
                model = request.model or answer.get("model")
            rag_sources = answer.get("rag_sources") or []
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            prov_label = requested_provider or "Gemini"
            raise HTTPException(
                status_code=502,
                detail=f"{prov_label} request failed: {exc}",
            ) from exc

    answer_to_save = answer
    if answer.get("generation_warning"):
        answer_to_save = {
            **answer,
            "summary": (
                "[Response could not be parsed; raw model output follows]\n\n"
                f"{answer['summary']}"
            ),
        }

    try:
        save_inquiry(db, answer_to_save, language=language, source=source, model=model, rag_sources=rag_sources)
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
            parent_id=getattr(inquiry, "parent_id", None),
            question=inquiry.question,
            language=inquiry.language,
            created_at=inquiry.created_at,
            source=inquiry.source,
            answer_type=inquiry.answer_type or "generated",
            ai_source=inquiry.ai_source,
            topics=getattr(inquiry, "topics", None) or [],
        )
        for inquiry in inquiries
    ]


@app.get("/manual-research/config", response_model=ManualResearchConfigResponse)
def manual_research_config_endpoint():
    return get_manual_research_config()


@app.post("/manual-research", response_model=InquiryDetail)
def create_manual_research_endpoint(
    request: ManualResearchCreateRequest,
    db: Session = Depends(get_db),
):
    question = request.question.strip()
    answer_text = request.answer.strip()
    ai_source = request.ai_source.strip()
    model = request.model.strip()
    if not question or not answer_text or not ai_source or not model:
        raise HTTPException(status_code=400, detail="Question, answer, AI source, and model are required.")
    try:
        topics = validate_manual_research_topics(request.topics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    answer = {
        "question": question,
        "summary": answer_text,
        "perspectives": {},
        "similarities": "",
        "differences": "",
        "references": [],
        "rag_sources": [],
    }
    inquiry = save_inquiry(
        db,
        answer,
        language=request.language,
        source=ai_source,
        model=model,
        rag_sources=[],
        answer_type="manual_answer",
        topics=topics,
    )
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
        audio_available=False,
        audio_filename=inquiry.audio_filename,
        audio_mime_type=inquiry.audio_mime_type,
        audio_voice=inquiry.audio_voice,
        audio_model=inquiry.audio_model,
        answer_type=inquiry.answer_type,
        manual_fields=[],
        ai_source=None,
        manual_system_prompt=None,
        manual_sections=[],
        topics=getattr(inquiry, "topics", None) or [],
    )


@app.get("/inquiries/{inquiry_id}", response_model=InquiryDetail)
def get_inquiry_endpoint(inquiry_id: int, db: Session = Depends(get_db)):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    audio_filename = inquiry.audio_filename or None
    audio_available = bool(
        audio_filename and get_inquiry_audio_file_path(inquiry_id, storage_root=get_audio_storage_root()).exists()
    )

    return InquiryDetail(
        id=inquiry.id,
        parent_id=getattr(inquiry, "parent_id", None),
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
        audio_available=audio_available,
        audio_filename=audio_filename,
        audio_mime_type=inquiry.audio_mime_type,
        audio_voice=inquiry.audio_voice,
        audio_model=inquiry.audio_model,
        answer_type=inquiry.answer_type or "generated",
        manual_fields=inquiry.manual_fields or [],
        ai_source=inquiry.ai_source,
        manual_system_prompt=inquiry.manual_system_prompt,
        manual_sections=inquiry.manual_sections or [],
        topics=getattr(inquiry, "topics", None) or [],
    )


@app.put("/inquiries/{inquiry_id}", response_model=InquiryDetail)
def update_inquiry_endpoint(
    inquiry_id: int,
    request: InquiryUpdateRequest,
    db: Session = Depends(get_db),
):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    updated = update_inquiry_answer(
        db,
        inquiry,
        summary=request.summary.strip(),
        perspectives=request.perspectives,
        similarities=request.similarities,
        differences=request.differences,
        references=request.references,
    )

    audio_filename = updated.audio_filename or None
    audio_available = bool(
        audio_filename and get_inquiry_audio_file_path(inquiry_id, storage_root=get_audio_storage_root()).exists()
    )

    return InquiryDetail(
        id=updated.id,
        parent_id=updated.parent_id,
        question=updated.question,
        summary=updated.summary,
        perspectives=updated.perspectives or {},
        similarities=updated.similarities,
        differences=updated.differences,
        references=updated.references or [],
        rag_sources=updated.rag_sources or [],
        language=updated.language,
        created_at=updated.created_at,
        source=updated.source,
        model=updated.model,
        audio_available=audio_available,
        audio_filename=audio_filename,
        audio_mime_type=updated.audio_mime_type,
        audio_voice=updated.audio_voice,
        audio_model=updated.audio_model,
        answer_type=updated.answer_type or "generated",
        manual_fields=updated.manual_fields or [],
        ai_source=updated.ai_source,
        manual_system_prompt=updated.manual_system_prompt,
        manual_sections=updated.manual_sections or [],
        topics=getattr(updated, "topics", None) or [],
    )


@app.post("/inquiries/{inquiry_id}/fork", response_model=InquiryDetail)
def fork_inquiry_endpoint(
    inquiry_id: int,
    request: InquiryUpdateRequest,
    db: Session = Depends(get_db),
):
    original = get_inquiry(db, inquiry_id)
    if original is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    forked = fork_inquiry(
        db,
        original,
        summary=request.summary.strip(),
        perspectives=request.perspectives,
        similarities=request.similarities,
        differences=request.differences,
        references=request.references,
    )

    return InquiryDetail(
        id=forked.id,
        parent_id=forked.parent_id,
        question=forked.question,
        summary=forked.summary,
        perspectives=forked.perspectives or {},
        similarities=forked.similarities,
        differences=forked.differences,
        references=forked.references or [],
        rag_sources=forked.rag_sources or [],
        language=forked.language,
        created_at=forked.created_at,
        source=forked.source,
        model=forked.model,
        audio_available=False,
        audio_filename=None,
        audio_mime_type=None,
        audio_voice=None,
        audio_model=None,
        answer_type=forked.answer_type or "generated",
        manual_fields=forked.manual_fields or [],
        ai_source=forked.ai_source,
        manual_system_prompt=forked.manual_system_prompt,
        manual_sections=forked.manual_sections or [],
        topics=getattr(forked, "topics", None) or [],
    )


@app.post("/inquiries/{inquiry_id}/audio", response_model=InquiryAudioResponse)
def save_inquiry_audio_endpoint(
    inquiry_id: int,
    request: TTSRequest,
    db: Session = Depends(get_db),
):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    try:
        audio_bytes, mime_type = synthesize_speech(
            text=request.text,
            voice=request.voice,
            model=request.model,
            language=request.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in inquiry audio generation")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {exc}") from exc

    root = get_audio_storage_root()
    file_path = root / f"inquiry_{inquiry_id}.wav"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(audio_bytes)

    updated = update_inquiry_audio(
        db,
        inquiry,
        audio_filename=file_path.name,
        audio_mime_type=mime_type,
        audio_voice=request.voice,
        audio_model=request.model,
    )
    return InquiryAudioResponse(
        inquiry_id=inquiry_id,
        audio_available=True,
        audio_filename=updated.audio_filename,
        audio_mime_type=updated.audio_mime_type,
        audio_voice=updated.audio_voice,
        audio_model=updated.audio_model,
        audio_created_at=getattr(updated, "audio_created_at", None),
    )


@app.get("/inquiries/{inquiry_id}/audio")
def get_inquiry_audio_endpoint(inquiry_id: int, db: Session = Depends(get_db)):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")
    if not inquiry.audio_filename:
        raise HTTPException(status_code=404, detail="Audio not found for this inquiry")

    file_path = get_inquiry_audio_file_path(inquiry_id, storage_root=get_audio_storage_root())
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")

    mime_type = inquiry.audio_mime_type or "audio/wav"
    return Response(
        content=file_path.read_bytes(),
        media_type=mime_type,
        headers={"Content-Disposition": f"inline; filename={inquiry.audio_filename}"},
    )


@app.delete("/inquiries/{inquiry_id}/audio", response_model=InquiryAudioResponse)
def delete_inquiry_audio_endpoint(inquiry_id: int, db: Session = Depends(get_db)):
    inquiry = get_inquiry(db, inquiry_id)
    if inquiry is None:
        raise HTTPException(status_code=404, detail="Inquiry not found")

    file_path = get_inquiry_audio_file_path(inquiry_id, storage_root=get_audio_storage_root())
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    updated = clear_inquiry_audio(db, inquiry)
    return InquiryAudioResponse(
        inquiry_id=inquiry_id,
        audio_available=False,
        audio_filename=None,
        audio_mime_type=None,
        audio_voice=None,
        audio_model=None,
        audio_created_at=None,
    )
