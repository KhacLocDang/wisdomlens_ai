import io

from PyPDF2 import PdfReader

from app.repositories.document_repository import create_document
from app.repositories.chunk_repository import create_document_chunk
from app.rag.chunker import clean_text, split_into_chunks
from app.rag.embedding import generate_embedding
from app.config import get_embedding_model


def extract_text_from_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text_parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


def extract_text_from_bytes(data: bytes, filename: str | None = None, content_type: str | None = None) -> str:
    if content_type == "application/pdf" or (filename and filename.lower().endswith(".pdf")):
        return extract_text_from_pdf_bytes(data)
    return data.decode("utf-8", errors="replace")


def ingest_document(db, *, title: str, category: str, author: str | None = None, source_url: str | None = None, content: str, metadata: dict | None = None):
    metadata = metadata or {}

    clean = clean_text(content)
    chunks = split_into_chunks(clean)

    with db.begin():
        document = create_document(
            db,
            title=title,
            category=category,
            author=author,
            source_url=source_url,
            metadata=metadata,
            commit=False,
        )

        for index, chunk_text in enumerate(chunks, start=1):
            chunk_metadata = {
                **metadata,
                "source_title": title,
                "source_url": source_url,
                "category": category,
                "chunk_index": index,
                "chunk_length": len(chunk_text.split()),
            }
            embedding = generate_embedding(chunk_text)
            create_document_chunk(
                db,
                document_id=document.id,
                content=chunk_text,
                metadata=chunk_metadata,
                embedding=embedding,
                embedding_model=get_embedding_model(),
                commit=False,
            )

    return document
