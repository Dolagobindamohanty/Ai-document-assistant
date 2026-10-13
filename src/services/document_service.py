import hashlib
from io import BytesIO
from typing import Any

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.models.schemas import DocumentPage
from src.utils.errors import InvalidPDFError


def calculate_content_hash(file_bytes: bytes) -> str:
    """Calculate SHA-256 hash of file content to detect duplicates."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_pdf_file(file_bytes: bytes, filename: str) -> None:
    """
    Validate that the uploaded file is a readable PDF within the configured size limit.
    """
    if not file_bytes:
        raise InvalidPDFError(filename, "File is completely empty (0 bytes).")

    # Check file size limit
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise InvalidPDFError(
            filename,
            f"File exceeds maximum allowed size of {config.MAX_UPLOAD_MB} MB."
        )

    # Check PDF magic bytes (%PDF-)
    if not file_bytes.startswith(b"%PDF-"):
        raise InvalidPDFError(filename, "PDF header is missing or invalid (%PDF- not found).")

    # Attempt to parse with pypdf
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if len(reader.pages) == 0:
            raise InvalidPDFError(filename, "PDF contains zero pages.")
    except Exception as exc:
        raise InvalidPDFError(filename, f"Unreadable or corrupt PDF structure: {exc}") from exc


def extract_pdf_pages(
    file_bytes: bytes,
    filename: str,
    document_id: str | None = None,
) -> list[DocumentPage]:
    """
    Extract page-by-page text from a PDF, preserving one-based page numbers
    and tracking low-text pages for potential OCR warnings.
    """
    validate_pdf_file(file_bytes, filename)
    doc_id = document_id or calculate_content_hash(file_bytes)[:12]

    reader = PdfReader(BytesIO(file_bytes))
    extracted_pages: list[DocumentPage] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        clean_text = page_text.strip()
        is_low_text = len(clean_text) < config.LOW_TEXT_CHAR_THRESHOLD

        extracted_pages.append(
            DocumentPage(
                text=clean_text,
                source=filename,
                page=page_num,
                document_id=doc_id,
                char_count=len(clean_text),
                is_low_text=is_low_text,
            )
        )

    return extracted_pages


def chunk_document_pages(
    pages: list[DocumentPage],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split DocumentPages into LangChain Document chunks while strictly preserving
    metadata (source, page, document_id) on every chunk.
    """
    effective_chunk_size = chunk_size or config.CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    langchain_docs: list[Document] = []

    for p in pages:
        if not p.text:
            continue
        langchain_docs.append(
            Document(
                page_content=p.text,
                metadata={
                    "source": p.source,
                    "page": p.page,
                    "document_id": p.document_id,
                },
            )
        )

    if not langchain_docs:
        return []

    return splitter.split_documents(langchain_docs)
