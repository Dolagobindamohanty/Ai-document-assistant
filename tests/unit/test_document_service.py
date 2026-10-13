import pytest
from src.services.document_service import (
    calculate_content_hash,
    chunk_document_pages,
    extract_pdf_pages,
    validate_pdf_file,
)
from src.utils.errors import InvalidPDFError
from tests.fixtures.pdf_fixtures import (
    generate_corrupt_pdf_bytes,
    generate_empty_bytes,
    generate_pdf_bytes,
)


def test_doc_t01_valid_text_pdf_extraction():
    """DOC-T01 & DOC-T04: Extract valid text PDF, verify page count and text."""
    pdf_bytes = generate_pdf_bytes(["Artificial Intelligence in 2026", "Retrieval Augmented Generation details"])
    pages = extract_pdf_pages(pdf_bytes, "ai_doc.pdf")

    assert len(pages) == 2
    assert pages[0].source == "ai_doc.pdf"
    assert pages[0].page == 1
    assert "Artificial Intelligence" in pages[0].text
    assert pages[1].page == 2
    assert "Retrieval Augmented Generation" in pages[1].text


def test_doc_t02_non_pdf_rejected():
    """DOC-T02: Upload a non-PDF file renamed to .pdf."""
    fake_bytes = b"This is a plain text file pretending to be a PDF."
    with pytest.raises(InvalidPDFError) as exc_info:
        validate_pdf_file(fake_bytes, "fake.pdf")
    assert "PDF header" in str(exc_info.value)


def test_doc_t03_empty_and_corrupt_rejected():
    """DOC-T03: Empty and corrupt PDFs are rejected with clean InvalidPDFError."""
    empty_bytes = generate_empty_bytes()
    with pytest.raises(InvalidPDFError) as exc_info:
        validate_pdf_file(empty_bytes, "empty.pdf")
    assert "empty" in str(exc_info.value).lower()

    corrupt_bytes = generate_corrupt_pdf_bytes()
    with pytest.raises(InvalidPDFError) as exc_info:
        validate_pdf_file(corrupt_bytes, "corrupt.pdf")
    assert "unreadable" in str(exc_info.value).lower() or "corrupt" in str(exc_info.value).lower()


def test_doc_t05_low_text_warning():
    """DOC-T05: Page with zero or very little text is flagged as low_text."""
    pdf_bytes = generate_pdf_bytes(["Normal narrative text with plenty of words to exceed threshold.", " "])
    pages = extract_pdf_pages(pdf_bytes, "scanned_sample.pdf")

    assert len(pages) == 2
    assert pages[0].is_low_text is False
    assert pages[1].is_low_text is True


def test_doc_t06_chunking_preserves_metadata():
    """DOC-T06: Chunks retain source, page number, and document_id metadata."""
    pdf_bytes = generate_pdf_bytes(["Line 1. Line 2. Detailed explanation of RAG architecture."])
    pages = extract_pdf_pages(pdf_bytes, "rag_guide.pdf")

    chunks = chunk_document_pages(pages, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata["source"] == "rag_guide.pdf"
        assert chunk.metadata["page"] == 1
        assert "document_id" in chunk.metadata


def test_doc_t08_duplicate_hash_detection():
    """DOC-T08 & DOC-08: Identical content produces identical SHA-256 hash."""
    bytes_a = generate_pdf_bytes(["Test content for duplicate detection"])
    bytes_b = generate_pdf_bytes(["Test content for duplicate detection"])
    bytes_c = generate_pdf_bytes(["Different content altogether"])

    hash_a = calculate_content_hash(bytes_a)
    hash_b = calculate_content_hash(bytes_b)
    hash_c = calculate_content_hash(bytes_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
