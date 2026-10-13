from unittest.mock import MagicMock
from src.models.schemas import DocumentPage
from src.services.summary_service import partition_pages_into_batches, summarize_full_document


class MockGenAIResponse:
    def __init__(self, text: str):
        self.text = text


def test_sum_t01_partition_covers_all_pages():
    """SUM-T01: All non-empty pages are included in batch partitions."""
    pages = [
        DocumentPage(text="Page 1 content", source="doc.pdf", page=1),
        DocumentPage(text="", source="doc.pdf", page=2),  # empty page
        DocumentPage(text="Page 3 content", source="doc.pdf", page=3),
    ]

    batches = partition_pages_into_batches(pages, max_chars=100)
    assert len(batches) >= 1

    all_batch_text = " ".join(b[0] for b in batches)
    assert "Page 1 content" in all_batch_text
    assert "Page 3 content" in all_batch_text


def test_sum_t02_sentinel_facts_summary_success():
    """SUM-T02: Full document summarization preserves multi-stage synthesis."""
    pages = [
        DocumentPage(text="Beginning sentinel: Alpha Corp was founded in 1999.", source="report.pdf", page=1),
        DocumentPage(text="Middle sentinel: Beta subsidiary launched in 2010.", source="report.pdf", page=10),
        DocumentPage(text="Ending sentinel: Gamma project finalized in 2025.", source="report.pdf", page=25),
    ]

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        MockGenAIResponse("Summary of Section 1: Alpha Corp 1999."),
        MockGenAIResponse("Summary of Section 2: Beta subsidiary 2010."),
        MockGenAIResponse("Summary of Section 3: Gamma project 2025."),
        # Reduction and final formatting
        MockGenAIResponse("Reduced summary combining all sections."),
        MockGenAIResponse("Final Complete Summary with Overview, Topics, and Key Facts."),
    ]

    result = summarize_full_document(
        pages=pages,
        client=mock_client,
    )

    assert result.status == "complete"
    assert result.pages_total == 3
    assert result.pages_processed == 3
    assert result.batches_completed == result.batches_total
    assert len(result.sources) == 3


def test_sum_t04_partial_failure_honesty():
    """SUM-T04: An unresolved batch failure marks status as partial, never complete."""
    pages = [
        DocumentPage(text="Batch 1 content", source="book.pdf", page=1),
        DocumentPage(text="Batch 2 content", source="book.pdf", page=2),
    ]

    mock_client = MagicMock()
    # Batch 1 succeeds, Batch 2 fails permanently with unrecoverable error, Final pass succeeds
    mock_client.models.generate_content.side_effect = [
        MockGenAIResponse("Summary of batch 1"),
        Exception("Permanent unrecoverable batch failure"),
        MockGenAIResponse("Final summary with partial content"),
    ]

    result = summarize_full_document(
        pages=pages,
        client=mock_client,
        max_chars_per_batch=50,
    )

    assert result.status == "partial"
    assert result.batches_completed < result.batches_total
    assert any("failed" in w.lower() for w in result.warnings)
