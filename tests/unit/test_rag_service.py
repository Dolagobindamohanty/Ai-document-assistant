from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.models.schemas import SourceReference
from src.services.index_service import build_workspace_index
from src.services.rag_service import answer_document_query


class MockGenAIResponse:
    def __init__(self, text: str):
        self.text = text


def test_rag_t01_grounded_query_success():
    """RAG-T01 & RAG-T05: Grounded query returns success with citations and model name."""
    chunks = [
        Document(
            page_content="Retrieval Augmented Generation combines retrieval with language models.",
            metadata={"source": "rag_paper.pdf", "page": 4},
        )
    ]
    index = build_workspace_index(chunks)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MockGenAIResponse(
        "RAG combines retrieval with language models [rag_paper.pdf, p. 4]."
    )

    result = answer_document_query(
        vector_store=index,
        query="What does RAG combine?",
        client=mock_client,
    )

    assert result.status == "success"
    assert "RAG combines retrieval" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].source == "rag_paper.pdf"
    assert result.sources[0].page == 4
    assert mock_client.models.generate_content.call_count == 1


def test_rag_t03_irrelevant_query_abstains_without_calling_llm():
    """RAG-T03 & RAG-T06: Distant/irrelevant query yields insufficient_evidence with zero LLM calls."""
    chunks = [
        Document(
            page_content="Quantum chromodynamics deals with quarks and gluons.",
            metadata={"source": "physics.pdf", "page": 12},
        )
    ]
    index = build_workspace_index(chunks)

    mock_client = MagicMock()

    # Query completely unrelated with strict threshold
    result = answer_document_query(
        vector_store=index,
        query="What is the recipe for chocolate cake?",
        max_distance=0.1,  # Strict threshold to force rejection
        client=mock_client,
    )

    assert result.status == "insufficient_evidence"
    assert "I don't know based on the provided documents" in result.answer
    assert len(result.sources) == 0
    # LLM must NOT be called when evidence is rejected!
    assert mock_client.models.generate_content.call_count == 0


def test_source_reference_citation_label():
    """RAG-T07: Citation label matches target specification [filename, p. X]."""
    ref = SourceReference(source="deep_learning.pdf", page=42)
    assert ref.to_citation_label() == "[deep_learning.pdf, p. 42]"
