from langchain_core.documents import Document
from src.services.index_service import build_workspace_index


def test_doc_t07_isolated_workspace_index_creation():
    """DOC-T07: In-memory workspace index isolates documents and returns correct source identities."""
    chunks = [
        Document(
            page_content="Machine learning enables pattern recognition.",
            metadata={"source": "doc1.pdf", "page": 1, "document_id": "doc1_hash"},
        ),
        Document(
            page_content="Retrieval augmented generation connects LLMs to private databases.",
            metadata={"source": "doc2.pdf", "page": 3, "document_id": "doc2_hash"},
        ),
    ]

    index = build_workspace_index(chunks)
    assert index is not None

    # Search for RAG concept
    results = index.similarity_search("retrieval augmented generation", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "doc2.pdf"
    assert results[0].metadata["page"] == 3
    assert results[0].metadata["document_id"] == "doc2_hash"
