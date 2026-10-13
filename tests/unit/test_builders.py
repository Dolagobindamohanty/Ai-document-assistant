from langchain_core.documents import Document
from src.context_builder import build_context, get_unique_sources
from src.prompt_builder import build_rag_prompt


def test_build_context_preserves_sources():
    docs = [
        Document(page_content="Machine learning is a subset of AI.", metadata={"source": "ml.pdf", "page": 1}),
        Document(page_content="Deep learning uses neural networks.", metadata={"source": "dl.pdf", "page": 5}),
    ]

    context = build_context(docs)
    assert "ml.pdf" in context
    assert "dl.pdf" in context
    assert "Machine learning is a subset of AI." in context
    assert "Deep learning uses neural networks." in context


def test_get_unique_sources():
    docs = [
        Document(page_content="chunk 1", metadata={"source": "docA.pdf"}),
        Document(page_content="chunk 2", metadata={"source": "docB.pdf"}),
        Document(page_content="chunk 3", metadata={"source": "docA.pdf"}),
    ]

    unique = get_unique_sources(docs)
    assert unique == ["docA.pdf", "docB.pdf"]


def test_build_rag_prompt_contains_grounding_rules():
    prompt = build_rag_prompt(query="What is AI?", context="AI is artificial intelligence.")
    assert "Answer the user's question using ONLY the information" in prompt
    assert 'say: "I don\'t know based on the provided documents."' in prompt
    assert "AI is artificial intelligence." in prompt
    assert "What is AI?" in prompt
