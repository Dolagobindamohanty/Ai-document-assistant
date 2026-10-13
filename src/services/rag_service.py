from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import config
from src.context_builder import build_context
from src.llm import generate_answer
from src.models.schemas import AnswerResult, SourceReference
from src.prompt_builder import build_rag_prompt
from src.utils.errors import InsufficientEvidenceError, RAGError


def retrieve_candidates(
    vector_store: FAISS,
    query: str,
    k: int | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
) -> list[Document]:
    """Retrieve diverse candidate documents using Maximal Marginal Relevance (MMR)."""
    return vector_store.max_marginal_relevance_search(
        query,
        k=k or config.RETRIEVAL_K,
        fetch_k=fetch_k or config.RETRIEVAL_FETCH_K,
        lambda_mult=lambda_mult or config.MMR_LAMBDA,
    )


def filter_by_relevance_distance(
    vector_store: FAISS,
    query: str,
    max_distance: float | None = None,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """
    Score-based relevance filter: Discard chunks whose L2 Euclidean distance exceeds threshold.
    """
    threshold = max_distance or config.MAX_DISTANCE
    top_k = k or config.RETRIEVAL_K

    scored_results = vector_store.similarity_search_with_score(query, k=top_k)
    accepted = []

    for doc, score in scored_results:
        float_score = float(score)
        if float_score <= threshold:
            accepted.append((doc, float_score))

    return accepted


def answer_document_query(
    vector_store: FAISS,
    query: str,
    max_distance: float | None = None,
    client: Any = None,
) -> AnswerResult:
    """
    Complete evidence-grounded Q&A pipeline:
    1. Relevance distance scoring to filter weak evidence.
    2. MMR retrieval for semantic diversity.
    3. Safe refusal if zero chunks pass threshold.
    4. Strict grounding prompt construction with context character budget.
    5. Resilient LLM inference with citations and source references.
    """
    if not query or not query.strip():
        return AnswerResult(
            answer="Please enter a valid question.",
            status="failed",
            warnings=["Empty query provided."],
        )

    # 1. Score & filter for relevance
    relevant_scored = filter_by_relevance_distance(
        vector_store=vector_store,
        query=query,
        max_distance=max_distance,
    )

    if not relevant_scored:
        # Zero chunks met the relevance threshold: Abstain immediately
        refusal_text = "I don't know based on the provided documents. No sufficiently relevant information was found."
        return AnswerResult(
            answer=refusal_text,
            sources=[],
            retrieved_count=0,
            accepted_count=0,
            status="insufficient_evidence",
            warnings=["Query exceeded maximum relevance distance threshold."],
        )

    # Extract accepted documents and create SourceReference list
    accepted_docs = [doc for doc, _ in relevant_scored]
    sources: list[SourceReference] = []
    seen_refs = set()

    for doc, score in relevant_scored:
        src = doc.metadata.get("source", "Unknown")
        pg = doc.metadata.get("page", 1)
        key = (src, pg)
        if key not in seen_refs:
            seen_refs.add(key)
            snippet = doc.page_content[:150] + ("..." if len(doc.page_content) > 150 else "")
            sources.append(
                SourceReference(
                    source=src,
                    page=pg,
                    snippet=snippet,
                    score=score,
                )
            )

    # 2. Build bounded context
    context = build_context(accepted_docs)
    if len(context) > config.MAX_CONTEXT_CHARS:
        context = context[: config.MAX_CONTEXT_CHARS] + "\n...[Context truncated to fit budget]"

    # 3. Build strict prompt
    prompt = build_rag_prompt(query=query, context=context)

    # 4. Generate answer
    try:
        answer_text, model_used = generate_answer(
            prompt=prompt,
            client=client,
            return_model=True,
        )

        return AnswerResult(
            answer=answer_text,
            sources=sources,
            retrieved_count=len(relevant_scored),
            accepted_count=len(accepted_docs),
            status="success",
            model_used=model_used,
        )

    except RAGError as rag_err:
        return AnswerResult(
            answer=rag_err.user_message,
            sources=sources,
            retrieved_count=len(relevant_scored),
            accepted_count=len(accepted_docs),
            status="failed",
            warnings=[str(rag_err)],
        )
    except Exception as exc:
        return AnswerResult(
            answer=f"Generation failed: {exc}",
            sources=sources,
            retrieved_count=len(relevant_scored),
            accepted_count=len(accepted_docs),
            status="failed",
            warnings=[str(exc)],
        )
