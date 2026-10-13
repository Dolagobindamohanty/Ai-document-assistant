from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import config

_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Lazy initialize and cache the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
    return _embedding_model


def build_workspace_index(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """
    Build an isolated in-memory FAISS vector index for the active workspace.
    Guarantees zero mutation of disk-based legacy faiss_db/.
    """
    if not chunks:
        raise ValueError("Cannot build index: zero usable text chunks provided.")

    embeddings = embedding_model or get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def reset_workspace_index() -> None:
    """
    Explicit helper to reset in-memory workspace references.
    """
    pass
