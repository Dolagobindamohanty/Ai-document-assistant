import os
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()


class AppConfig:
    # 1. API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_PRIMARY_MODEL: str = os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.8-flash")
    
    # Fallback models parsed from comma-separated string
    _fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
    GEMINI_FALLBACK_MODELS: list[str] = [m.strip() for m in _fallbacks.split(",") if m.strip()]

    # 2. Embedding & Indexing
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # 3. Retrieval & Relevance
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
    RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "10"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.7"))
    MAX_DISTANCE: float = float(os.getenv("MAX_DISTANCE", "1.90"))
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))

    # 4. Ingestion Limits
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    LOW_TEXT_CHAR_THRESHOLD: int = int(os.getenv("LOW_TEXT_CHAR_THRESHOLD", "50"))

    # 5. Summarization
    SUMMARY_BATCH_BUDGET: int = int(os.getenv("SUMMARY_BATCH_BUDGET", "12000"))

    # 6. Resilience & Retries
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    LLM_RETRY_BASE_SECONDS: float = float(os.getenv("LLM_RETRY_BASE_SECONDS", "2.0"))

    # 7. Privacy
    LOG_DOCUMENT_TEXT: bool = os.getenv("LOG_DOCUMENT_TEXT", "false").lower() in ("true", "1", "yes")

    @classmethod
    def get_all_models(cls) -> list[str]:
        models = [cls.GEMINI_PRIMARY_MODEL]
        for m in cls.GEMINI_FALLBACK_MODELS:
            if m not in models:
                models.append(m)
        return models

    @classmethod
    def validate_api_key(cls) -> bool:
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your-gemini-api-key-here")


config = AppConfig()
