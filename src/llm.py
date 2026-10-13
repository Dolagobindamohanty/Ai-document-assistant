import os
import random
import time
from typing import Any

from google import genai

from src.config import config
from src.utils.errors import (
    AuthenticationError,
    EmptyResponseError,
    ModelNotFoundError,
    ModelUnavailableError,
    QuotaExhaustedError,
    RAGError,
    classify_gemini_error,
    redact_secrets,
)

_client: Any = None


def get_genai_client(api_key: str | None = None) -> Any:
    """Lazy initialize the GenAI client."""
    global _client
    key = api_key or config.GEMINI_API_KEY
    if not key or key == "your-gemini-api-key-here":
        raise AuthenticationError("GEMINI_API_KEY is missing or invalid in configuration.")
    if _client is None or api_key is not None:
        _client = genai.Client(api_key=key)
    return _client


def generate_answer(
    prompt: str,
    client: Any = None,
    models: list[str] | None = None,
    return_model: bool = False,
) -> str | tuple[str, str]:
    """
    Generate an answer using Google Gemini with resilient multi-model failover,
    exponential backoff with jitter, and normalized error classification.
    """
    candidate_models = models or config.get_all_models()
    max_retries = config.LLM_MAX_RETRIES
    base_backoff = config.LLM_RETRY_BASE_SECONDS

    # Resolve client lazily
    active_client = client or get_genai_client()

    last_classified_error: RAGError | None = None

    for model in candidate_models:
        for attempt in range(max_retries + 1):
            try:
                print(f"[LLM] Trying model: {model} (attempt {attempt + 1}/{max_retries + 1})")

                response = active_client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if response and hasattr(response, "text") and response.text:
                    clean_text = response.text.strip()
                    if clean_text:
                        print(f"[LLM] Answer successfully generated using {model}")
                        if return_model:
                            return clean_text, model
                        return clean_text

                raise EmptyResponseError(model_name=model)

            except Exception as exc:
                classified = classify_gemini_error(exc, secret_to_redact=config.GEMINI_API_KEY)
                last_classified_error = classified

                # If quota exhausted or authentication failed, do NOT retry or burn other models
                if isinstance(classified, (QuotaExhaustedError, AuthenticationError)):
                    print(f"[LLM] Non-retryable error ({classified.__class__.__name__}): {classified.user_message}")
                    raise classified from exc

                # If model not found, try next model in fallback list immediately
                if isinstance(classified, ModelNotFoundError):
                    print(f"[LLM] Model '{model}' not found. Trying next fallback model...")
                    break

                # If temporary server / capacity issue (503 / 500)
                if isinstance(classified, ModelUnavailableError):
                    if attempt < max_retries:
                        jitter = random.uniform(0.1, 0.5)
                        backoff = (base_backoff * (2 ** attempt)) + jitter
                        print(f"[LLM] Temporary error on {model}. Retrying in {backoff:.2f}s...")
                        time.sleep(backoff)
                        continue
                    else:
                        print(f"[LLM] Exhausted retries for {model}. Falling back to next model...")
                        break

                # Empty response: retry once, then try next model
                if isinstance(classified, EmptyResponseError):
                    if attempt < 1:
                        time.sleep(1.0)
                        continue
                    break

                # Unclassified/Other error: fast-fail
                raise classified from exc

    # If all models and retries are exhausted
    if last_classified_error:
        raise last_classified_error

    raise ModelUnavailableError("All configured Gemini models failed to respond.")
