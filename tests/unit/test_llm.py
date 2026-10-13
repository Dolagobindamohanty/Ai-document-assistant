from unittest.mock import MagicMock
import pytest

from src.llm import generate_answer
from src.utils.errors import (
    AuthenticationError,
    EmptyResponseError,
    ModelNotFoundError,
    ModelUnavailableError,
    QuotaExhaustedError,
    redact_secrets,
)


class MockGenAIResponse:
    def __init__(self, text: str | None):
        self.text = text


def test_llm_t01_retry_success_after_503(monkeypatch):
    """LLM-T01: Primary model fails with 503 once, then succeeds on retry."""
    mock_client = MagicMock()
    # First call raises 503, second call succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("503 Service Unavailable"),
        MockGenAIResponse("Grounded answer from retry"),
    ]

    monkeypatch.setattr("time.sleep", lambda s: None)  # Skip sleep in tests

    answer, model = generate_answer(
        prompt="What is RAG?",
        client=mock_client,
        models=["gemini-3.8-flash"],
        return_model=True,
    )

    assert answer == "Grounded answer from retry"
    assert model == "gemini-3.8-flash"
    assert mock_client.models.generate_content.call_count == 2


def test_llm_t02_fallback_when_primary_exhausted(monkeypatch):
    """LLM-T02: Primary model fails with 503 repeatedly; fallback model succeeds."""
    mock_client = MagicMock()
    # Primary model (3 attempts: 1 initial + 2 retries) fails with 503, secondary succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("503 Server Overloaded"),
        Exception("503 Server Overloaded"),
        Exception("503 Server Overloaded"),
        MockGenAIResponse("Answer from fallback model"),
    ]

    monkeypatch.setattr("time.sleep", lambda s: None)

    answer, model = generate_answer(
        prompt="Explain embeddings",
        client=mock_client,
        models=["gemini-3.8-flash", "gemini-3.6-flash"],
        return_model=True,
    )

    assert answer == "Answer from fallback model"
    assert model == "gemini-3.6-flash"


def test_llm_t03_quota_429_fails_fast(monkeypatch):
    """LLM-T03: Quota 429 raises QuotaExhaustedError without retrying the exhausted model."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 Resource has been exhausted (quota exceeded)")

    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(QuotaExhaustedError):
        generate_answer(
            prompt="Hello",
            client=mock_client,
            models=["gemini-3.8-flash"],
        )

    # Should only call once, no retrying the exhausted model
    assert mock_client.models.generate_content.call_count == 1


def test_llm_t03b_quota_fallback_to_second_model(monkeypatch):
    """LLM-T03b: If model A has exhausted per-model quota, model B with remaining quota succeeds."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("429 Resource has been exhausted (quota exceeded)"),
        MockGenAIResponse("Answer from fallback model with remaining quota"),
    ]

    monkeypatch.setattr("time.sleep", lambda s: None)

    answer, model = generate_answer(
        prompt="Hello",
        client=mock_client,
        models=["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        return_model=True,
    )

    assert answer == "Answer from fallback model with remaining quota"
    assert model == "gemini-3.5-flash-lite"
    assert mock_client.models.generate_content.call_count == 2


def test_llm_t04_auth_fails_fast(monkeypatch):
    """LLM-T04: Auth failure (401/403) raises AuthenticationError immediately."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("403 Forbidden: API key not valid")

    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(AuthenticationError):
        generate_answer(
            prompt="Hello",
            client=mock_client,
            models=["gemini-3.8-flash", "gemini-3.6-flash"],
        )

    assert mock_client.models.generate_content.call_count == 1


def test_llm_t05_model_not_found_falls_back(monkeypatch):
    """LLM-T05: Model not found (404) immediately falls back to next model."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        Exception("404 Model not found"),
        MockGenAIResponse("Success with valid model"),
    ]

    monkeypatch.setattr("time.sleep", lambda s: None)

    answer, model = generate_answer(
        prompt="Hello",
        client=mock_client,
        models=["invalid-model-name", "gemini-3.6-flash"],
        return_model=True,
    )

    assert answer == "Success with valid model"
    assert model == "gemini-3.6-flash"


def test_llm_t06_empty_response_raises(monkeypatch):
    """LLM-T06: Model returning empty text raises EmptyResponseError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MockGenAIResponse("")

    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(EmptyResponseError):
        generate_answer(
            prompt="Hello",
            client=mock_client,
            models=["gemini-3.8-flash"],
        )


def test_llm_t07_secrets_redaction():
    """LLM-T07: Ensure API key is redacted from exception messages and logs."""
    fake_key = "AIzaSyFakeSecretKeyForTesting12345"
    error_msg = f"Failed to connect using key {fake_key} to endpoint."

    redacted = redact_secrets(error_msg, fake_key)
    assert fake_key not in redacted
    assert "[REDACTED_API_KEY]" in redacted
