class RAGError(Exception):
    """Base exception class for all RAG system errors."""
    def __init__(self, message: str, user_message: str | None = None):
        super().__init__(message)
        self.user_message = user_message or message


class ModelUnavailableError(RAGError):
    """Raised when the LLM service suffers temporary capacity/503/500 downtime."""
    def __init__(self, message: str = "The AI service is temporarily unavailable due to server load."):
        super().__init__(
            message=message,
            user_message="The AI service is temporarily unavailable. Your documents are still indexed; please retry in a moment."
        )


class QuotaExhaustedError(RAGError):
    """Raised when API quota or rate limits (429) are reached."""
    def __init__(self, message: str = "API quota or rate limit exceeded."):
        super().__init__(
            message=message,
            user_message="The model provider's quota or rate limit was reached. Check provider usage or retry after the limit resets."
        )


class AuthenticationError(RAGError):
    """Raised when the API key or authentication credentials fail (401/403)."""
    def __init__(self, message: str = "Invalid or missing API key."):
        super().__init__(
            message=message,
            user_message="The configured model API credentials could not be validated. Please verify your GEMINI_API_KEY."
        )


class ModelNotFoundError(RAGError):
    """Raised when a configured model name does not exist or is deprecated."""
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' was not found or is not supported.",
            user_message=f"The requested AI model ('{model_name}') is unavailable. Please check your model configuration."
        )


class InvalidRequestError(RAGError):
    """Raised when the request prompt is malformed or rejected."""
    def __init__(self, message: str = "Invalid request structure or payload."):
        super().__init__(
            message=message,
            user_message="The request could not be processed due to invalid parameters or formatting."
        )


class EmptyResponseError(RAGError):
    """Raised when the LLM returns an empty text response."""
    def __init__(self, model_name: str = "LLM"):
        super().__init__(
            message=f"Model {model_name} returned an empty response.",
            user_message="The AI service returned an empty response. Please try rephrasing your question."
        )


class InvalidPDFError(RAGError):
    """Raised when uploaded file is corrupt, empty, or not a valid PDF."""
    def __init__(self, filename: str, reason: str = "Corrupt or unreadable PDF"):
        super().__init__(
            message=f"File '{filename}' could not be processed: {reason}",
            user_message=f"Could not read '{filename}': {reason}. Please check the file and try again."
        )


class InsufficientEvidenceError(RAGError):
    """Raised when no retrieved chunks meet the relevance threshold."""
    def __init__(self, message: str = "No sufficiently relevant information found in the documents."):
        super().__init__(
            message=message,
            user_message="I don't know based on the provided documents. No sufficiently relevant information was found."
        )


def redact_secrets(text: str, secret: str | None = None) -> str:
    """Helper to redact API keys from logs and exception messages."""
    if not text:
        return ""
    if secret and len(secret) > 6:
        text = text.replace(secret, "[REDACTED_API_KEY]")
    # Also catch common API key formats (e.g. AIzaSy...)
    import re
    text = re.sub(r'AIzaSy[A-Za-z0-9_-]{33}', '[REDACTED_API_KEY]', text)
    return text


def classify_gemini_error(error: Exception, secret_to_redact: str | None = None) -> RAGError:
    """
    Classify raw API/SDK exceptions into normalized RAGError subclasses.
    """
    if isinstance(error, RAGError):
        return error

    raw_message = redact_secrets(str(error), secret_to_redact)
    msg = raw_message.lower()

    # Quota / Rate limit (429)
    if "429" in msg or "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg:
        return QuotaExhaustedError(raw_message)

    # Authentication / Permissions (401 / 403)
    if "401" in msg or "403" in msg or "api_key" in msg or "permission" in msg or "unauthenticated" in msg:
        return AuthenticationError(raw_message)

    # Model not found (404)
    if "404" in msg or "not found" in msg or "is not supported" in msg:
        return ModelNotFoundError(raw_message)

    # Temporary server outage / 503 / 500 / Overloaded
    if any(term in msg for term in ["503", "unavailable", "overloaded", "500", "internal error", "deadline_exceeded", "server error"]):
        return ModelUnavailableError(raw_message)

    # Invalid request (400)
    if "400" in msg or "bad request" in msg or "invalid argument" in msg:
        return InvalidRequestError(raw_message)

    # Fallback to generic RAGError
    return RAGError(f"Unexpected error: {raw_message}", user_message=f"An unexpected error occurred: {raw_message}")
