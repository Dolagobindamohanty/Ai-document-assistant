from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DocumentPage:
    """Represents an extracted single page from a document."""
    text: str
    source: str
    page: int
    document_id: str = ""
    char_count: int = 0
    is_low_text: bool = False

    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.text.strip())


@dataclass
class SourceReference:
    """Represents a verifiable citation to a specific page of a document."""
    source: str
    page: int
    snippet: str | None = None
    score: float | None = None

    def to_citation_label(self) -> str:
        return f"[{self.source}, p. {self.page}]"


@dataclass
class AnswerResult:
    """Standardized response from RAG Question-Answering service."""
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    retrieved_count: int = 0
    accepted_count: int = 0
    status: Literal["success", "insufficient_evidence", "failed"] = "success"
    model_used: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class SummaryResult:
    """Standardized response from full-document summarization service."""
    summary: str
    sources: list[SourceReference] = field(default_factory=list)
    pages_total: int = 0
    pages_processed: int = 0
    batches_total: int = 0
    batches_completed: int = 0
    status: Literal["complete", "partial", "failed"] = "complete"
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuizItem:
    """Represents an individual multiple-choice question."""
    question: str
    options: list[str]
    correct_answer_index: int
    explanation: str = ""
    source: SourceReference | None = None

    def is_valid(self) -> bool:
        """Check if question has at least 2 options and a valid answer index."""
        return (
            len(self.options) >= 2
            and 0 <= self.correct_answer_index < len(self.options)
            and bool(self.question.strip())
        )


@dataclass
class QuizResult:
    """Standardized response from study quiz generation service."""
    items: list[QuizItem] = field(default_factory=list)
    topic: str = ""
    status: Literal["success", "partial", "failed"] = "success"
    raw_response: str = ""
    warnings: list[str] = field(default_factory=list)
