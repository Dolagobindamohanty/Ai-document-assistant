from unittest.mock import MagicMock
from src.models.schemas import AnswerResult, QuizItem, QuizResult, SourceReference, SummaryResult
from src.services.export_service import export_answer_to_markdown, export_quiz_to_markdown, export_summary_to_markdown
from src.services.study_service import generate_quiz, generate_study_notes, parse_and_validate_quiz_json


class MockGenAIResponse:
    def __init__(self, text: str):
        self.text = text


def test_study_t01_generate_study_notes():
    """STUDY-T01: Generate notes from document context."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MockGenAIResponse(
        "# Topic Overview: Machine Learning\n## Key Concepts\n- Supervised learning uses labeled data."
    )

    notes = generate_study_notes(
        topic="Machine Learning",
        context="Supervised learning uses labeled data to train algorithms.",
        client=mock_client,
    )

    assert "Topic Overview: Machine Learning" in notes
    assert "Supervised learning" in notes


def test_study_t02_quiz_json_parsing_and_validation():
    """STUDY-T02: Valid JSON array parses into QuizItems with validated options and answer index."""
    raw_json = """
[
  {
    "question": "What does FAISS stand for?",
    "options": ["Facebook AI Similarity Search", "Fast AI Simple Search", "File Access Index System", "None"],
    "correct_answer_index": 0,
    "explanation": "FAISS is Facebook AI Similarity Search library."
  },
  {
    "question": "Invalid item with out of range index",
    "options": ["A", "B"],
    "correct_answer_index": 5
  }
]
"""
    items = parse_and_validate_quiz_json(raw_json)
    assert len(items) == 1
    assert items[0].question == "What does FAISS stand for?"
    assert items[0].correct_answer_index == 0
    assert len(items[0].options) == 4
    assert items[0].is_valid() is True


def test_study_t03_quiz_text_fallback_parsing():
    """STUDY-T03: If model outputs formatted text instead of JSON, parser recovers quiz items."""
    raw_text = """
1. What is an embedding vector?
A) A dense numerical representation of text semantics
B) A database table
C) A hard drive partition
D) A programming language
Correct Answer: A
"""
    items = parse_and_validate_quiz_json(raw_text)
    assert len(items) == 1
    assert "What is an embedding vector?" in items[0].question
    assert items[0].correct_answer_index == 0
    assert len(items[0].options) == 4


def test_ux_t05_export_markdown_formatting():
    """UX-T05: Exports contain correct headers, citations, coverage, and answer keys."""
    # Answer export
    ans = AnswerResult(
        answer="RAG grounds LLMs.",
        sources=[SourceReference(source="doc.pdf", page=3, snippet="snippet text")],
        model_used="gemini-3.8-flash",
    )
    ans_md = export_answer_to_markdown(ans)
    assert "# AI Document Assistant - Question Answer" in ans_md
    assert "doc.pdf" in ans_md
    assert "Page 3" in ans_md

    # Summary export
    summ = SummaryResult(
        summary="Document summary text",
        sources=[SourceReference(source="doc.pdf", page=1)],
        pages_total=5,
        pages_processed=5,
        batches_total=1,
        batches_completed=1,
        status="complete",
    )
    summ_md = export_summary_to_markdown(summ)
    assert "**Pages Processed:** 5/5" in summ_md
    assert "**Status:** COMPLETE" in summ_md

    # Quiz export
    quiz = QuizResult(
        items=[
            QuizItem(
                question="What is MMR?",
                options=["Maximal Marginal Relevance", "Minimum Match Ratio"],
                correct_answer_index=0,
                explanation="MMR balances relevance and diversity.",
            )
        ],
        topic="Retrieval",
    )
    quiz_md = export_quiz_to_markdown(quiz)
    assert "Document Quiz: Retrieval" in quiz_md
    assert "Answer Key & Explanations" in quiz_md
    assert "**Q1 Answer:** A) Maximal Marginal Relevance" in quiz_md
