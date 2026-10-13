import json
import re
from typing import Any

from src.llm import generate_answer
from src.models.schemas import QuizItem, QuizResult


def generate_study_notes(
    topic: str,
    context: str,
    client: Any = None,
) -> str:
    """
    Generate clear, structured revision notes from document context.
    """
    prompt = f"""
You are an expert study guide creator.

Create comprehensive study and revision notes for the topic: '{topic}'
using ONLY the provided document context.

Format your response with:
# Topic Overview: {topic}
## Key Concepts & Definitions
- [Bullet points with concise definitions]
## Important Mechanisms / How It Works
- [Step-by-step points]
## Critical Facts to Remember
- [Key takeaways]

DOCUMENT CONTEXT:
{context}
"""
    return generate_answer(prompt, client=client)


def parse_and_validate_quiz_json(raw_text: str) -> list[QuizItem]:
    """
    Extract and validate QuizItem structures from model output (JSON or structured text).
    """
    items: list[QuizItem] = []

    # Attempt JSON block extraction
    json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                for obj in parsed:
                    if not isinstance(obj, dict):
                        continue
                    q = obj.get("question", "").strip()
                    options = obj.get("options", [])
                    idx = obj.get("correct_answer_index", 0)
                    exp = obj.get("explanation", "")

                    if isinstance(options, list) and len(options) >= 2:
                        try:
                            idx_int = int(idx)
                            if 0 <= idx_int < len(options) and q:
                                items.append(
                                    QuizItem(
                                        question=q,
                                        options=[str(opt) for opt in options],
                                        correct_answer_index=idx_int,
                                        explanation=str(exp),
                                    )
                                )
                        except (ValueError, TypeError):
                            pass
        except Exception:
            pass

    # If JSON parsing yielded valid items, return them
    if items:
        return items

    # Fallback heuristic parser for text formatted quizzes:
    # Q: ...
    # A) ... B) ... C) ... D) ...
    # Correct: ...
    q_blocks = re.split(r"(?:Question|\d+\.)\s*:\s*|\n(?=\d+\.\s+)", raw_text)
    for block in q_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue

        q_text = lines[0]
        options = []
        correct_idx = 0
        explanation = ""

        for line in lines[1:]:
            opt_match = re.match(r"^[A-D\d][\.\)]\s*(.*)", line, re.IGNORECASE)
            if opt_match:
                options.append(opt_match.group(1).strip())
            elif "correct" in line.lower() or "answer" in line.lower():
                # Detect letter A, B, C, D
                ans_letter = re.search(r"\b([A-D])\b", line, re.IGNORECASE)
                if ans_letter:
                    correct_idx = ord(ans_letter.group(1).upper()) - ord("A")
                explanation = line

        if len(options) >= 2 and 0 <= correct_idx < len(options) and q_text:
            items.append(
                QuizItem(
                    question=q_text,
                    options=options,
                    correct_answer_index=correct_idx,
                    explanation=explanation,
                )
            )

    return items


def generate_quiz(
    topic: str,
    context: str,
    num_questions: int = 5,
    client: Any = None,
) -> QuizResult:
    """
    Generate multiple-choice questions grounded in context and validated against schema.
    """
    prompt = f"""
You are an expert examiner. Generate {num_questions} multiple-choice questions about: '{topic}'.
Use ONLY the provided context.

You MUST respond with a valid JSON array of objects. Do not wrap in markdown tags if possible.
Each object must have these exact keys:
- "question": "Question text here"
- "options": ["Option A text", "Option B text", "Option C text", "Option D text"]
- "correct_answer_index": 0 (integer between 0 and 3 indicating the correct option)
- "explanation": "Brief explanation why this option is correct based on the text"

CONTEXT:
{context}
"""
    raw_response = generate_answer(prompt, client=client)
    items = parse_and_validate_quiz_json(raw_response)

    if not items:
        return QuizResult(
            items=[],
            topic=topic,
            status="failed",
            raw_response=raw_response,
            warnings=["Could not extract valid structured quiz questions from model response."],
        )

    valid_items = [item for item in items if item.is_valid()]
    status = "success" if len(valid_items) >= num_questions else "partial"

    return QuizResult(
        items=valid_items,
        topic=topic,
        status=status,
        raw_response=raw_response,
    )
