from src.models.schemas import AnswerResult, QuizResult, SummaryResult


def export_answer_to_markdown(result: AnswerResult) -> str:
    """Format AnswerResult as clean GitHub markdown."""
    lines = [
        "# AI Document Assistant - Question Answer",
        "",
        result.answer,
        "",
        "---",
        "## Referenced Sources",
    ]
    if result.sources:
        for s in result.sources:
            lines.append(f"- **{s.source}** (Page {s.page})")
            if s.snippet:
                lines.append(f"  > *\"{s.snippet}\"*")
    else:
        lines.append("- *No specific sources cited.*")

    if result.model_used:
        lines.append("")
        lines.append(f"*Generated using model: {result.model_used}*")

    return "\n".join(lines)


def export_summary_to_markdown(result: SummaryResult) -> str:
    """Format SummaryResult as clean markdown with coverage manifest."""
    lines = [
        "# AI Document Assistant - Document Summary",
        f"**Status:** {result.status.upper()} | **Pages Processed:** {result.pages_processed}/{result.pages_total} | **Batches Completed:** {result.batches_completed}/{result.batches_total}",
        "",
        result.summary,
        "",
        "---",
        "## Source Documents Covered",
    ]
    for s in result.sources:
        lines.append(f"- {s.source} — Page {s.page}")

    if result.warnings:
        lines.append("")
        lines.append("## Processing Notes & Warnings")
        for w in result.warnings:
            lines.append(f"- ⚠️ {w}")

    return "\n".join(lines)


def export_quiz_to_markdown(result: QuizResult) -> str:
    """Format QuizResult as clean markdown test sheet with answer key."""
    lines = [
        f"# Document Quiz: {result.topic}",
        f"**Total Questions:** {len(result.items)}",
        "",
    ]
    for i, item in enumerate(result.items, start=1):
        lines.append(f"### Question {i}: {item.question}")
        for opt_idx, opt in enumerate(item.options):
            letter = chr(ord("A") + opt_idx)
            lines.append(f"- **{letter})** {opt}")
        lines.append("")

    lines.append("---")
    lines.append("## Answer Key & Explanations")
    for i, item in enumerate(result.items, start=1):
        letter = chr(ord("A") + item.correct_answer_index)
        ans_text = item.options[item.correct_answer_index]
        lines.append(f"**Q{i} Answer:** {letter}) {ans_text}")
        if item.explanation:
            lines.append(f"*Explanation:* {item.explanation}")
        lines.append("")

    return "\n".join(lines)
