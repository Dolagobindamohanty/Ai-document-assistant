from typing import Any, Callable

from src.config import config
from src.llm import generate_answer
from src.models.schemas import DocumentPage, SourceReference, SummaryResult
from src.utils.errors import RAGError


def partition_pages_into_batches(
    pages: list[DocumentPage],
    max_chars: int | None = None,
) -> list[tuple[str, list[SourceReference]]]:
    """
    Partition all extracted document pages into context-bounded batches
    while strictly carrying source and page provenance.
    """
    budget = max_chars or config.SUMMARY_BATCH_BUDGET
    batches: list[tuple[str, list[SourceReference]]] = []

    current_chunks: list[str] = []
    current_sources: list[SourceReference] = []
    current_length = 0

    for p in pages:
        text = p.text.strip()
        if not text:
            continue

        ref = SourceReference(source=p.source, page=p.page)
        chunk_text = f"\n[Source: {p.source}, Page: {p.page}]\n{text}\n"

        # Split unusually huge individual pages
        if len(chunk_text) > budget:
            if current_chunks:
                batches.append(("\n".join(current_chunks), list(current_sources)))
                current_chunks = []
                current_sources = []
                current_length = 0

            for start in range(0, len(chunk_text), budget):
                slice_text = chunk_text[start : start + budget]
                batches.append((slice_text, [ref]))
            continue

        if current_length + len(chunk_text) > budget and current_chunks:
            batches.append(("\n".join(current_chunks), list(current_sources)))
            current_chunks = []
            current_sources = []
            current_length = 0

        current_chunks.append(chunk_text)
        current_sources.append(ref)
        current_length += len(chunk_text)

    if current_chunks:
        batches.append(("\n".join(current_chunks), list(current_sources)))

    return batches


def summarize_full_document(
    pages: list[DocumentPage],
    progress_callback: Callable[[float, str], None] | None = None,
    client: Any = None,
    max_chars_per_batch: int | None = None,
) -> SummaryResult:
    """
    Full-document hierarchical Map-Reduce summarization with coverage tracking,
    honest partial-failure classification, and page reference retention.
    """
    if not pages:
        return SummaryResult(
            summary="No document pages provided for summarization.",
            pages_total=0,
            pages_processed=0,
            batches_total=0,
            batches_completed=0,
            status="failed",
            warnings=["Zero pages provided."],
        )

    # Filter out empty pages and track low text
    extractable_pages = [p for p in pages if p.text.strip()]
    warnings: list[str] = []

    low_text_count = sum(1 for p in pages if p.is_low_text)
    if low_text_count > 0:
        warnings.append(f"{low_text_count} page(s) contained little or no selectable text.")

    if not extractable_pages:
        return SummaryResult(
            summary="No readable text found in document pages.",
            pages_total=len(pages),
            pages_processed=0,
            batches_total=0,
            batches_completed=0,
            status="failed",
            warnings=["No extractable text found across all pages."],
        )

    batches = partition_pages_into_batches(extractable_pages, max_chars=max_chars_per_batch)
    total_batches = len(batches)

    partial_summaries: list[str] = []
    failed_batches: list[int] = []
    all_sources: list[SourceReference] = []

    for idx, (batch_text, batch_sources) in enumerate(batches, start=1):
        all_sources.extend(batch_sources)

        if progress_callback:
            progress_callback(
                idx / (total_batches + 2),
                f"Summarizing section {idx} of {total_batches}...",
            )

        map_prompt = f"""
You are an expert document summarizer.

Summarize the following section faithfully.
Preserve important facts, definitions, arguments, numbers, and conclusions.
Do not invent information. Retain page references when available.

DOCUMENT SECTION:
{batch_text}
"""
        try:
            summary = generate_answer(map_prompt, client=client)
            partial_summaries.append(f"SECTION SUMMARY {idx}:\n{summary}")
        except Exception as exc:
            failed_batches.append(idx)
            warnings.append(f"Batch {idx} summarization failed: {exc}")

    if not partial_summaries:
        return SummaryResult(
            summary="Full-document summarization failed completely across all sections.",
            pages_total=len(pages),
            pages_processed=0,
            batches_total=total_batches,
            batches_completed=0,
            status="failed",
            warnings=warnings,
        )

    # REDUCE phase: hierarchically combine summaries
    while len(partial_summaries) > 1:
        combined = []
        for start in range(0, len(partial_summaries), 5):
            group = partial_summaries[start : start + 5]
            if len(group) == 1:
                combined.append(group[0])
                continue

            reduce_prompt = f"""
Combine these section summaries into one coherent intermediate summary.
Preserve key facts, concepts, definitions, and page references without repetition.

SUMMARIES:
{"".join(group)}
"""
            try:
                res = generate_answer(reduce_prompt, client=client)
                combined.append(res)
            except Exception as exc:
                combined.append("\n".join(group))
                warnings.append(f"Reduce step failed: {exc}")

        partial_summaries = combined

    if progress_callback:
        progress_callback(0.95, "Finalizing complete document synthesis...")

    # Final editorial pass
    final_prompt = f"""
Create a comprehensive summary of the entire document using the synthesized notes below.

Organize the result into:
1. Document Overview
2. Main Topics and Sections
3. Important Concepts and Definitions
4. Key Findings and Conclusions
5. Important Facts to Remember

Preserve page references where available.

NOTES:
{partial_summaries[0]}
"""
    try:
        final_summary = generate_answer(final_prompt, client=client)
    except Exception as exc:
        final_summary = partial_summaries[0]
        warnings.append(f"Final formatting pass failed: {exc}")

    if progress_callback:
        progress_callback(1.0, "Summarization complete.")

    # Deduplicate sources
    unique_sources: list[SourceReference] = []
    seen = set()
    for s in all_sources:
        k = (s.source, s.page)
        if k not in seen:
            seen.add(k)
            unique_sources.append(s)

    status = "partial" if failed_batches else "complete"

    return SummaryResult(
        summary=final_summary,
        sources=unique_sources,
        pages_total=len(pages),
        pages_processed=len(extractable_pages),
        batches_total=total_batches,
        batches_completed=total_batches - len(failed_batches),
        status=status,
        warnings=warnings,
    )
