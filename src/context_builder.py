def build_context(documents):
    """
    Build a clean context string from retrieved documents.

    Each retrieved chunk keeps its source information.
    """

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"[Chunk {i}]\n"
            f"Source: {source}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(context_parts)


def get_unique_sources(documents):
    """
    Return unique source files from retrieved documents.
    """

    sources = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        sources.add(source)

    return sorted(sources)