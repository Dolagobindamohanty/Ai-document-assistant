def filter_documents(
    vector_store,
    query,
    max_distance=1.4,
    k=5
):
    """
    Retrieve documents with similarity scores
    and keep only sufficiently relevant documents.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    relevant_documents = []

    for document, score in results:

        if score <= max_distance:

            relevant_documents.append(
                (document, float(score))
            )

    return relevant_documents