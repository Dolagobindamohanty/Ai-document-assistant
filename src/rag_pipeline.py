from src.retriever import load_vector_store, retrieve_documents
from src.relevance_checker import filter_documents
from src.context_builder import build_context, get_unique_sources
from src.prompt_builder import build_rag_prompt
from src.llm import generate_answer
# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5
FETCH_K = 10
LAMBDA_MULT = 0.7
MAX_DISTANCE = 1.4



def run_rag_cli(query=None):
    # ============================================================
    # 1. LOAD FAISS VECTOR STORE
    # ============================================================
    print("Loading FAISS vector store...")
    vector_store = load_vector_store()
    print("FAISS database loaded successfully!")

    # ============================================================
    # 2. GET USER QUESTION
    # ============================================================
    if query is None:
        query = input("\nAsk a question: ")

    # ============================================================
    # 3. RETRIEVE DOCUMENTS USING MMR
    # ============================================================
    retrieved_documents = retrieve_documents(
        vector_store=vector_store,
        query=query,
        k=TOP_K,
        fetch_k=FETCH_K,
        lambda_mult=LAMBDA_MULT
    )

    print("\n" + "=" * 70)
    print("MMR RETRIEVAL")
    print("=" * 70)
    print(f"Retrieved {len(retrieved_documents)} candidate document(s).")

    # ============================================================
    # 4. CHECK RELEVANCE
    # ============================================================
    relevant_results = filter_documents(
        vector_store=vector_store,
        query=query,
        max_distance=MAX_DISTANCE,
        k=TOP_K
    )

    # Extract only the documents
    relevant_documents = [
        document
        for document, score in relevant_results
    ]

    # ============================================================
    # 5. CHECK WHETHER ANYTHING IS RELEVANT
    # ============================================================
    if not relevant_documents:
        print("\n" + "=" * 70)
        print("RELEVANCE CHECK")
        print("=" * 70)
        print("\nNo sufficiently relevant information was found in the knowledge base.")
        refusal = "I don't know based on the provided documents."
        print("\nRAG ANSWER:")
        print(refusal)
        return refusal

    print("\n" + "=" * 70)
    print("RELEVANCE CHECK")
    print("=" * 70)
    print(f"\nAccepted {len(relevant_documents)} relevant document(s).")

    # ========================================================
    # 6. BUILD CONTEXT
    # ========================================================
    context = build_context(relevant_documents)

    # ========================================================
    # 7. BUILD PROMPT
    # ========================================================
    prompt = build_rag_prompt(query=query, context=context)

    # ========================================================
    # 8. DISPLAY CONTEXT
    # ========================================================
    print("\n" + "=" * 70)
    print("CONTEXT")
    print("=" * 70)
    print(context)

    # ========================================================
    # 9. DISPLAY PROMPT
    # ========================================================
    print("\n" + "=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(prompt)

    # ========================================================
    # 10. DISPLAY SOURCES
    # ========================================================
    sources = get_unique_sources(relevant_documents)
    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for source in sources:
        print("-", source)

    # ========================================================
    # 11. GENERATE FINAL ANSWER USING GEMINI
    # ========================================================
    print("\n" + "=" * 70)
    print("GEMINI GENERATION")
    print("=" * 70)

    answer = generate_answer(prompt)
    print("\nRAG ANSWER:")
    print(answer)

    print("""
    Retriever             : SUCCESS
    MMR                   : SUCCESS
    Relevance Checker     : SUCCESS
    Context Builder       : SUCCESS
    Prompt Builder        : SUCCESS
    LLM Generation        : SUCCESS
    """)
    return answer


if __name__ == "__main__":
    run_rag_cli()