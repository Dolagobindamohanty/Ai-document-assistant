from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOAD FAISS DATABASE
# ============================================================

vector_store = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


# ============================================================
# 3. GET USER QUESTION
# ============================================================

query = input("\nAsk a question: ")


# ============================================================
# 4. RETRIEVE DOCUMENTS WITH SCORES
# ============================================================

results = vector_store.similarity_search_with_score(
    query,
    k=5
)


# ============================================================
# 5. SET RELEVANCE THRESHOLD
# ============================================================

MAX_DISTANCE = 1.4


# ============================================================
# 6. FILTER DOCUMENTS
# ============================================================

relevant_documents = []

for document, score in results:

    if score <= MAX_DISTANCE:

        relevant_documents.append(
            (document, score)
        )


# ============================================================
# 7. DISPLAY ALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ALL RETRIEVED RESULTS")
print("=" * 70)

for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")
    print("Score:", round(float(score), 4))
    print("Source:", document.metadata.get("source"))
    print(document.page_content)


# ============================================================
# 8. DISPLAY FILTERED RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RELEVANT DOCUMENTS")
print("=" * 70)

if not relevant_documents:

    print("\nNo sufficiently relevant documents found.")

else:

    for i, (document, score) in enumerate(
        relevant_documents,
        start=1
    ):

        print(f"\n--- Relevant Document {i} ---")

        print("Score:", round(float(score), 4))

        print(
            "Source:",
            document.metadata.get("source")
        )

        print(document.page_content)


# ============================================================
# 9. FINAL DECISION
# ============================================================

print("\n" + "=" * 70)
print("RELEVANCE DECISION")
print("=" * 70)

if relevant_documents:

    print(
        f"ACCEPTED: {len(relevant_documents)} "
        "relevant document(s) found."
    )

else:

    print(
        "REJECTED: No relevant information "
        "was found in the knowledge base."
    )