from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 2. LOAD FAISS
# ============================================================

vector_store = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


# ============================================================
# 3. GET QUESTION
# ============================================================

query = input("\nAsk a question: ")


# ============================================================
# 4. SEARCH WITH SCORES
# ============================================================

results = vector_store.similarity_search_with_score(
    query,
    k=5
)


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RELEVANCE RESULTS")
print("=" * 70)


for i, (document, score) in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print("Score:", score)

    print("Source:", document.metadata.get("source"))

    print("Content:")
    print(document.page_content)