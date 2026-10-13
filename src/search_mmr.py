from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
vector_store = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "What is Artificial Intelligence?"

# Normal similarity search
similarity_results = vector_store.similarity_search(
    query,
    k=3
)

# MMR search
mmr_results = vector_store.max_marginal_relevance_search(
    query,
    k=3,
    fetch_k=3,
    lambda_mult=0.5
)


print("=" * 60)
print("NORMAL SIMILARITY SEARCH")
print("=" * 60)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)


print("\n\n" + "=" * 60)
print("MMR SEARCH")
print("=" * 60)

for i, doc in enumerate(mmr_results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)