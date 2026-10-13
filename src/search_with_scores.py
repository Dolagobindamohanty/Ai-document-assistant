from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Create the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 2. Load FAISS
vector_store = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


# 3. Questions to test
questions = [
    "What is Machine Learning?",
    "What can Generative AI create?",
    "What is Retrieval Augmented Generation?",
    "What is Artificial Intelligence?"
]


# 4. Search with similarity scores
for query in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", query)
    print("=" * 60)

    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    for i, (document, score) in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print("Score:", score)
        print("Content:")
        print(document.page_content)
        print("Metadata:")
        print(document.metadata)