from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------
# 1. Create embedding model
# ---------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------
# 2. Load saved FAISS database
# ---------------------------------------

vector_store = FAISS.load_local(
    "faiss_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded successfully!")


# ---------------------------------------
# 3. Ask questions
# ---------------------------------------

questions = [
    "What is Machine Learning?",
    "What can Generative AI create?",
    "What is Artificial Intelligence?"
]


# ---------------------------------------
# 4. Search FAISS for each question
# ---------------------------------------

for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    results = vector_store.similarity_search(
        question,
        k=2
    )

    print("\nRetrieved Chunks:")

    for i, document in enumerate(results):

        print(f"\n--- Result {i + 1} ---")

        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)