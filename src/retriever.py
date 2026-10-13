from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_vector_store():
    """
    Load the existing FAISS vector database.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_documents(
    vector_store,
    query,
    k=5,
    fetch_k=10,
    lambda_mult=0.7
):
    """
    Retrieve documents using MMR.
    """

    documents = vector_store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    return documents