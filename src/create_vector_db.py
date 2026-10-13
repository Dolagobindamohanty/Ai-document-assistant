from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================

documents = []

documents_folder = Path("documents")

for file_path in documents_folder.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


print(f"Documents loaded: {len(documents)}")


# ============================================================
# 2. CHUNK DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE FAISS DATABASE
# ============================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# ============================================================
# 5. SAVE DATABASE
# ============================================================

vector_store.save_local("faiss_db")

print("FAISS database created successfully!")