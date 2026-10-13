from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load all documents
# --------------------------------------------------

documents = []

documents_folder = Path("documents")

for file_path in documents_folder.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


print(f"Documents loaded: {len(documents)}")


# --------------------------------------------------
# 2. Create text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# --------------------------------------------------
# 3. Split documents
# --------------------------------------------------

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# 4. Display results
# --------------------------------------------------

print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):

    print(f"\n--- Chunk {i} ---")
    print("Source:", chunk.metadata["source"])
    print(chunk.page_content)