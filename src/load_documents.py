from pathlib import Path

from langchain_community.document_loaders import TextLoader


documents = []

documents_folder = Path("documents")

for file_path in documents_folder.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


print(f"Total documents loaded: {len(documents)}")

for i, document in enumerate(documents, start=1):

    print(f"\n--- Document {i} ---")
    print("Source:", document.metadata["source"])
    print(document.page_content[:200])