"""
Load and populate ChromaDB with book documents.

This script:
1. Loads all book text files from data/books/
2. Splits them into chunks using RecursiveCharacterTextSplitter
3. Creates a ChromaDB collection with embeddings
4. Populates the collection with the document chunks
"""

import hashlib
import re
import time
from datetime import datetime

import chromadb
from chromadb import Collection
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyprojroot import here

from odyssey import OllamaEmbeddings


def extract_number_from_file_name(name: str) -> int:
    """Extract the book number from filename like 'book_i (1).txt'."""
    result = re.search(r"\(\d+\)", name)
    if result is None:
        raise ValueError(f"Could not extract number from filename: {name}")

    return int(result.group()[1:-1])


def load_documents() -> list[Document]:
    """Load all book documents from data/books/."""
    books_dir = here("data/books")
    docs: list[Document] = []

    for file in books_dir.glob("*.txt"):
        book_num = extract_number_from_file_name(file.name)

        with open(file, "r") as fp:
            page_content = fp.read()

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "document_id": hashlib.sha256(
                        page_content.encode("utf-8")
                    ).hexdigest(),
                    "source": str(file),
                    "book": page_content.splitlines()[0],
                    "book_num": book_num,
                },
            )
        )

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50, add_start_index=True
    )

    splits = splitter.split_documents(docs)
    print(f"Created {len(splits)} splits.")
    return splits


def hash_document(document: Document) -> str:
    """Generate a unique hash for a document."""
    content = (
        f"{document.metadata['source']}-"
        f"{document.metadata['start_index']}-"
        f"{document.page_content}"
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def add_in_batches(
    collection: Collection,
    splits: list[Document],
    batch_size: int = 32,
    retries: int = 3,
) -> None:
    """Add documents to ChromaDB collection in batches with retry logic."""
    total_batches = (len(splits) + batch_size - 1) // batch_size
    batch_times: list[float] = []

    for batch_num, i in enumerate(range(0, len(splits), batch_size), start=1):
        batch = splits[i : i + batch_size]

        print(f"Adding batch {batch_num}/{total_batches} ({len(batch)} documents)")

        batch_start = time.time()

        for attempt in range(1, retries + 1):
            try:
                collection.add(
                    ids=[x.id for x in batch],  # type: ignore
                    documents=[x.page_content for x in batch],
                    metadatas=[x.metadata for x in batch],
                )

                break

            except Exception as e:
                print(f"Batch {batch_num} failed (attempt {attempt}/{retries}): {e}")

                if attempt == retries:
                    raise

                time.sleep(2**attempt)  # exponential backoff

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(f"\tCompleted in {batch_time:.2f}s")

    if batch_times:
        mean_time = sum(batch_times) / len(batch_times)
        print(f"\nMean time per batch: {mean_time:.2f}s")


def main():
    """Main execution function."""
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("\nSplitting documents...")
    splits = split_documents(docs)

    print("\nHashing documents...")
    for doc in splits:
        doc.id = hash_document(doc)

    print("\nInitialising embeddings...")
    embed_model = "qwen3-embedding:4b"
    embed_fn = OllamaEmbeddings(model=embed_model)

    print("Initialising ChromaDB...")
    client = chromadb.PersistentClient(
        path=here("db/chroma"), settings=Settings(allow_reset=True)
    )
    client.reset()

    print("Creating collection...")
    collection = client.create_collection(
        name="books",
        embedding_function=embed_fn,
        metadata={"embed_model": embed_model, "created_at": str(datetime.now())},
    )

    print("\nPopulating ChromaDB...")
    add_in_batches(
        collection,
        splits,
        batch_size=32,
        retries=3,
    )

    print("\nSuccessfully populated ChromaDB with documents")


if __name__ == "__main__":
    main()
