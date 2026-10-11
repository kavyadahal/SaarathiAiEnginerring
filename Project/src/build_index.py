
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB collection
COLLECTION_NAME = "Project"

BATCH_SIZE = 64


def main():
    # 1. Check that chunks exist
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_FILE}. "
            "Run chunk_pages.py first."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json contains no chunks.")

    # 2. Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Open a persistent ChromaDB database
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    # Rebuild the collection so repeated runs do not
    # accidentally leave old or duplicate documents.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted previous collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Embed and store chunks in batches
    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [item["text"] for item in batch]
        ids = [item["chunk_id"] for item in batch]

        metadatas = [
            {
                "source": item["source"],
                "page": int(item["page"]),
                "chunk_index": int(item["chunk_index"])
            }
            for item in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Indexed {min(start + len(batch), total)}/{total}")

    # 5. Verify the index
    print("\n--- Index Summary ---")
    print(f"Chunks in JSON: {total}")
    print(f"Chunks in ChromaDB: {collection.count()}")
    print(f"Database location: {DB_DIR}")

    # 6. Run a sample semantic search
    question = "What does the report say about economic growth?"

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\n--- Sample Search Results ---")
    print(f"Question: {question}")

    for rank, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):
        print(f"\nResult {rank}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {document[:400]}")


if __name__ == "__main__":
    main()
