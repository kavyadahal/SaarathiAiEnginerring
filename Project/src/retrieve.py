
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "Project"


# 2. Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# 3. Connect to the existing ChromaDB database
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


def retrieve(question, top_k=5):
    """
    Retrieve the most relevant document chunks for a question.

    Returns a list of chunks with their source, page,
    similarity distance, and text.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Convert the question into an embedding vector
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve the nearest chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        retrieved_chunks.append({
            "rank": rank,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
            "text": document
        })

    return retrieved_chunks


def main():
    # Test with a question about the report
    question = (
        "What was the estimated economic growth rate "
        "of Nepal in fiscal year 2023/24?"
    )

    print("\nQuestion:", question)
    print("\n--- Retrieved Results ---")

    results = retrieve(question, top_k=5)

    for item in results:
        print(f"\nRank: {item['rank']}")
        print(f"Source: {item['source']}")
        print(f"PDF page: {item['page']}")
        print(f"Cosine distance: {item['distance']:.4f}")
        print("Text:")
        print(item["text"][:700])
        print("-" * 60)


if __name__ == "__main__":
    main()
