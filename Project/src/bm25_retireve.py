
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"


def tokenize(text):
    """
    Convert text into lowercase word tokens.
    Example: 'Nepal's Economic Growth'
    becomes ['nepal', 's', 'economic', 'growth'].
    """
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def main():
    # 2. Load the existing chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_FILE}. "
            "Run chunk_pages.py first."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json contains no chunks.")

    # 3. Tokenize all chunk text
    print("Tokenizing document chunks...")
    tokenized_chunks = [
        tokenize(item["text"]) for item in chunks
    ]

    # 4. Build the BM25 search index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_chunks)

    # 5. Search for a test question
    question = (
        "What was the estimated economic growth rate "
        "of Nepal in fiscal year 2023/24?"
    )

    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    # Sort chunk indexes from highest to lowest BM25 score
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )[:5]

    # 6. Print the top five results
    print("\nQuestion:", question)
    print("\n--- BM25 Search Results ---")

    for rank, index in enumerate(ranked_indexes, start=1):
        chunk = chunks[index]

        print(f"\nRank: {rank}")
        print(f"Source: {chunk['source']}")
        print(f"PDF page: {chunk['page']}")
        print(f"BM25 score: {scores[index]:.4f}")
        print("Text:")
        print(chunk["text"][:700])
        print("-" * 60)


if __name__ == "__main__":
    main()
