import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "Project"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EVIDENCE_PAGES = {4, 20, 35}   # pages you verified in the PDF
ANSWER_TEXT = "3.87"           # the number the correct chunk must contain
TOP_K = 20

QUESTIONS = [
    "What was the estimated economic growth rate of Nepal in fiscal year 2023/24?",
    "estimated economic growth rate Nepal fiscal year 2023/24",
    "By how much is the Nepali economy estimated to expand this year?",
]

STOPWORDS = {
    "what", "was", "the", "of", "in", "is", "a", "an", "and", "to", "for",
    "on", "at", "by", "with", "as", "were", "be", "from", "or", "how", "much",
    "this", "that", "are",
}


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def is_hit(chunk_text, page):
    return int(page) in EVIDENCE_PAGES and ANSWER_TEXT in chunk_text


def show_table(rows, headers):
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    line = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def first_hit_rank(ranked_items):
    """ranked_items: list of (page, text). Returns rank of first hit, or None."""
    for rank, (page, text) in enumerate(ranked_items, start=1):
        if is_hit(text, page):
            return rank
    return None


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    print(f"Chunks loaded: {len(chunks)}")

    hit_count = sum(1 for c in chunks if is_hit(c["text"], c["page"]))
    print(f"Chunks that count as correct evidence: {hit_count}")
    if hit_count == 0:
        print("No correct chunk exists. Re-run fix_garbled_pages.py and chunk_pages.py.")
        return

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"ChromaDB chunks: {collection.count()}  (must equal chunks above)")

    summary_rows = []

    for question in QUESTIONS:
        print(f"\n==================== {question}")

        # --- BM25 ---
        scores = bm25.get_scores(tokenize(question))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        bm25_items = [(chunks[i]["page"], chunks[i]["text"]) for i in order]
        bm25_rank = first_hit_rank(bm25_items)

        # --- Dense ---
        embedding = model.encode(question, normalize_embeddings=True).tolist()
        result = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        dense_items = [
            (meta["page"], doc)
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]
        dense_rank = first_hit_rank(dense_items)

        for name, items in (("BM25", bm25_items), ("DENSE", dense_items)):
            print(f"\n{name} top 5:")
            rows = []
            for rank, (page, text) in enumerate(items[:5], start=1):
                rows.append([
                    rank,
                    page,
                    "YES" if is_hit(text, page) else "no",
                    " ".join(text.split())[:70],
                ])
            show_table(rows, ["rank", "page", "correct?", "preview"])

        summary_rows.append([
            question[:45],
            bm25_rank if bm25_rank else f"not in top {TOP_K}",
            dense_rank if dense_rank else f"not in top {TOP_K}",
        ])

    print("\n==================== SUMMARY: rank of the first correct chunk")
    show_table(summary_rows, ["question", "BM25 rank", "dense rank"])
    print("\nRecall@5 counts a question as a success if the rank is 5 or lower.")


if __name__ == "__main__":
    main()