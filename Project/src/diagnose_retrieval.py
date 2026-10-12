import json
import re
from rank_bm25 import BM25Okapi

PAGES_PATH = "data/processed/pages.json"
CHUNKS_PATH = "data/processed/chunks.json"
QUESTION = "What was the estimated economic growth rate of Nepal in fiscal year 2023/24?"

PHRASES = [
    "economic growth",
    "gdp growth",
    "growth rate",
    "2023/24",
    "fiscal year",
]

# "growth" followed within ~80 characters by a number with a percent sign
GROWTH_PERCENT = re.compile(r"growth.{0,80}?\d+(\.\d+)?\s*(%|percent)", re.IGNORECASE | re.DOTALL)


def get_text(record):
    for key in ("text", "chunk", "content"):
        if key in record:
            return record[key]
    raise KeyError(f"No text field found. Keys are: {list(record.keys())}")


def get_page(record):
    for key in ("page", "page_number", "page_num"):
        if key in record:
            return record[key]
    return "?"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def show_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    with open(PAGES_PATH, encoding="utf-8") as f:
        pages = json.load(f)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Pages loaded:  {len(pages)}")
    print(f"Chunks loaded: {len(chunks)}")

    # ---------- CHECK 1: phrase counts in pages vs chunks ----------
    print("\n=== CHECK 1: How many pages / chunks contain each phrase? ===")
    rows = []
    for phrase in PHRASES:
        p = phrase.lower()
        in_pages = sum(1 for r in pages if p in get_text(r).lower())
        in_chunks = sum(1 for r in chunks if p in get_text(r).lower())
        rows.append([phrase, in_pages, in_chunks])
    show_table(rows, ["phrase", "pages", "chunks"])

    # ---------- CHECK 2: "growth ... number %" pattern ----------
    print("\n=== CHECK 2: Pages where 'growth' is followed by a percent number ===")
    page_hits = []
    for r in pages:
        m = GROWTH_PERCENT.search(get_text(r))
        if m:
            snippet = " ".join(m.group(0).split())[:100]
            page_hits.append([get_page(r), snippet])
    if page_hits:
        show_table(page_hits[:25], ["page", "snippet"])
        print(f"\n({len(page_hits)} pages matched, showing up to 25)")
    else:
        print("No page matched. The statistic may be missing or garbled in extraction.")

    print("\n=== CHECK 2b: Same pattern, but in CHUNKS ===")
    chunk_hit_ids = set()
    for i, r in enumerate(chunks):
        if GROWTH_PERCENT.search(get_text(r)):
            chunk_hit_ids.add(i)
    print(f"{len(chunk_hit_ids)} chunks matched.")
    print("(If pages matched in CHECK 2 but chunks did not, chunking split it.)")

    # ---------- CHECK 3: BM25 top 10 ----------
    print("\n=== CHECK 3: BM25 top 10 for the question ===")
    tokenized_chunks = [tokenize(get_text(r)) for r in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = tokenize(QUESTION)
    print(f"Query tokens: {query_tokens}")

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    rows = []
    for rank, i in enumerate(ranked[:10], start=1):
        has_pattern = "YES" if i in chunk_hit_ids else "no"
        preview = " ".join(get_text(chunks[i]).split())[:70]
        rows.append([rank, i, get_page(chunks[i]), round(float(scores[i]), 2), has_pattern, preview])
    show_table(rows, ["rank", "chunk_id", "page", "score", "growth%?", "preview"])

    # ---------- Where do the pattern-matching chunks rank? ----------
    print("\n=== Where do the 'growth + percent' chunks rank in BM25? ===")
    position = {i: rank for rank, i in enumerate(ranked, start=1)}
    rows = []
    for i in sorted(chunk_hit_ids, key=lambda c: position[c])[:15]:
        rows.append([position[i], i, get_page(chunks[i])])
    if rows:
        show_table(rows, ["bm25_rank", "chunk_id", "page"])
    else:
        print("None found.")


if __name__ == "__main__":
    main()