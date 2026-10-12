import json
import re
from rank_bm25 import BM25Okapi

PAGES_PATH = "data/processed/pages.json"
CHUNKS_PATH = "data/processed/chunks.json"

STOPWORDS = {
    "what", "was", "the", "of", "in", "is", "a", "an", "and", "to", "for",
    "on", "at", "by", "with", "as", "were", "be", "from", "or",
}


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


def clean(text):
    return " ".join(text.split())


def snippet(text, phrase, width=150):
    flat = clean(text)
    i = flat.lower().find(phrase.lower())
    if i == -1:
        return ""
    start = max(0, i - width)
    return flat[start : i + len(phrase) + width]


def show_table(rows, headers):
    if not rows:
        print("(nothing found)")
        return
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    line = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    with open(PAGES_PATH, encoding="utf-8") as f:
        pages = json.load(f)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    # ---------- CHECK 1: how much text does each page have? ----------
    print("=== CHECK 1: Page text length (characters) ===")
    buckets = [
        ("under 50 (nearly empty)", 0, 50),
        ("50 - 299", 50, 300),
        ("300 - 999", 300, 1000),
        ("1000 - 1999", 1000, 2000),
        ("2000 and more", 2000, 10**9),
    ]
    rows = []
    for name, low, high in buckets:
        count = sum(1 for r in pages if low <= len(get_text(r).strip()) < high)
        rows.append([name, count])
    show_table(rows, ["characters on page", "number of pages"])

    shortest = sorted(pages, key=lambda r: len(get_text(r).strip()))[:10]
    print("\n10 shortest pages:")
    show_table(
        [[get_page(r), len(get_text(r).strip())] for r in shortest],
        ["page", "characters"],
    )

    # ---------- CHECK 2: the pages that DO say economic growth / gdp growth ----------
    print("\n=== CHECK 2: Where 'economic growth' and 'gdp growth' appear ===")
    for phrase in ("economic growth", "gdp growth"):
        print(f"\nPhrase: {phrase}")
        rows = []
        for r in pages:
            if phrase in get_text(r).lower():
                rows.append([get_page(r), snippet(get_text(r), phrase)])
        show_table(rows, ["page", "text around the phrase"])

    # ---------- CHECK 3: wider word search ----------
    print("\n=== CHECK 3: How many pages contain each word? ===")
    terms = ["gdp", "gross domestic product", "growth", "grew", "grow", "percent", "estimated"]
    rows = []
    for t in terms:
        rows.append([t, sum(1 for r in pages if t in get_text(r).lower())])
    show_table(rows, ["word", "pages"])

    # ---------- CHECK 4: pages with the most 'gdp' mentions ----------
    print("\n=== CHECK 4: Top 10 pages by number of 'gdp' mentions ===")
    counted = [(get_text(r).lower().count("gdp"), r) for r in pages]
    counted.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for count, r in counted[:10]:
        rows.append([get_page(r), count, clean(get_text(r))[:80]])
    show_table(rows, ["page", "gdp count", "start of page"])

    # ---------- CHECK 5: pages with gdp + (grow/growth) + percent ----------
    print("\n=== CHECK 5: Pages with 'gdp' AND 'grow' AND 'percent' ===")
    rows = []
    for r in pages:
        low = get_text(r).lower()
        if "gdp" in low and "grow" in low and "percent" in low:
            rows.append([get_page(r), snippet(get_text(r), "grow", width=100)])
    print(f"{len(rows)} pages matched (showing up to 15)")
    show_table(rows[:15], ["page", "text around 'grow'"])

    # ---------- CHECK 6: BM25 with a cleaner query ----------
    print("\n=== CHECK 6: BM25 top 10 with a cleaner query ===")
    query = "estimated economic growth rate GDP"
    query_tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
    print(f"Query tokens: {query_tokens}")

    tokenized = [re.findall(r"\w+", get_text(r).lower()) for r in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:10]

    rows = []
    for rank, i in enumerate(ranked, start=1):
        text = get_text(chunks[i])
        rows.append([
            rank,
            i,
            get_page(chunks[i]),
            round(float(scores[i]), 2),
            "yes" if "gdp" in text.lower() else "no",
            clean(text)[:70],
        ])
    show_table(rows, ["rank", "chunk_id", "page", "score", "has gdp?", "preview"])


if __name__ == "__main__":
    main()