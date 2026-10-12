import json
import re

PAGES_PATH = "data/processed/pages.json"

# a sentence that says "grow" / "growth" AND has a number with percent or %
HAS_GROWTH = re.compile(r"\bgrow(th|n|s)?\b", re.IGNORECASE)
HAS_PERCENT = re.compile(r"\d+(\.\d+)?\s*(%|percent)", re.IGNORECASE)
HAS_ECONOMY = re.compile(r"(economy|economic|gdp|gross domestic)", re.IGNORECASE)


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

    # ---------- CHECK 1: word counts after the repair ----------
    print("=== CHECK 1: How many pages contain each word now? ===")
    rows = []
    for word in ["growth", "grow", "grew", "gdp", "estimated", "economic growth", "percent"]:
        rows.append([word, sum(1 for r in pages if word in get_text(r).lower())])
    show_table(rows, ["word", "pages"])

    # ---------- CHECK 2: sentences about economic growth with a percent ----------
    print("\n=== CHECK 2: Sentences with growth + economy + a percent number ===")
    rows = []
    for r in pages:
        flat = " ".join(get_text(r).split())
        sentences = re.split(r"(?<=[.!?])\s+", flat)
        for s in sentences:
            if HAS_GROWTH.search(s) and HAS_PERCENT.search(s) and HAS_ECONOMY.search(s):
                rows.append([get_page(r), s[:220]])
    print(f"{len(rows)} sentences found (showing the first 25)")
    show_table(rows[:25], ["page", "sentence"])

    # ---------- CHECK 3: the Executive Summary opening pages ----------
    print("\n=== CHECK 3: Pages 20 to 22 (Executive Summary), first 600 characters ===")
    for r in pages:
        if get_page(r) in (20, 21, 22):
            flat = " ".join(get_text(r).split())
            print(f"\n--- page {get_page(r)} ---")
            print(flat[:600])


if __name__ == "__main__":
    main()