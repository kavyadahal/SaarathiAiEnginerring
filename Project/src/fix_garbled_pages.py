import json
import os
import re
import shutil

PAGES_PATH = "data/processed/pages.json"
BACKUP_PATH = "data/processed/pages_raw.json"
SHIFT = 29  # "7DEOH" + 29 on each character code = "Table"

COMMON_WORDS = {
    "the", "of", "and", "in", "to", "for", "is", "was", "by", "on", "with", "as",
    "from", "that", "are", "at", "total", "table", "year", "fiscal", "nepal",
    "economic", "growth", "percent", "per", "sector", "million", "billion",
    "rate", "service", "services", "government", "price", "prices", "trade",
    "during", "has", "have", "been", "were", "this", "which", "its", "an", "it",
}


def get_text(record):
    for key in ("text", "chunk", "content"):
        if key in record:
            return record[key]
    raise KeyError(f"No text field found. Keys are: {list(record.keys())}")


def set_text(record, new_text):
    for key in ("text", "chunk", "content"):
        if key in record:
            record[key] = new_text
            return
    raise KeyError("No text field to write to.")


def get_page(record):
    for key in ("page", "page_number", "page_num"):
        if key in record:
            return record[key]
    return "?"


def english_score(text):
    """Fraction of letter-words that are common English words (0.0 to 1.0)."""
    words = re.findall(r"[A-Za-z]{2,}", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in COMMON_WORDS)
    return hits / len(words)


def shift_text(text):
    out = []
    for ch in text:
        code = ord(ch)
        if ch in (" ", "\n", "\t", "\r", "\u2426"):
            out.append(" ")          # real spaces and line breaks stay spaces
        elif ch == "\u00b6":
            out.append("'")          # the apostrophe shows up as a paragraph sign
        elif 3 <= code < 100:
            out.append(chr(code + SHIFT))
        else:
            out.append(ch)
    return "".join(out)


def clean(text):
    return " ".join(text.split())


def show_table(rows, headers):
    if not rows:
        print("(nothing)")
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
    # Always read from the raw backup if it exists, so re-running is safe.
    if not os.path.exists(BACKUP_PATH):
        shutil.copyfile(PAGES_PATH, BACKUP_PATH)
        print(f"Backup saved: {BACKUP_PATH}")
    with open(BACKUP_PATH, encoding="utf-8") as f:
        pages = json.load(f)

    fixed_rows = []
    suspect_rows = []
    ok_count = 0

    for record in pages:
        text = get_text(record)
        original_score = english_score(text)
        shifted = shift_text(text)
        shifted_score = english_score(shifted)

        if shifted_score >= 0.15 and shifted_score > original_score + 0.10:
            set_text(record, clean(shifted))
            fixed_rows.append([
                get_page(record),
                round(original_score, 2),
                round(shifted_score, 2),
                clean(shifted)[:70],
            ])
        else:
            ok_count += 1
            if len(text.strip()) > 300 and original_score < 0.05:
                suspect_rows.append([
                    get_page(record), round(original_score, 2), clean(text)[:70]
                ])

    with open(PAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    show_table(
        [
            ["total pages", len(pages)],
            ["fixed (shift decoded)", len(fixed_rows)],
            ["left unchanged", ok_count],
            ["unchanged but still look odd", len(suspect_rows)],
        ],
        ["group", "pages"],
    )

    print("\n=== FIXED PAGES (first 10): the text should have no '=' between words ===")
    show_table(fixed_rows[:10], ["page", "score before", "score after", "start of fixed text"])

    print("\n=== UNCHANGED BUT ODD (first 15) ===")
    show_table(suspect_rows[:15], ["page", "score", "start of page"])

    print(f"\nSaved the repaired pages to {PAGES_PATH}")


if __name__ == "__main__":
    main()