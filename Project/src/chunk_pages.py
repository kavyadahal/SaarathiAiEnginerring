import json
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pages.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"

#Initial chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def split_text(text):
    """Split text into overlapping character-based chunks """

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE , len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Run extract_pdf.py first"
        )

    with open(INPUT_FILE , "r" , encoding = "utf - 8") as file:
        pages = json.load(file)

    all_chunks = []

    for page_data in pages:
        text = page_data["text"].strip()

        if not text:
            continue

        page_chunks = split_text(text)

        for index, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id": f'{page_data["source"]}_p{page_data["page"]}_c{index + 1}',
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_index": index + 1,
                "text": chunk_text
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE , "w" , encoding='utf-8') as file:
        json.dump(
            all_chunks ,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n--- Chunking Summary ---")
    print(f"Pages read: {len(pages)}")
    print(f"Chunks created: {len(all_chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Overlap: {CHUNK_OVERLAP} characters")
    print(f"Output: {OUTPUT_FILE}")

    if all_chunks:
        print("\n--- First Chunk Preview ---")
        print(f"Source: {all_chunks[0]['source']}")
        print(f"Page: {all_chunks[0]['page']}")
        print(all_chunks[0]["text"][:500])

if __name__ == "__main__":
    main()    
    