
import json
from pathlib import Path
import fitz  # PyMuPDF

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "pages.json"


def extract_pdf_pages(pdf_path):
    pages = []

    with fitz.open(pdf_path) as pdf:
        total_pdf_pages = len(pdf)

        print(f"\nReading: {pdf_path.name}")
        print(f"Physical PDF pages: {total_pdf_pages}")

        for page_index in range(total_pdf_pages):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()

            pages.append({
                "source": pdf_path.name,
                "page": page_index + 1,
                "text": text
            })

            if (page_index + 1) % 50 == 0:
                print(f"Processed {page_index + 1}/{total_pdf_pages} pages")

    return pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF found in {RAW_DIR}. "
            "Place your government PDF inside data/raw."
        )

    all_pages = []

    for pdf_path in pdf_files:
        pages = extract_pdf_pages(pdf_path)
        all_pages.extend(pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(all_pages, file, ensure_ascii=False, indent=2)

    # Verify the saved file by loading it again.
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        saved_pages = json.load(file)

    nonempty_pages = sum(
        bool(page["text"].strip()) for page in saved_pages
    )

    print("\n--- Extraction Summary ---")
    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Page records saved: {len(saved_pages)}")
    print(f"Pages containing text: {nonempty_pages}")
    print(f"Pages without extracted text: {len(saved_pages) - nonempty_pages}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
