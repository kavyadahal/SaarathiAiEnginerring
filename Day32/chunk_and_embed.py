"""
Day 31 · Exercise 1 — chunking: cut a document into searchable pieces
On Day 25 our RAG retrieved whole notes. Real documents are long, so we split
them into small overlapping CHUNKS, then embed each chunk into a vector. This
file builds the chunks + embeddings; Exercise 2 stores and searches them.
Run with:  python chunk_and_embed.py
Needs:     pip install scikit-learn numpy
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------- a small document (built right here) ----------
document = """
Overfitting is when a model memorises its training data and fails on new data.
The fix is a validation set and techniques like early stopping.
A random forest averages many decision trees to reduce those errors.
Because each tree sees different rows and features, their mistakes cancel out.
Gradient descent trains a model by taking small steps downhill on the error.
The learning rate sets how big each step is; too big and the loss diverges.
Attention lets each word look at the other words that are most relevant to it.
A transformer stacks many attention layers and is the basis of modern LLMs.
""".strip()


# ---------- chunk into overlapping windows of words ----------
def chunk(text, size=12, overlap=4):
    words = text.split()
    step = size - overlap                      # slide forward, keeping some overlap
    out = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if piece:
            out.append(" ".join(piece))
    return out


chunks = chunk(document, size=12, overlap=4)

# ---------- embed each chunk into a vector (TF-IDF, offline) ----------
vectorizer = TfidfVectorizer(stop_words="english")
chunk_embeddings = normalize(vectorizer.fit_transform(chunks)).toarray().astype("float32")


def embed(text):
    """Embed any query into the SAME vector space as the chunks."""
    return normalize(vectorizer.transform([text])).toarray().astype("float32")[0]


if __name__ == "__main__":
    print(f"document -> {len(chunks)} chunks (size 12 words, overlap 4)\n")
    for i, c in enumerate(chunks):
        print(f"  [{i:2d}] {c[:58]}...")
    print(f"\neach chunk is a vector of length {chunk_embeddings.shape[1]} (the vocabulary size)")
    print("overlap means a sentence split across two chunks still stays findable.")
