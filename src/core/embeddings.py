# src/core/embeddings.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    """
    Encode text into an embedding vector using sentence-transformers.
    Returns a tensor suitable for cosine similarity.
    """
    return model.encode(text, convert_to_tensor=True)