"""
Embedding utility module for vectorizing user queries and documentation chunks.

Utilizes SentenceTransformers to map text into a dense vector space for 
semantic retrieval. Includes support for HyDE (Hypothetical Document 
Embeddings) to improve retrieval performance on short queries by expanding 
them into a more descriptive semantic context.
"""
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
        _embedder = SentenceTransformer("all-mpnet-base-v2")
        print("Embedder ready.")
    return _embedder


def warmup():
    """Pre-load the embedding model at server startup to eliminate cold-start."""
    get_embedder()
    # run a dummy encode to fully JIT-compile the forward pass
    get_embedder().encode(["warmup query"], convert_to_numpy=True)
    print("Embedder warmup complete.")


def _normalize_query(query: str) -> str:
    """Normalize the raw user query."""
    query = query.strip().lower()
    query = re.sub(r'[?.!]+$', '', query).strip()
    return query


def embed_query(query: str, use_hyde: bool = False, groq_client=None) -> np.ndarray:
    """
    Returns a normalized embedding for the query.

    If use_hyde=True AND the query is short (< 8 words), generates a
    hypothetical answer using Groq's 8B model and embeds that instead.
    Falls back to raw query embedding on any error.
    """
    normalized = _normalize_query(query)
    text_to_embed = normalized
    if use_hyde and groq_client is not None and len(normalized.split()) < 8:
        try:
            hyde_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Clearpath documentation assistant. "
                            "Generate a short, plausible excerpt from our product documentation "
                            "that would directly answer the following user question. "
                            "Write as if it were a real support doc passage (2-4 sentences)."
                        )
                    },
                    {"role": "user", "content": normalized}
                ],
                temperature=0.3,
                max_tokens=120
            )
            hypothetical = hyde_response.choices[0].message.content.strip()
            if hypothetical:
                text_to_embed = hypothetical
                print(f"[HyDE] Query: '{normalized}' → hypothetical used for embedding.")
        except Exception as e:
            print(f"[HyDE] Skipped (error): {e}")

    model = get_embedder()
    embedding = model.encode([text_to_embed], convert_to_numpy=True)
    faiss.normalize_L2(embedding)
    return embedding
