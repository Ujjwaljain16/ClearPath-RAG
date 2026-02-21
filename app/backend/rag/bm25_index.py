"""
BM25 keyword index management for hybrid retrieval.

While dense retrieval (FAISS) handles semantic similarity, it can sometimes 
miss exact terminology or error codes (like "OAuth 403"). This module 
integrates BM25 to ensure precise literal token recall, which is then fused 
with dense results for a more robust retrieval experience.
"""
import os
import json
import pickle
import re
from typing import List, Dict, Any

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("WARNING: rank-bm25 not installed. BM25 hybrid search disabled. Run: pip install rank-bm25")

BM25_PATH = os.path.join(os.path.dirname(__file__), "../bm25_index.pkl")

_bm25_index: "BM25Okapi" = None
_bm25_corpus: List[str] = None  # raw texts for position mapping


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def build_bm25(chunks: List[Dict[str, Any]]) -> "BM25Okapi":
    if not BM25_AVAILABLE:
        return None
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def save_bm25(bm25_index, chunks: List[Dict[str, Any]], path: str = BM25_PATH):
    if bm25_index is None:
        return
    corpus_texts = [c["text"] for c in chunks]
    with open(path, "wb") as f:
        pickle.dump({"index": bm25_index, "corpus": corpus_texts}, f)
    print(f"BM25 index saved to {path}")


def load_bm25(path: str = BM25_PATH):
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    _bm25_index = data["index"]
    _bm25_corpus = data["corpus"]
    return _bm25_index, _bm25_corpus


def bm25_search(query: str, top_k: int = 15):
    """
    Returns top_k (index, score) pairs using BM25.
    Returns empty list if BM25 not available.
    """
    bm25, corpus = load_bm25()
    if bm25 is None or corpus is None:
        return []
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    # Get top_k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(idx, float(scores[idx])) for idx in top_indices]
