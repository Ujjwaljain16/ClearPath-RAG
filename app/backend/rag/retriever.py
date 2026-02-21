"""
Retriever module implementing a hybrid search strategy.
Combines dense vector search (FAISS) with sparse keyword search (BM25)
to provide high-precision context for the RAG pipeline.
"""
import os
import json
import faiss
import numpy as np
from typing import Dict, Any, List, Tuple

from rag.embedder import embed_query
from rag.bm25_index import bm25_search

# Index paths
INDEX_PATH    = os.path.join(os.path.dirname(__file__), "../index.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "../chunk_metadata.json")

# Model and data singletons initialized at startup
_index:     faiss.Index = None
_metadata:  List[Dict]  = None
_reranker                = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder reranker loaded.")
        except Exception as e:
            print(f"WARNING: Cross-encoder not available: {e}. Skipping reranking.")
    return _reranker


def load_index():
    """Load FAISS index and metadata into memory. Called once at startup."""
    global _index, _metadata
    if _index is None:
        if os.path.exists(INDEX_PATH):
            print(f"Loading FAISS index from {INDEX_PATH}")
            _index = faiss.read_index(INDEX_PATH)
        else:
            print("WARNING: FAISS index not found. Run ingestion first.")
            return False
    if _metadata is None:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            print("WARNING: Metadata file not found.")
            return False
    return True


def _rrf_fuse(dense_ranked: List[Tuple[int, float]],
              bm25_ranked: List[Tuple[int, float]],
              k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion standard formula: score(d) = Σ 1/(k + rank_i(d))
    Fuses dense and sparse ranked lists into a single combined ranking.
    k=60 is the standard default from the original RRF paper.
    """
    scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_context(query: str, expanded_query: str = None, top_k: int = 15, max_tokens: int = 1800) -> Dict[str, Any]:
    """
    Retrieval pipeline:
      1. Dense retrieval using FAISS (supports HyDE expansion)
      2. Sparse retrieval using BM25 for exact keyword matching
      3. Reciprocal Rank Fusion (RRF) for result merging
      4. Chunk deduplication and dynamic thresholding
      5. Cross-encoder reranking for precision
      6. Context window optimization
    """
    if not load_index():
        return {"chunks": [], "avg_similarity": 0.0}

    # 1) FAISS dense retrieval
    search_query = expanded_query if expanded_query else query
    q_emb = embed_query(search_query)
    distances, indices = _index.search(q_emb, top_k)
    dense_ranked = [
        (int(indices[0][i]), float(distances[0][i]))
        for i in range(len(indices[0]))
        if indices[0][i] != -1
    ]

    # 2) BM25 sparse keyword retrieval
    bm25_ranked = bm25_search(query, top_k=top_k)

    # 3) Fuse results via RRF
    fused_ranked = _rrf_fuse(dense_ranked, bm25_ranked)

    # 4) Map indices to metadata and deduplicate chunks
    seen_chunk_ids = set()
    candidates = []
    for idx, rrf_score in fused_ranked:
        if idx >= len(_metadata):
            continue
        chunk = dict(_metadata[idx])

        # Attach dense similarity score (for thresholding) or 0 if only in BM25
        dense_sim = next((s for i, s in dense_ranked if i == idx), 0.0)
        chunk["similarity"] = dense_sim
        chunk["rrf_score"]  = rrf_score

        cid = chunk.get("chunk_id", chunk.get("text", "")[:80])
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            candidates.append(chunk)

    if not candidates:
        return {"chunks": [], "avg_similarity": 0.0}

    # 5) Dynamic similarity thresholding
    sims = [c["similarity"] for c in candidates if c["similarity"] > 0]
    if sims:
        mean_sim = float(np.mean(sims))
        std_sim  = float(np.std(sims))
        std_sim  = max(std_sim, 0.01)
        threshold = max(mean_sim - std_sim, 0.15)  # hard floor at 0.15
        filtered = [
            c for c in candidates
            if c["similarity"] >= threshold or c["rrf_score"] > 0.025
        ]
    else:
        # All candidates came from BM25 keep top-K by RRF
        filtered = candidates[:top_k]

    # 6) Final reranking using a cross-encoder model
    # To reduce latency we bypass reranking if the top candidate is a strong match
    top_sim = filtered[0].get("similarity", 0) if filtered else 0
    
    if top_sim > 0.6:
        pass
    else:
        reranker = _get_reranker()
        if reranker is not None and len(filtered) > 1:
            rerank_pool = filtered[:6]
            pairs = [(query, c["text"]) for c in rerank_pool]
            try:
                ce_scores = reranker.predict(pairs)
                for i, c in enumerate(rerank_pool):
                    c["ce_score"] = float(ce_scores[i])
                filtered[:6] = sorted(rerank_pool, key=lambda c: c.get("ce_score", 0), reverse=True)
            except Exception as e:
                print(f"Reranker failed: {e}")

    # 7) Token cap
    final_chunks = []
    total_chars  = 0
    max_chars    = max_tokens * 4

    for c in filtered:
        char_len = len(c["text"])
        if total_chars + char_len <= max_chars:
            final_chunks.append(c)
            total_chars += char_len
        else:
            break

    avg_sim = (
        sum(c["similarity"] for c in final_chunks) / len(final_chunks)
        if final_chunks else 0.0
    )

    return {"chunks": final_chunks, "avg_similarity": avg_sim}
