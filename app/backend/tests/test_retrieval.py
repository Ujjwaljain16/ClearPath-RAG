import sys
import os
import pytest

# Add backend to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import retrieve_context

def test_retrieval_smoke():
    """Verify that a basic query returns chunks from the index."""
    # Note: Requires the index to be present in backend/index.faiss
    result = retrieve_context("Clearpath")
    assert "chunks" in result
    assert len(result["chunks"]) > 0
    assert "similarity" in result["chunks"][0]

def test_retrieval_high_relevance():
    """Verify that a query for a known document title returns a high similarity score."""
    result = retrieve_context("Data Security and Privacy Policy")
    top_chunk = result["chunks"][0]
    # We expect a high-quality match to have score > 0.3
    assert top_chunk["similarity"] > 0.3
