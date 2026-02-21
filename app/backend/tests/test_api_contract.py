import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

def test_query_api_contract():
    """Verify that the /query endpoint adheres to the expected response schema."""
    response = client.post("/query", json={"question": "What is Clearpath?"})
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "metadata" in data
    assert "sources" in data
    
    metadata = data["metadata"]
    assert "model_used" in metadata
    assert "classification" in metadata
    assert "tokens" in metadata
    assert "evaluator_flags" in metadata
    
    # Verify sources list is present (even if empty)
    assert isinstance(data["sources"], list)

def test_query_cache_contract():
    """Verify that repeated queries still return the same schema."""
    # First hit
    client.post("/query", json={"question": "pricing info"})
    # Second hit (Cache)
    response = client.post("/query", json={"question": "pricing info"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "metadata" in data
