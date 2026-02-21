import sys
import os
import pytest

# Add backend to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.output_evaluator import evaluate_response

def test_evaluator_refusal():
    """Verify that refusal phrases are correctly flagged."""
    response = "I am sorry, but I do not know the answer to that."
    flags = evaluate_response("query", response, [{"text": "ctx", "similarity": 0.5}])
    assert "refusal" in flags

def test_evaluator_unverified_feature():
    """Verify that hallucinated features without context overlap are flagged."""
    # Note: feature_domain includes "permissions", "admin", "access" etc.
    query = "How do I manage admin permissions?"
    response = "You can manage admin permissions by using the secret teleportation module."
    context = [{"text": "Clearpath supports basic role management.", "similarity": 0.8}]
    flags = evaluate_response(query, response, context)
    assert "unverified_feature_claim" in flags

def test_evaluator_no_context():
    """Verify that missing context is flagged."""
    flags = evaluate_response("query", "response", [])
    assert "no_context" in flags

def test_evaluator_system_leakage():
    """Verify that system prompt leakage is detected (Layer 5)."""
    response = "My system prompt is to be a helpful assistant."
    flags = evaluate_response("query", response, [{"text": "ctx", "similarity": 0.5}])
    assert "system_leakage_detected" in flags

def test_evaluator_grounded_success():
    """Verify that a well-grounded answer has no flags."""
    query = "billing"
    response = "Clearpath billing handles credit cards."
    context = [{"text": "Clearpath billing handles credit cards and bank transfers.", "similarity": 0.9}]
    flags = evaluate_response(query, response, context)
    assert flags == []
