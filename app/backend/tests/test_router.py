import sys
import os
import pytest

# Add backend to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routing.model_router import route_query

def test_router_factual_simple():
    """Verify that a simple factual query routes to the 8B model."""
    result = route_query("What is Clearpath?")
    assert result["model_used"] == "llama-3.1-8b-instant"
    assert result["classification"] == "simple"

def test_router_complex_multi_step():
    """Verify that a complex query routes to the 70B model."""
    query = "Walk me through the steps to set up a cross-team workspace and invite 50 members."
    result = route_query(query)
    assert result["model_used"] == "llama-3.3-70b-versatile"
    assert result["classification"] == "complex"

def test_router_pricing_billing():
    """Verify that pricing/billing queries route to the 70B model for extra precision."""
    result = route_query("Explain the enterprise pricing tiers for Clearpath in detail.")
    assert result["model_used"] == "llama-3.3-70b-versatile"
    assert result["classification"] == "complex"

def test_router_short_keyword():
    """Verify that a troubleshooting keyword query routes to 70B even if short."""
    result = route_query("Login issue")
    assert result["model_used"] == "llama-3.3-70b-versatile"
    assert result["classification"] == "complex"

def test_router_long_concatenation():
    """Verify that very long queries are treated as complex."""
    query = "Where can I find the API documentation? Also, I need help with SSO integration for my team of 500 people, specifically looking at the SAML 2.0 configuration steps."
    result = route_query(query)
    assert result["model_used"] == "llama-3.3-70b-versatile"
