import re
from typing import Dict, Any

def route_query(query: str) -> Dict[str, Any]:
    """
    Categorizes incoming queries by technical complexity to determine 
    the optimal model for generation. 
    
    Heuristics include query length, reasoning markers, and the presence 
    of troubleshooting or procedural keywords. Matches scoring 2 or higher 
    are routed to the larger 70B model, while simpler lookups use the 8B model.
    """
    
    score = 0
    query_lower = query.lower()
    
    # Length based check
    words = query_lower.split()
    if len(words) > 15:
        score += 1
        
    # Check for multiple questions or nested queries
    if query.count("?") > 1:
        score += 1
        
    # Identify reasoning intent (e.g., comparisons, explanations)
    reasoning_keywords = ["why", "compare", "evaluate", "difference", "explain", "reason"]
    if any(re.search(r'\b' + kw + r'\b', query_lower) for kw in reasoning_keywords):
        score += 2
        
    # Identify troubleshooting intent
    troubleshooting_keywords = ["fail", "error", "broken", "bug", "issue", "doesn't work", "not working", "crash"]
    if any(re.search(r'\b' + kw + r'\b', query_lower) for kw in troubleshooting_keywords):
        score += 2
        
    # Identify procedural or "how-to" queries
    procedural_keywords = ["how to", "steps", "process", "walk me through", "guide", "tutorial"]
    if any(kw in query_lower for kw in procedural_keywords):
        score += 2
        
    # Factor in sentiment or urgency markers
    emotional_keywords = ["frustrated", "complaint", "urgent", "asap", "angry", "terrible", "worst"]
    if any(re.search(r'\b' + kw + r'\b', query_lower) for kw in emotional_keywords):
        score += 1
        
    # Classification
    if score >= 2:
        classification = "complex"
        model_used = "llama-3.3-70b-versatile"
    else:
        classification = "simple"
        model_used = "llama-3.1-8b-instant"
        
    return {
        "classification": classification,
        "model_used": model_used,
        "score": score
    }
