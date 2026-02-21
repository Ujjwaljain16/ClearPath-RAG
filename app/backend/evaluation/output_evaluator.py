import re
import string
from typing import List, Dict, Any

# Comprehensive stopword list for filtering noise tokens during keyword extraction.
# Includes common English functional words and domain-specific navigation terms.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "shall", "to", "of", "in",
    "on", "at", "for", "from", "with", "by", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "this",
    "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "same", "than", "too", "very", "just", "own", "so", "if", "then",
    "also", "up", "out", "any", "here", "there", "now", "get", "use", "used",
    "using", "can", "like", "well", "new", "user", "click", "go", "see", "make",
    "note", "please", "refer"
}


def extract_keywords(text: str, top_n: int = 15) -> set:
    """
    Extract the top_n most frequent meaningful terms from text,
    filtering with a comprehensive stopword list and min-length of 4.
    """
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = text.split()
    keywords = [w for w in words if w not in _STOPWORDS and len(w) >= 4]

    freq: Dict[str, int] = {}
    for kw in keywords:
        freq[kw] = freq.get(kw, 0) + 1

    sorted_kws = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return {k for k, _ in sorted_kws[:top_n]}


def evaluate_response(
    query: str,
    response: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> List[str]:
    """
    Evaluates the LLM's response for quality and grounding:
    - Flags cases where no supporting documentation was found.
    - Detects explicit refusals or "I don't know" states.
    - Performs keyword overlap analysis to verify feature claims against context.
    - Checks for sensitive information leakage (system prompt, hidden rules).
    """
    flags: List[str] = []
    response_lower = response.lower()
    if not retrieved_chunks:
        flags.append("no_context")

    # Detect explicit refusals in the response
    refusal_phrases = [
        "i don't know", "i do not know", "i could not find",
        "i am sorry", "i'm sorry", "not mentioned in the provided",
        "cannot answer", "does not contain information",
        "not found in the clearpath", "no information available"
    ]
    if any(phrase in response_lower for phrase in refusal_phrases):
        flags.append("refusal")

    #Domain feature check only run if context exists and no refusal
    feature_domain = {
        "workspace", "billing", "permissions", "integration", "api",
        "plan", "enterprise", "sso", "oauth", "webhook", "pricing",
        "subscription", "admin", "role", "access"
    }
    query_lower = query.lower()
    has_feature = any(kw in query_lower for kw in feature_domain)

    if has_feature and "refusal" not in flags and retrieved_chunks:
        context_keywords: set = set()
        valid_ctx = False
        for chunk in retrieved_chunks:
            if chunk.get("similarity", 0.0) > 0.3:
                valid_ctx = True
                context_keywords |= extract_keywords(chunk.get("text", ""))

        if not valid_ctx:
            flags.append("unverified_feature_claim")
        else:
            response_kws = extract_keywords(response)
            overlap = context_keywords & response_kws
            if len(overlap) < 2:
                flags.append("unverified_feature_claim")

    # Detect potential prompt leakage or jailbreak attempts
    # Checks if the model is echoing internal instructions or policies.
    import re
    leakage_patterns = [
        r"system\s*prompt", r"hidden\s*polic", r"ignore\s*previous",
        r"developer\s*mode", r"internal\s*reasoning", r"untrusted\s*data"
    ]
    for pattern in leakage_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            flags.append("system_leakage_detected")
            break

    return flags
