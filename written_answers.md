# Clearpath Nexus — Written Answers

### Q1: Routing Logic
**Describe your deterministic routing logic. Why did you choose this boundary? Give one example of a query that might be misclassified. How would you improve it (without using an LLM)?**

The router utilizes a deterministic point-based heuristic system. Points are awarded for query length (>20 words: +2pts), presence of high-logic intent keywords like "integrate", "troubleshoot", "architecture", or "SLA" (+1pt), and grammatical structures indicating multi-step reasoning such as "difference between", "how do I", or "why is" (+1pt). A consolidated score of 0-1 routes to Llama 3.1 8B, while a score of >= 2 triggers the Llama 3.3 70B model.

This boundary was drawn to reserve the expensive 70B model for synthesis and complex troubleshooting, while delegating high-volume, low-complexity fact lookups to the efficient 8B model. This mimics a "Level 1 vs Level 2" support triage system common in enterprise IT.

**Misclassification Example:** A user asks, *"What is SSO?"* This is short and lacks complex tokens, scoring a 0. The 8B model gives a generic dictionary definition. However, if the user's implicit intent was to find the *Clearpath-specific SSO configuration steps*, the 8B model might miss the nuance required to link varied segments of the "Data Security" and "API Documentation" files. The 70B model would have proactively synthesized the security requirements.

**Improvement (Non-LLM):** I would implement an N-gram based cosine similarity check against a "Complexity Corpus"—a small, local vector space of known complex support tickets. If a query is mathematically similar to known complex issues (even without matching specific keywords), it would be escalated. This significantly improves precision over basic regex without the latency of an LLM call.

---

### Q2: Retrieval Failure
**Identify a specific failure case in your RAG retrieval. What was retrieved? Why did it fail? How would you fix it?**

**Failure Case:** Querying *"What are the exact error codes for the OAuth API?"*
**What was retrieved:** Sections from the User Guide mentioning "login errors" and sections from the System Architecture about "OAuth flows", but missing the specific API error table.
**Why it failed:** Dense embeddings (SentenceTransformers) heavily compress semantic meaning. "OAuth" and "error" map closely to general authentication documentation. The specific numerical error codes (e.g., "401", "403") lose their strict token identity in dense vector space, causing precise tabular data to be outranked by semantically broad paragraphs. Furthermore, standard pdf extractors routinely drop table cells entirely.
**The Fix (Implemented):** I implemented two critical fixes to solve this: 1) Explicit table cell extraction during ingestion (converting rows to pipe-delimited text strings), and 2) A hybrid retrieval pipeline combining FAISS with a BM25 sparse index. BM25 performs exact keyword matching, guaranteeing that literal tokens like "OAuth 403" are surfaced, which are then merged with dense results via Reciprocal Rank Fusion (RRF).

---

### Q3: Cost and Scale
**Assume 5,000 queries/day. Estimate token usage and break down cost per model. What is the biggest cost driver? What is the highest ROI cost optimization? Name one optimization you should avoid.**

**Assumption:** Average 1,500 input tokens (prompt + 3 retrieved chunks) and 150 output tokens.
**Distribution:** 70% simple queries (3,500 to 8B), 30% complex (1,500 to 70B).
- **8B Model:** 5.25M IN, 525k OUT per day. (At Groq rates: ~$0.26/day)
- **70B Model:** 2.25M IN, 225k OUT per day. (At Groq rates: ~$1.33/day)
**Biggest Cost Driver:** Input tokens directed to the 70B model. Because RAG injects ~1200 tokens of context per query, the input volume dominates the output volume by a 10:1 ratio.
**Highest ROI Optimization (Implemented):** An exact-match LRU Query Cache. Because customer support queries heavily follow a power-law distribution (e.g., "How do I reset my password?" asked hundreds of times), caching the final LLM string response bypasses both retrieval and generation layers entirely for repeat queries, slashing costs and dropping latency to ~2ms.
**Optimization to Avoid:** Aggressively shrinking the context window (e.g., dropping `top_k` from 3 to 1 to save input tokens). This will critically degrade response accuracy and trigger hallucinations, frustrating users and shifting the cost from LLM compute to human support loads.

---

### Q4: What Is Broken
**Identify one genuine technical limitation in your current system. Why did you ship it this way? What single change would fix it?**

**Limitation:** The system currently utilizes a synchronous Cross-Encoder reranking model (`ms-marco-MiniLM-L-6-v2`) that runs on the application server's CPU during the request hot path.

**Rationale for Shipment:** For a take-home assignment demonstrating end-to-end RAG architecture and logic, running the reranker locally inside the FastAPI monolith simplifies the environment setup for reviewers. It avoids the massive complexity of standing up a distributed GPU worker pool or requiring the reviewer to have local CUDA drivers. For single-user testing or small internal demos, the ~150-200ms CPU penalty is an acceptable trade-off for a "one-click" setup experience.

**Impact at Scale:** If deployed to a real production environment with 5,000 queries per day, this component would become a catastrophic bottleneck. Because the reranker is a heavy neural network running on the CPU, it blocks the FastAPI event loop thread. Under high concurrency (e.g., 50 simultaneous users), the server would experience "death by CPU saturation," resulting in multi-second response times and potential timeouts for all users.

**The Single Change Fix:** I would decouple the Cross-Encoder into a standalone microservice running on GPU-accelerated infrastructure (using a framework like NVIDIA Triton or Bentoml). The FastAPI backend would then communicate with this service via gRPC or a high-speed message queue. This change would drop the reranking latency to under 15ms and ensure the main API server remains responsive to IO-bound tasks while the GPU handles the heavy arithmetic.

---

### AI Usage
As per the assignment requirements, the following is a log of the primary prompts used with the AI coding assistant (Antigravity) to build this project:

1. "Research the repository and implement a robust RAG pipeline using FAISS and BM25 that can handle tabular data in the PDFs."
2. "Build a model router that uses rule-based heuristics to switch between Llama 3.1 8B and 3.3 70B without calling an LLM for the decision."
3. "Implement a safeguard evaluator that flags no-context answers and unverified feature claims."
4. "Optimize the backend for latency using singleton clients, async logging, and LRU caching."
5. "Rebrand the entire application to 'Clearpath Nexus' and ensure all documentation reflects this new identity."
6. "Implement conversation memory as a bonus challenge by injecting history into the LLM prompt."

---

### Bonus Challenge Notes

**Conversation Memory: Design & Tradeoffs**
The memory system is implemented using a **sliding window buffer** (last 6 messages) sent from the frontend to the backend.
- **Design Decision**: I chose a stateless backend approach where the client provides the history. This avoids the need for a database (Redis/Postgres) on the backend, keeping the system "one-click" portable while still maintaining context. 
- **Token Cost Tradeoff**: Each turn adds approximately 150-300 tokens to the prompt. By capping it at 6 messages (3 full turns), we strike a balance: the LLM remembers the immediate context (like "tell me more about the first one") without hitting the 8B model's context limits or significantly increasing Groq's input token costs for long-running sessions.

**Streaming & Structured Output Parsing**
The "Streaming" challenge requirement asks where structured output parsing breaks when streaming. In a standard REST response, the LLM provides a complete JSON object which can be easily validated and parsed. However, when **streaming token-by-token**, the intermediate states (e.g., `{"ans`) are incomplete and strictly invalid JSON. This means tools that rely on regex or JSON parsers to extract "citations" or "metadata" from the LLM's raw text will fail until the stream is finalized. This system handles this by streaming the raw `answer` content only, while metrics are calculated separately.
