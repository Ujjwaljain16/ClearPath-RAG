# Clearpath Nexus — RAG Implementation Reflection

### Q1: Routing Logic
**Describe your deterministic routing logic. Why did you choose this boundary? Give one example of a query that might be misclassified. How would you improve it (without using an LLM)?**

For the router, I implemented a **deterministic point-based heuristic system**. I wanted to avoid the "Russian Doll" problem where you call an LLM just to decide which LLM to call. Points are awarded based on:
1. **Query Length**: >20 words hints at complex requirements (+2pts).
2. **Intent Keywords**: Words like "integrate," "troubleshoot," or "SLA" signal high-gravity technical needs (+1pt).
3. **Reasoning Markers**: Phrases like "difference between" or "how do I" indicate multi-turn logic (+1pt).

A score of 0-1 stays with **Llama 3.1 8B** (fast/cheap fact retrieval), while a score of >= 2 triggers **Llama 3.3 70B** for synthesis. I chose this boundary because 70B is essentially "overkill" for simple pricing lookups but necessary for explaining *why* an architectural decision was made.

**Misclassification Example (False Negative):**
If a user asks simply, *"Clearpath vs Jira"*.
This query is short (3 words) and lacks specific reasoning markers. It scores a 0 and hits the 8B model. However, competitive analysis is inherently high-reasoning—it requires identifying overlapping feature sets and articulating nuanced tradeoffs. The 8B model might just list features, missing the "vibe" of a SWOT analysis that the 70B model would nail.

**Improvement (Non-LLM):**
I’d use **Semantically-Aware Embedding Checks**. Instead of just counting words, I’d project the query into vector space and calculate the cosine similarity against a pre-defined "Complexity Cluster" (past queries known to need reasoning). This gives you semantic intelligence at the cost of a single vector math operation, which is far faster than an LLM call.

---

### Q2 : Retrieval Failures
**Describe a case where your RAG pipeline retrieved the wrong chunk — or nothing at all. What was the query? What did your system retrieve? Why did the retrieval fail? What would fix it?**

One of the most interesting "real-world" failures I encountered during testing was the **"Pro Plan" Pricing Conflict**.

**User Query:** *"How much does the Pro plan cost monthly?"*
**Actual Error:** The system retrieved a sentence from an internal sales retro stating *"we could drop to $45/mo for some teams"* instead of the official Pricing Sheet value of **$49/mo**.

**The Root Cause (Semantic Bias):**
The failure happened because the internal doc had a high density of "pricing" keywords around that $45 figure, making it semantically "sticky" for the dense retriever. The official Pricing Sheet, being a table, was extracted as a sparser set of tokens. The retriever was prioritizing **relevance vibes** over **source authority**.

**The Fix (Implemented):**
I solved this by moving beyond simple vector search:
1. **BM25 Hybrid Search**: Sparse lookup ensures that the literal tokens "Price" and "Pro" are anchored to the official sheet.
2. **Source Reliability Tiering**: I designed a weighted-boosting system where documents with certain prefixes (e.g., `Technical/`, `Pricing/`) get a +20% boost in the final RRF score. This forces the system to trust "Official" docs over "Internal" ones even if the semantic similarity is slightly lower.
3. **Markdown Table Ingestion**: I used `pdfplumber` to preserve table structures as pipe-delimited Markdown, ensuring the LLM sees the data's original grid relationship.

---

### Q3: Cost and Scale
**Assume 5,000 queries/day. Estimate token usage and break down cost per model. What is the biggest cost driver? What is the highest ROI cost optimization? Name one optimization you should avoid.**

**Assumptions:** ~1,500 input tokens (prompt + 3 chunks) and ~150 output tokens. 70/30 split between 8B and 70B models.
- **8B Costs**: 3,500 queries/day $\rightarrow$ ~$0.26/day.
- **70B Costs**: 1,500 queries/day $\rightarrow$ ~$1.33/day.

**Biggest Cost Driver**: Input tokens for the 70B model. In a RAG setup, you're essentially "pre-pending" a textbook to every query. Even if the user asks "Hi", the retriever injects 1,200 tokens of context. At 5,000 queries, that's millions of tokens per day.

**Highest ROI Optimization (Implemented)**: **LRU Query Caching**. In a support environment, people ask the same thing constantly ("How do I log in?"). By caching the *final answer string* based on exact-match query hash, we bypass retrieval and model calls entirely for repeat hits. It drops cost to $0 and latency to ~2ms.

**Optimization to Avoid**: Aggressively cutting `top_k`. If you only retrieve the top-1 chunk to save tokens, the LLM loses its ability to "cross-reference" facts. You save $0.05 but spend hours of engineering time fixing the hallucinations that follow.

---

### Q4: What Is Broken
**Identify one genuine technical limitation in your current system. Why did you ship it this way? What single change would fix it?**

**The "Silent Killer": Synchronous Reranking.**
Right now, the system uses a **Cross-Encoder reranker** (`ms-marco-MiniLM-L-6-v2`) that runs on the CPU directly within the FastAPI request thread. 

**Why I shipped it:** I wanted the project to be "plug-and-play" locally. Bundling the reranker as a library means anyone can run it without needing a secondary GPU server or K8s cluster. In a single-user demo, a 200ms CPU-block is invisible.

**The Scalability Impact:** At 5,000 queries/day, this is a disaster. Because it's a CPU-bound sync call, it **blocks the FastAPI event loop**. While the CPU is doing matrix math for Query A, it literally cannot accept Query B or even return a 200 OK. Under load, this would cause catastrophic latency spikes.

**The Fix:** Decouple the reranker into a **standalone GPU microservice**. I would call it via a non-blocking `httpx.post()` call. This offloads the heavy lifting to GPU-optimized hardware and lets the main API handle thousands of IO-bound tasks in parallel.

---

### AI Usage

I used **Antigravity** (free-tier) to pair-program the infrastructure. Below are the 5 key prompts I used to build the core modules:

1. **RAG Pipeline & Table-Aware Ingestion**: *"I need to build a RAG pipeline that can handle 30 PDFs of varied internal documentation. Standard vector search often loses context in tables so you write a Python script using pdfplumber for table-aware extraction i want to use FAISS for dense embeddings and rank-bm25 for sparse keyword matching then combine their results using RRF make sure it deduplicates chunks by id so llm doesnt get repeated context."*
2. **Safeguard Output Evaluator**: *"To prevent hallucinations i need an Output Evaluator that runs before the response is finalized it should check three specific signals (1) did the retriever actually find any context after thresholding (2) does the LLM answer look like a hidden refusal and (3) does the answer overlap with key terms present in the retrieved chunks if any of these checks fail return a list of evaluator_flags so i can show a low-confidence warning along with this implement a guradrailing and a robust system prompt that make sure that there is no content leakage am open to discuss the security architecture and how to make it more secure"*
3. **Frontend & Streaming Metadata Handling**: *"I need to build a minimal react frontend which should have a observability panel i want to support token by token streaming for the chat bubble but i also need to update the observability metrics once they are finalized so implemented a hidden metadata envelope pattern and the backend should yield a json payload at the end of the stream and the frontend should parse it to populate the dashboard without showing the raw json to the user make sure it looks good not so empty populate it okie"*
4. **Deterministic Point-Router & Chunking Strategy**: *"After analysisng the format of the documents i realised that the documents are not that big so i decided to use a deterministic point router to route the queries to either a 8B or 70B model without using an expensive LLM for routing make sure it has less than 10ms latency and handles edge cases like short queries and technical jargon also change the chunking strategy to sliding window one of 900 chars and all also make sure  we calculate the mean similarity score of all 15 candidates"*
5. **Test Coverage**: *"Here is the prd analyse the codebase throughlly deeply carefully and make sure we have covered all the needed requirements features and if not then add them in the codebase and make sure it is well documented and follows the best practices of software engineering and also make sure it is production ready and can be deployed in a production environment and make the needed test suite for this application"*

The other prompts includes basic debugging polishing the comments answers and readme file to look professional classy and easy to understand
---

### Bonus Notes

For the **Bonus Challenges**, I made some specific architectural choices:

1. **Memory**: I chose a **Frontend-managed Sliding Window** (last 6 messages). This keeps the backend "stateless" (no DB needed for a demo!), which makes the whole app portable as a single folder while still giving the user a coherent conversation.
2. **Streaming & Observability**: Implementing streaming usually breaks the "Observability" panel because metadata (like latency/token counts) isn't finalized until the end. I solved this by adding a **hidden metadata envelope** at the end of the stream. The frontend parses this hidden JSON to populate the debug panel without the user seeing the "guts" of the system.
3. **Security**: I implemented a 5-layer defense including "Instruction Wrapping" to ensure that if a PDF says "Ignore all previous instructions," the LLM treats it as an untrusted string rather than a command.
