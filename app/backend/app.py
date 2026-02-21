import os
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

from routing.model_router import route_query
from rag.retriever import retrieve_context, load_index
from rag.embedder import warmup as warmup_embedder
from rag.generator import generate_answer, get_groq_client, expand_query
from rag.cache import query_cache
from evaluation.output_evaluator import evaluate_response
from query_logging.query_logger import log_query_async


# Startup lifespan warms up all singletons at boot time
# Pre-loading the embedder model, FAISS index, and Groq client ensures 
# that the first query doesn't suffer from cold-start latency
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server startup: warming up all singletons...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, warmup_embedder)
    await loop.run_in_executor(None, load_index)
    await loop.run_in_executor(None, get_groq_client)
    from rag.retriever import _get_reranker
    await loop.run_in_executor(None, _get_reranker)
    print("All singletons warmed up. Ready for requests.")
    yield
    print("Server shutting down.")


app = FastAPI(title="Clearpath Nexus RAG API", lifespan=lifespan)

# CORS Configuration reads from environment for production readiness
CORS_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Contract Models
class ChatMessage(BaseModel):
    role: str
    text: str

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

class TokenUsage(BaseModel):
    input: int
    output: int

class MetadataResponse(BaseModel):
    model_used: str
    classification: str
    tokens: TokenUsage
    latency_ms: int
    retrieval_latency_ms: int
    num_chunks_retrieved: int
    avg_similarity_score: float
    routing_score: int
    evaluator_flags: List[str]
    cache_hit: bool = False

class SourceResponse(BaseModel):
    document: str
    section: Optional[str] = None
    page: Optional[int] = None
    relevance_score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    metadata: MetadataResponse
    sources: List[SourceResponse]
    conversation_id: str


# Endpoints
@app.get("/")
def read_root():
    return {"message": "Clearpath RAG API is running", "cache_stats": query_cache.stats()}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    start_time = time.time()

    # Check cache
    cached = query_cache.get(request.question)
    if cached:
        cached["metadata"]["latency_ms"] = int((time.time() - start_time) * 1000)
        cached["metadata"]["cache_hit"] = True
        return QueryResponse(**cached)

    # Routing phase - determine complexity
    try:
        route_decision = route_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {e}")

    # Retrieval phase using hybrid search
    retrieval_start = time.time()
    try:
        with ThreadPoolExecutor() as executor:
            expansion_future = None
            if len(request.question.split()) < 8:
                expansion_future = executor.submit(expand_query, request.question)
            
            expanded_query = expansion_future.result() if expansion_future else None
            
        retrieval_result = retrieve_context(request.question, expanded_query=expanded_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    retrieval_latency = int((time.time() - retrieval_start) * 1000)

    # Generating answer via LLM
    model_string = route_decision["model_used"]
    llm_result = generate_answer(
        query=request.question,
        retrieved_chunks=retrieval_result.get("chunks", []),
        model_string=model_string,
        history=request.history
    )

    answer       = llm_result["answer"]
    input_tokens  = llm_result["usage"]["prompt_tokens"]
    output_tokens = llm_result["usage"]["completion_tokens"]

    # Content evaluation and leakage detection
    flags = evaluate_response(
        query=request.question,
        response=answer,
        retrieved_chunks=retrieval_result.get("chunks", [])
    )

    # Final sanitization pass strips sensitive internal tokens
    patterns = [
        r"system\s*prompt", r"hidden\s*polic", r"untrusted\s*data", 
        r"ignore\s*previous", r"@@@\s*chunk", r"\[START\s*OF\s*UNTRUSTED",
        r"documentation\s*chunk"
    ]
    for pattern in patterns:
        answer = re.sub(pattern, "[REDACTED]", answer, flags=re.IGNORECASE)

    total_latency = int((time.time() - start_time) * 1000)

    # Query logging
    log_data = {
        "query":               request.question,
        "classification":      route_decision["classification"],
        "model_used":          route_decision["model_used"],
        "routing_score":       route_decision["score"],
        "tokens_input":        input_tokens,
        "tokens_output":       output_tokens,
        "latency_ms":          total_latency,
        "retrieval_latency_ms": retrieval_latency,
        "num_chunks_retrieved": len(retrieval_result.get("chunks", [])),
        "avg_similarity_score": retrieval_result.get("avg_similarity", 0.0),
        "evaluator_flags":     flags
    }
    asyncio.create_task(log_query_async(log_data))

    # Build sources with section info
    sources = [
        SourceResponse(
            document=c.get("doc_id", "unknown"),
            section=c.get("section_title"),
            page=c.get("page"),
            relevance_score=round(c.get("similarity", 0.0), 4)
        )
        for c in retrieval_result.get("chunks", [])
    ]

    # Response caching
    metadata = MetadataResponse(
        model_used=route_decision["model_used"],
        classification=route_decision["classification"],
        tokens=TokenUsage(input=input_tokens, output=output_tokens),
        latency_ms=total_latency,
        retrieval_latency_ms=retrieval_latency,
        num_chunks_retrieved=len(retrieval_result.get("chunks", [])),
        avg_similarity_score=retrieval_result.get("avg_similarity", 0.0),
        routing_score=route_decision["score"],
        evaluator_flags=flags,
        cache_hit=False
    )
    response_obj = QueryResponse(
        answer=answer,
        metadata=metadata,
        sources=sources,
        conversation_id=request.conversation_id or f"conv_{int(time.time())}"
    )
    query_cache.set(request.question, response_obj.model_dump())

    return response_obj

@app.post("/query_stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Bonus: Streaming endpoint for token-by-token response.
    Note: Structured logging and evaluation metrics are captured at the end 
    of the stream to handle partial data trade-offs.
    """
    route_decision = route_query(request.question)
    
    with ThreadPoolExecutor() as executor:
        expansion_future = None
        if len(request.question.split()) < 8:
            expansion_future = executor.submit(expand_query, request.question)
        
        expanded_query = expansion_future.result() if expansion_future else None
        
    retrieval_result = retrieve_context(request.question, expanded_query=expanded_query)
    
    return StreamingResponse(
        generate_answer_stream(
            query=request.question,
            retrieved_chunks=retrieval_result.get("chunks", []),
            model_string=route_decision["model_used"],
            history=request.history
        ),
        media_type="text/plain"
    )
