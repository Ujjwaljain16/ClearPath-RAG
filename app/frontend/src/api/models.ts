export interface MetadataTokens {
    input: number
    output: number
}

export interface MetadataResponse {
    model_used: string
    classification: string
    tokens: MetadataTokens
    latency_ms: number
    retrieval_latency_ms: number
    num_chunks_retrieved: number
    avg_similarity_score: number
    routing_score: number
    evaluator_flags: string[]
}

export interface SourceResponse {
    document: string
    section?: string
    page?: number
    relevance_score?: number
}

export interface QueryResponse {
    answer: string
    metadata: MetadataResponse
    sources: SourceResponse[]
    conversation_id: string
}

export interface ChatMessage {
    role: 'user' | 'bot'
    text: string
}

export interface QueryRequest {
    question: string
    conversation_id?: string
    history?: ChatMessage[]
}
