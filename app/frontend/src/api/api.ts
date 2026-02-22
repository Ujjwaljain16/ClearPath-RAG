import { QueryRequest, QueryResponse } from "./models";
import { API_BASE_URL } from "../config";

export async function askApi(request: QueryRequest): Promise<QueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(request)
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    return await response.json();
}

export async function* askStreamApi(request: QueryRequest): AsyncIterableIterator<string> {
    const response = await fetch(`${API_BASE_URL}/query_stream`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(request)
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) return;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        yield decoder.decode(value);
    }
}
