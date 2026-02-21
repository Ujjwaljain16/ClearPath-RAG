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
