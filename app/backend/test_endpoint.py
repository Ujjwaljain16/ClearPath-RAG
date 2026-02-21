import requests
import json
import time
import os

# Production-ready: Use environment variable if provided, default to localhost
URL = os.getenv("TEST_API_URL", "http://localhost:8000/query")

queries = [
    # Should be Simple (Llama 8B) 
    "How to clear cache?",
    "What is the base url?",
    
    # Should be Complex (Llama 70B)
    "Walk me through the steps to connect to a secure websocket with the Clearpath API and handle 403 errors.",
    "Compare the security features of the Basic plan versus the Enterprise plan and explain which is safer for a financial institution."
]

def run_tests():
    print(f"🚀 Starting endpoint verification at: {URL}\n")
    for q in queries:
        print(f"Query: '{q}'")
        try:
            start = time.time()
            response = requests.post(URL, json={"question": q}, timeout=15)
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ [200 OK] in {latency}ms")
                print(f"  Model: {data['metadata']['model_used']} ({data['metadata']['classification']})")
                print(f"  Answer Preview: {data['answer'][:80]}...")
                print(f"  Sources: {len(data['sources'])}")
            else:
                print(f"  ❌ [Error {response.status_code}]: {response.text}")
        except Exception as e:
            print(f"  ❌ [Fail]: {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
