"""
Asynchronous utility for logging system queries and performance metrics.

By submitting log writes to a background thread pool, we ensure that disk I/O 
operations do not block the main request handler, maintaining tight response 
SLAs for the end user.
"""
import json
import asyncio
import concurrent.futures
from typing import Dict, Any

LOG_FILE = "query_logs.jsonl"
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="logger")

def _write_log_sync(log_data: Dict[str, Any]):
    """Synchronous write — runs in background thread."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        print(f"Failed to write log: {e}")

async def log_query_async(log_data: Dict[str, Any]):
    """
    Async wrapper that submits the log write to a background thread.
    Call with: asyncio.create_task(log_query_async(data))
    Returns immediately without blocking the response.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _write_log_sync, log_data)

# Sync fallback for compatibility
def log_query(log_data: Dict[str, Any]):
    _write_log_sync(log_data)
