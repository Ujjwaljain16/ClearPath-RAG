"""
Thread-safe in-memory LRU cache for query results.

Caches processed answers and their associated metadata to avoid redundant 
expensive LLM calls and retrieval operations for repeated queries.
"""
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

_CACHE_TTL_SECONDS = 300  # 5mins
_CACHE_MAX_SIZE = 256

class _LRUCache:
    def __init__(self, max_size: int, ttl: float):
        self._store: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = threading.Lock()

    def _normalize_key(self, query: str) -> str:
        """Normalize query to improve cache hit rate."""
        return query.strip().lower().rstrip("?!.")

    def get(self, query: str) -> Optional[Dict]:
        key = self._normalize_key(query)
        with self._lock:
            if key not in self._store:
                return None
            value, ts = self._store[key]
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, query: str, value: Dict):
        key = self._normalize_key(query)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def stats(self) -> Dict:
        with self._lock:
            return {"size": len(self._store), "max_size": self._max_size}


# global singleton
query_cache = _LRUCache(max_size=_CACHE_MAX_SIZE, ttl=_CACHE_TTL_SECONDS)
