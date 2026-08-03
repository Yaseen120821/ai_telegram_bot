r"""
app/rag/cache_manager.py - Multi-Level LRU Caching Engine
==========================================================

1. PURPOSE:
-----------
Implements high-performance, thread-safe LRU (Least Recently Used) caching for dense 384-dimensional vector embeddings
and complete RAG search retrieval query results.

2. WHY IT EXISTS (SUB-MILLISECOND LATENCY & COMPUTE SAVINGS):
-------------------------------------------------------------
Generating vector embeddings via PyTorch models requires matrix multiplication operations.
- Caching embedding vectors by text hash avoids re-embedding identical text passages.
- Caching full retrieval results for repeated user queries returns results instantly (< 1ms latency).

3. RESPONSIBILITIES:
--------------------
- Provide LRU Embedding Vector Cache (`get_embedding`, `put_embedding`).
- Provide TTL-based Retrieval Query Result Cache (`get_retrieval_result`, `put_retrieval_result`).
- Support automatic cache invalidation (`clear_cache`, `invalidate_retrieval_cache`).
- Track cache hit and miss statistics.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Interfaced by `EmbeddingManager`, `RetrievalManager`, and `RAGManager`.

5. COMPLETE CODE:
-----------------
"""

import time
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Optional, List, Dict, Any

logger = logging.getLogger("sana_ai.rag.cache")


class CacheManager:
    """
    Thread-safe Singleton managing embedding vectors and retrieval query result LRU caches.
    """
    _instance: Optional["CacheManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        max_embedding_entries: int = 2000,
        max_retrieval_entries: int = 500,
        retrieval_ttl_seconds: float = 300.0
    ) -> None:
        """
        Initializes CacheManager LRU caches.

        Args:
            max_embedding_entries (int): Max text embedding vector entries to cache.
            max_retrieval_entries (int): Max search query result entries to cache.
            retrieval_ttl_seconds (float): Retrieval query cache entry Time-To-Live in seconds.
        """
        if CacheManager._instance is not None:
            raise RuntimeError("CacheManager is a Singleton! Use `CacheManager.get_instance()` instead.")

        self.max_embedding_entries: int = max_embedding_entries
        self.max_retrieval_entries: int = max_retrieval_entries
        self.retrieval_ttl_seconds: float = retrieval_ttl_seconds

        # OrderedDict LRU caches: hash_key -> data
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._retrieval_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Telemetry metrics
        self.embedding_hits: int = 0
        self.embedding_misses: int = 0
        self.retrieval_hits: int = 0
        self.retrieval_misses: int = 0

        logger.info("⚡ CacheManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "CacheManager":
        """Thread-safe accessor for shared CacheManager Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # EMBEDDING CACHE METHODS
    # ------------------------------------------------------------------

    def _hash_text(self, text: str) -> str:
        """Generates SHA256 key for a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieves cached embedding vector for text string.

        Args:
            text (str): Input text passage.

        Returns:
            Optional[List[float]]: 384-d float vector if cached, else None.
        """
        key = self._hash_text(text)
        with self._lock:
            if key in self._embedding_cache:
                self._embedding_cache.move_to_end(key)
                self.embedding_hits += 1
                logger.debug(f"⚡ Embedding Cache HIT for key '{key}'")
                return self._embedding_cache[key]
            self.embedding_misses += 1
            return None

    def put_embedding(self, text: str, vector: List[float]) -> None:
        """
        Caches an embedding vector for a text string.

        Args:
            text (str): Input text passage.
            vector (List[float]): 384-d float vector.
        """
        key = self._hash_text(text)
        with self._lock:
            if key in self._embedding_cache:
                self._embedding_cache.move_to_end(key)
            self._embedding_cache[key] = vector
            if len(self._embedding_cache) > self.max_embedding_entries:
                self._embedding_cache.popitem(last=False)  # Evict LRU item

    # ------------------------------------------------------------------
    # RETRIEVAL RESULT CACHE METHODS
    # ------------------------------------------------------------------

    def get_retrieval_result(self, query: str, top_k: int = 5) -> Optional[Any]:
        """
        Retrieves cached RetrievalResult payload if within TTL.

        Args:
            query (str): Search query string.
            top_k (int): Candidate count requested.

        Returns:
            Optional[Any]: Cached RetrievalResult object if valid, else None.
        """
        key = f"{self._hash_text(query)}_top_{top_k}"
        now = time.time()
        with self._lock:
            if key in self._retrieval_cache:
                entry = self._retrieval_cache[key]
                if now - entry["timestamp"] <= self.retrieval_ttl_seconds:
                    self._retrieval_cache.move_to_end(key)
                    self.retrieval_hits += 1
                    logger.debug(f"⚡ Retrieval Query Cache HIT for query '{query[:20]}...'")
                    return entry["result"]
                else:
                    # Expired entry
                    del self._retrieval_cache[key]
            self.retrieval_misses += 1
            return None

    def put_retrieval_result(self, query: str, top_k: int, result: Any) -> None:
        """
        Caches a RetrievalResult payload for a search query.

        Args:
            query (str): Search query string.
            top_k (int): Candidate count requested.
            result (Any): RetrievalResult object.
        """
        key = f"{self._hash_text(query)}_top_{top_k}"
        with self._lock:
            if key in self._retrieval_cache:
                self._retrieval_cache.move_to_end(key)
            self._retrieval_cache[key] = {
                "result": result,
                "timestamp": time.time()
            }
            if len(self._retrieval_cache) > self.max_retrieval_entries:
                self._retrieval_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # INVALIDATION & STATISTICS
    # ------------------------------------------------------------------

    def invalidate_retrieval_cache(self) -> None:
        """Purges all cached query retrieval results (triggered on doc indexing/deletion)."""
        with self._lock:
            count = len(self._retrieval_cache)
            self._retrieval_cache.clear()
            logger.info(f"🧹 Invalidated Retrieval Cache ({count} entries purged).")

    def clear_all(self) -> None:
        """Purges both embedding and retrieval caches."""
        with self._lock:
            self._embedding_cache.clear()
            self._retrieval_cache.clear()
            self.embedding_hits = 0
            self.embedding_misses = 0
            self.retrieval_hits = 0
            self.retrieval_misses = 0
            logger.info("🧹 Purged all CacheManager LRU cache entries.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns telemetry cache hit ratios."""
        emb_total = self.embedding_hits + self.embedding_misses
        ret_total = self.retrieval_hits + self.retrieval_misses

        emb_ratio = (self.embedding_hits / emb_total * 100.0) if emb_total > 0 else 0.0
        ret_ratio = (self.retrieval_hits / ret_total * 100.0) if ret_total > 0 else 0.0

        return {
            "embedding_entries": len(self._embedding_cache),
            "embedding_hit_ratio_pct": round(emb_ratio, 2),
            "retrieval_entries": len(self._retrieval_cache),
            "retrieval_hit_ratio_pct": round(ret_ratio, 2),
        }
