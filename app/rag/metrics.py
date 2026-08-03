r"""
app/rag/metrics.py - RAG Metrics & Telemetry Collector
======================================================

1. PURPOSE:
-----------
Collects and aggregates operational metrics across SANA AI's RAG Subsystem.

2. WHY IT EXISTS (SYSTEM TELEMETRY & MONITORING):
-------------------------------------------------
Production AI systems require metric tracking to monitor indexing volume, search frequency, vector dimensions,
cache efficiency, and performance trends over time.

3. RESPONSIBILITIES:
--------------------
- Track document ingestion and chunk indexing totals.
- Track total vector embedding generation counts.
- Track cache hit and miss ratios.
- Aggregate system performance metrics into a clean dictionary.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Interfaced by `RAGManager`, `IndexManager`, `CacheManager`, `EmbeddingManager`, and `RetrievalManager`.

5. COMPLETE CODE:
-----------------
"""

import logging
import threading
from typing import Optional, Dict, Any

from app.rag.cache_manager import CacheManager
from app.rag.performance_monitor import PerformanceMonitor

logger = logging.getLogger("sana_ai.rag.metrics")


class MetricsCollector:
    """
    Thread-safe Singleton telemetry collector aggregating RAG operational metrics.
    """
    _instance: Optional["MetricsCollector"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if MetricsCollector._instance is not None:
            raise RuntimeError("MetricsCollector is a Singleton! Use `MetricsCollector.get_instance()` instead.")

        self.total_queries: int = 0
        self.successful_retrievals: int = 0
        self.failed_retrievals: int = 0
        self.documents_indexed_total: int = 0
        self.chunks_generated_total: int = 0

        logger.info("📊 MetricsCollector Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Thread-safe accessor for shared MetricsCollector Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_query(self, success: bool = True) -> None:
        """Records a search query execution."""
        with self._lock:
            self.total_queries += 1
            if success:
                self.successful_retrievals += 1
            else:
                self.failed_retrievals += 1

    def record_indexing(self, chunks_count: int) -> None:
        """Records document indexing totals."""
        with self._lock:
            self.documents_indexed_total += 1
            self.chunks_generated_total += chunks_count

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry summary dictionary."""
        cache_stats = CacheManager.get_instance().get_stats()
        perf_stats = PerformanceMonitor.get_instance().get_performance_summary()

        with self._lock:
            return {
                "queries": {
                    "total": self.total_queries,
                    "successful": self.successful_retrievals,
                    "failed": self.failed_retrievals
                },
                "indexing": {
                    "documents_indexed": self.documents_indexed_total,
                    "chunks_generated": self.chunks_generated_total
                },
                "cache": cache_stats,
                "performance": perf_stats
            }
