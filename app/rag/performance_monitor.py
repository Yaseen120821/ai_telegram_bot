r"""
app/rag/performance_monitor.py - RAG Performance Monitoring & SLA Engine
==========================================================================

1. PURPOSE:
-----------
Measures execution latency across RAG subsystem operations (Document Parsing, Chunking, Embedding Generation,
Vector Retrieval, Composite Ranking, Context Building) and logs performance warnings when SLA thresholds are exceeded.

2. WHY IT EXISTS (OBSERVABILITY & SLA ENFORCEMENT):
---------------------------------------------------
System latency directly impacts user experience. `PerformanceMonitor` records operation durations, provides a context manager
(`measure_stage`), and alerts developers if retrieval SLA limits ($> 300\text{ms}$) are breached.

3. RESPONSIBILITIES:
--------------------
- Measure stage execution duration in milliseconds.
- Record latency history for performance analytics.
- Issue SLA warnings when latency exceeds configured threshold limit (`PERFORMANCE_THRESHOLD_MS = 300.0`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Interfaced by `RAGManager`, `RetrievalManager`, `EmbeddingManager`, and `DocumentManager`.

5. COMPLETE CODE:
-----------------
"""

import time
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Generator

from app.rag.rag_config import PERFORMANCE_THRESHOLD_MS

logger = logging.getLogger("sana_ai.rag.performance")


class PerformanceMonitor:
    """
    Thread-safe Singleton tracking execution durations and SLA threshold warnings.
    """
    _instance: Optional["PerformanceMonitor"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        if PerformanceMonitor._instance is not None:
            raise RuntimeError("PerformanceMonitor is a Singleton! Use `PerformanceMonitor.get_instance()` instead.")

        self._latencies: Dict[str, List[float]] = {}
        logger.info("⏱️ PerformanceMonitor Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        """Thread-safe accessor for shared PerformanceMonitor Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @contextmanager
    def measure_stage(self, stage_name: str, sla_threshold_ms: float = PERFORMANCE_THRESHOLD_MS) -> Generator[None, None, None]:
        """
        Context manager measuring execution duration of a stage block.

        Args:
            stage_name (str): Operational stage identifier.
            sla_threshold_ms (float): Max acceptable duration limit in milliseconds.
        """
        start_t = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_t) * 1000.0
            self.record_latency(stage_name, elapsed_ms)
            if elapsed_ms > sla_threshold_ms:
                logger.warning(
                    f"⚠️ Performance SLA Threshold Exceeded in stage '{stage_name}' | "
                    f"Elapsed: {elapsed_ms:.2f}ms > SLA Threshold {sla_threshold_ms:.1f}ms"
                )
            else:
                logger.debug(f"⏱️ Stage '{stage_name}' completed in {elapsed_ms:.2f}ms")

    def record_latency(self, stage_name: str, latency_ms: float) -> None:
        """
        Records a stage latency duration.

        Args:
            stage_name (str): Operational stage identifier.
            latency_ms (float): Execution duration in milliseconds.
        """
        with self._lock:
            if stage_name not in self._latencies:
                self._latencies[stage_name] = []
            self._latencies[stage_name].append(latency_ms)
            # Keep last 500 samples
            if len(self._latencies[stage_name]) > 500:
                self._latencies[stage_name].pop(0)

    def get_average_latency(self, stage_name: str) -> float:
        """Calculates average latency for a stage."""
        with self._lock:
            samples = self._latencies.get(stage_name, [])
            if not samples:
                return 0.0
            return sum(samples) / len(samples)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Returns summary of average latencies across stages."""
        with self._lock:
            summary = {}
            for stage, samples in self._latencies.items():
                if samples:
                    summary[stage] = {
                        "avg_ms": round(sum(samples) / len(samples), 2),
                        "min_ms": round(min(samples), 2),
                        "max_ms": round(max(samples), 2),
                        "sample_count": len(samples)
                    }
            return summary
