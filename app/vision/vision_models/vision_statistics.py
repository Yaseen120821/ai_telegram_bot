"""
app/vision/vision_models/vision_statistics.py - Operational Telemetry Tracker
=============================================================================
Records timing metrics, success rates, failure counters, and latency averages for Vision AI inference.
"""

from typing import Optional
from app.vision.vision_models.vision_schemas import VisionStatistics


class VisionStatisticsTracker:
    """Telemetry collector for Vision AI performance metrics."""

    _instance: Optional["VisionStatisticsTracker"] = None

    def __init__(self):
        self.stats = VisionStatistics()

    @classmethod
    def get_instance(cls) -> "VisionStatisticsTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_request(self, success: bool, latency_ms: float, cache_hit: bool = False) -> None:
        self.stats.total_requests += 1
        if cache_hit:
            self.stats.cache_hits += 1

        if success:
            self.stats.successful_requests += 1
            self.stats.total_inference_time_ms += latency_ms
            total_time_records = max(1, self.stats.successful_requests)
            self.stats.average_latency_ms = self.stats.total_inference_time_ms / total_time_records
        else:
            self.stats.failed_requests += 1

    def get_statistics(self) -> VisionStatistics:
        return self.stats

    def reset(self) -> None:
        self.stats = VisionStatistics()
