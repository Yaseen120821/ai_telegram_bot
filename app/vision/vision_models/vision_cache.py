"""
app/vision/vision_models/vision_cache.py - In-Memory Vision Inference Cache
=============================================================================
LRU cache store for vision model responses indexed by image checksum and task prompt,
preventing duplicate neural model inference on identical images.
"""

import logging
from typing import Optional, Dict
from collections import OrderedDict
from app.vision.vision_models.vision_schemas import VisionResponse
from app.vision.vision_models.vision_config import get_vision_model_config

logger = logging.getLogger("sana_ai.vision.models.cache")


class VisionCache:
    """Thread-safe LRU Cache for Vision AI responses."""

    _instance: Optional["VisionCache"] = None

    def __init__(self, max_capacity: Optional[int] = None):
        config = get_vision_model_config()
        self.max_capacity = max_capacity or config.cache_capacity
        self._cache: OrderedDict[str, VisionResponse] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @classmethod
    def get_instance(cls) -> "VisionCache":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, key: str) -> Optional[VisionResponse]:
        """Retrieves cached VisionResponse by key."""
        if key in self._cache:
            res = self._cache[key]
            self._cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"VisionCache hit for key '{key[:8]}'.")
            return res
        self.misses += 1
        return None

    def put(self, key: str, response: VisionResponse) -> None:
        """Stores VisionResponse in cache, evicting oldest record if capacity reached."""
        if key in self._cache:
            self._cache.move_to_end(key)

        self._cache[key] = response

        if len(self._cache) > self.max_capacity:
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"VisionCache capacity reached. Evicted key '{oldest_key[:8]}'.")

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
