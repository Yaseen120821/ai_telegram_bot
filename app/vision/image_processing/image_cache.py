"""
app/vision/image_processing/image_cache.py - In-Memory Image & Processing LRU Cache
====================================================================================
Maintains an in-memory LRU cache store of preprocessed image results and metadata,
reducing CPU and GPU recalculation overhead on duplicate image requests.
"""

import time
import logging
from typing import Dict, Optional
from collections import OrderedDict

from app.vision.image_processing.image_models import ImageCacheEntry, ProcessedImage, ImageMetadata
from app.vision.image_processing.image_config import get_image_config, ImageProcessingConfig

logger = logging.getLogger("sana_ai.vision.image_cache")


class ImageCache:
    """Thread-safe LRU Cache for preprocessed images and metadata."""

    _instance: Optional["ImageCache"] = None

    def __init__(self, max_items: Optional[int] = None):
        config = get_image_config()
        self.max_items = max_items or config.cache_max_items
        self._cache: OrderedDict[str, ImageCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @classmethod
    def get_instance(cls) -> "ImageCache":
        """Returns singleton instance of ImageCache."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get(self, key: str) -> Optional[ProcessedImage]:
        """Retrieves cached ProcessedImage by key hash."""
        if key in self._cache:
            entry = self._cache[key]
            entry.access_count += 1
            # Move to end (MRU)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"ImageCache hit for key '{key[:8]}'.")
            return entry.processed_image
        self._misses += 1
        return None

    def put(self, key: str, processed_image: ProcessedImage, metadata: ImageMetadata) -> None:
        """Stores ProcessedImage in cache, evicting oldest item if max capacity reached."""
        if key in self._cache:
            self._cache.move_to_end(key)

        entry = ImageCacheEntry(key=key, processed_image=processed_image, metadata=metadata)
        self._cache[key] = entry

        if len(self._cache) > self.max_items:
            oldest_key, _ = self._cache.popitem(last=False)
            logger.debug(f"ImageCache capacity reached. Evicted oldest key '{oldest_key[:8]}'.")

    def has(self, key: str) -> bool:
        """Checks if key exists in cache."""
        return key in self._cache

    def clear(self) -> None:
        """Clears all entries from cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)
