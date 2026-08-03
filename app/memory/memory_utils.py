"""
app/memory/memory_utils.py - Text Sanitization, Validation & Helper Utilities
================================================================================

1. PURPOSE:
-----------
Provides validation and text normalization functions for sanitizing memory keys, generating ISO 8601 timestamps,
validating importance/confidence ranges, and filtering out transient prompts.

2. WHY IT EXISTS:
-----------------
Raw input strings (like `" My Name is Yaseen! "`) contain whitespace, trailing punctuation, and non-standard capitalization.
`memory_utils.py` normalizes strings so property keys match reliably in SQLite queries.

3. RESPONSIBILITIES:
--------------------
- Clean and sanitize input text.
- Normalize memory property keys (e.g. `'Favorite Language'` -> `'favorite_language'`).
- Validate and clamp `importance` scores (1 to 10) and `confidence` scores (0.0 to 1.0).
- Validate category taxonomy membership against `MemoryCategory`.
- Filter out transient non-memory prompts (greetings, general questions, code requests, weather).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/memory/memory_classifier.py`, `app/memory/memory_store.py`, `app/memory/memory_retriever.py`,
  and `app/memory/memory_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional
from app.memory.memory_types import MemoryCategory

logger = logging.getLogger("sana_ai.memory.utils")

# Transient message triggers (never save these as permanent memories)
TRANSIENT_PREFIXES = (
    "hello", "hi", "hey", "thanks", "thank you", "good morning", "good evening",
    "what is", "who is", "how do", "explain", "write code", "tell me", "show me",
    "can you", "where is", "why does", "help me", "today's weather", "weather is"
)


class MemoryUtils:
    """
    Utility helpers for text normalization, validation, and transient prompt detection.
    """

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generates a standard ISO 8601 UTC timestamp string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def clean_user_input(text: str) -> str:
        """
        Strips surrounding whitespace and unwanted trailing punctuation from user input.

        Args:
            text (str): Raw input text.

        Returns:
            str: Cleaned string.
        """
        if not text:
            return ""
        cleaned = text.strip().strip(".!?,:;")
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def normalize_key_fact(key: str) -> str:
        """
        Normalizes a property key into snake_case (e.g. 'Favorite Language' -> 'favorite_language').

        Args:
            key (str): Key property string.

        Returns:
            str: Normalized snake_case key string.
        """
        cleaned = MemoryUtils.clean_user_input(key).lower()
        cleaned = re.sub(r"[^\w\s]", "", cleaned)
        return re.sub(r"\s+", "_", cleaned)

    @staticmethod
    def validate_category(category: str) -> bool:
        """
        Validates if a category string is a valid member of MemoryCategory.

        Args:
            category (str): Category string to check.

        Returns:
            bool: True if valid taxonomy member, False otherwise.
        """
        if not category:
            return False
        valid_values = {c.value for c in MemoryCategory}
        return category.lower() in valid_values

    @staticmethod
    def validate_importance(importance: int) -> int:
        """
        Clamps importance score within the valid range [1, 10].

        Args:
            importance (int): Raw importance score.

        Returns:
            int: Clamped importance integer between 1 and 10.
        """
        try:
            val = int(importance)
            return max(1, min(10, val))
        except (ValueError, TypeError):
            return 5

    @staticmethod
    def validate_confidence(confidence: float) -> float:
        """
        Clamps confidence score within the valid range [0.0, 1.0].

        Args:
            confidence (float): Raw confidence score.

        Returns:
            float: Clamped confidence float between 0.0 and 1.0.
        """
        try:
            val = float(confidence)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.95

    @staticmethod
    def is_transient_prompt(text: str) -> bool:
        """
        Detects if text is a transient query, greeting, or scratch request.

        Args:
            text (str): Incoming user text.

        Returns:
            bool: True if prompt is transient noise and should not be stored.
        """
        if not text or len(text.strip()) < 3:
            return True

        lowered = text.strip().lower()

        # Questions ending in '?' without personal assertion keywords are transient
        if lowered.endswith("?") and not any(
            kw in lowered for kw in ["my ", "i am ", "i live ", "i work ", "i like ", "i prefer ", "i study ", "i want "]
        ):
            return True

        for prefix in TRANSIENT_PREFIXES:
            if lowered.startswith(prefix) and not any(
                kw in lowered for kw in ["my ", "i am ", "i prefer ", "i like ", "i live "]
            ):
                return True

        return False
