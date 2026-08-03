"""
app/emotion/emotion_utils.py - Emotion Helper & Validation Utilities
======================================================================

1. PURPOSE:
-----------
Provides text processing utilities, confidence normalization helpers, intensity calculation rules, and empathy
prompt directive formatters for the emotion layer.

2. WHY IT EXISTS:
-----------------
Decouples helper logic from detector and classifier core modules. Ensures scores are normalized, confidence ranges are
clamped, and empathy directives are formatted consistently.

3. RESPONSIBILITIES:
--------------------
- Clean and sanitize input text for emotion analysis.
- Clamp confidence scores within valid bounds [0.0, 1.0].
- Calculate `EmotionIntensity` based on confidence score cutoffs.
- Fetch formatted `[EMPATHY DIRECTIVE]` instructions from `EMPATHY_PROMPT_MAP`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/emotion/emotion_classifier.py` and `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import re
import logging
from app.emotion.emotion_config import (
    EMPATHY_PROMPT_MAP,
    EXTREME_INTENSITY_THRESHOLD,
    HIGH_INTENSITY_THRESHOLD,
    MEDIUM_INTENSITY_THRESHOLD
)
from app.emotion.emotion_types import EmotionIntensity, EmotionType

logger = logging.getLogger("sana_ai.emotion.utils")


class EmotionUtils:
    """
    Utility functions for emotion confidence normalization, intensity calculation, and empathy directives.
    """

    @staticmethod
    def clean_emotion_text(text: str) -> str:
        """
        Strips surrounding whitespace, normalizes Unicode (NFKC), removes null bytes and control characters,
        and truncates text to INPUT_MAX_CHARS for safe, performant emotion detection.

        Args:
            text (str): Input text statement.

        Returns:
            str: Cleaned, sanitized, and truncated string.
        """
        if not text:
            return ""

        try:
            import unicodedata
            from app.emotion.emotion_config import INPUT_MAX_CHARS

            # 1. Unicode NFKC Normalization
            normalized = unicodedata.normalize("NFKC", text)

            # 2. Strip null bytes and non-printable control characters
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", normalized).strip()

            # 3. Collapse multiple whitespace spaces into single space
            collapsed = re.sub(r"\s+", " ", cleaned)

            # 4. Truncate text boundary to INPUT_MAX_CHARS
            if len(collapsed) > INPUT_MAX_CHARS:
                logger.debug(f"✂️ User input truncated for emotion analysis ({len(collapsed)} > {INPUT_MAX_CHARS} chars).")
                collapsed = collapsed[:INPUT_MAX_CHARS]

            return collapsed
        except Exception as err:
            logger.warning(f"Error during emotion text sanitization ({err}). Returning basic trimmed text.")
            return text.strip()[:2000]

    @staticmethod
    def normalize_confidence(confidence: float) -> float:
        """
        Clamps confidence score strictly within [0.0, 1.0].

        Args:
            confidence (float): Raw score.

        Returns:
            float: Clamped score.
        """
        try:
            val = float(confidence)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.95

    @staticmethod
    def calculate_intensity(confidence: float) -> str:
        """
        Calculates EmotionIntensity enum value based on confidence thresholds.

        Args:
            confidence (float): Emotion confidence score.

        Returns:
            str: Intensity level string ('low', 'medium', 'high', 'extreme').
        """
        norm_score = EmotionUtils.normalize_confidence(confidence)
        if norm_score >= EXTREME_INTENSITY_THRESHOLD:
            return EmotionIntensity.EXTREME.value
        elif norm_score >= HIGH_INTENSITY_THRESHOLD:
            return EmotionIntensity.HIGH.value
        elif norm_score >= MEDIUM_INTENSITY_THRESHOLD:
            return EmotionIntensity.MEDIUM.value
        return EmotionIntensity.LOW.value

    @staticmethod
    def get_empathy_prompt(primary_emotion: str) -> str:
        """
        Retrieves the formatted System Prompt empathy directive for a primary emotion.

        Args:
            primary_emotion (str): Primary emotion string (e.g. 'sadness', 'frustration').

        Returns:
            str: Formatted empathy directive string.
        """
        key = primary_emotion.lower() if primary_emotion else "neutral"
        return EMPATHY_PROMPT_MAP.get(key, "")
