"""
app/emotion package initializer.
Exposes public API for Emotion AI analysis, context tracking, types, models, analytics, and manager.
"""

from app.emotion.emotion_config import (
    EMPATHY_PROMPT_MAP,
    DEFAULT_MIN_CONFIDENCE,
    EMOTION_IMPORTANCE_MAP,
    MIN_MEMORY_IMPORTANCE_THRESHOLD,
    PERFORMANCE_THRESHOLD_MS,
    INPUT_MAX_CHARS
)
from app.emotion.emotion_types import EmotionType, EmotionIntensity
from app.emotion.emotion_models import (
    EmotionResult,
    EmotionContext,
    PredictionResult,
    EmotionStatistics,
    EmotionTransition,
    EmotionSummary,
    EmotionMemoryRecord
)
from app.emotion.emotion_utils import EmotionUtils
from app.emotion.emotion_detector import EmotionDetector
from app.emotion.emotion_classifier import EmotionClassifier
from app.emotion.emotion_context import EmotionContextTracker
from app.emotion.emotion_memory import EmotionMemory
from app.emotion.emotion_analytics import EmotionAnalyticsTracker
from app.emotion.emotion_manager import EmotionManager

__all__ = [
    "EMPATHY_PROMPT_MAP",
    "DEFAULT_MIN_CONFIDENCE",
    "EMOTION_IMPORTANCE_MAP",
    "MIN_MEMORY_IMPORTANCE_THRESHOLD",
    "PERFORMANCE_THRESHOLD_MS",
    "INPUT_MAX_CHARS",
    "EmotionType",
    "EmotionIntensity",
    "EmotionResult",
    "EmotionContext",
    "PredictionResult",
    "EmotionStatistics",
    "EmotionTransition",
    "EmotionSummary",
    "EmotionMemoryRecord",
    "EmotionUtils",
    "EmotionDetector",
    "EmotionClassifier",
    "EmotionContextTracker",
    "EmotionMemory",
    "EmotionAnalyticsTracker",
    "EmotionManager"
]
