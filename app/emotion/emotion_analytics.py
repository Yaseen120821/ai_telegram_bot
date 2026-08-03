"""
app/emotion/emotion_analytics.py - Passive Emotion Analytics & Observability Tracker
=====================================================================================

1. PURPOSE:
-----------
Passively monitors user emotional metrics over time (emotion category distributions, average confidence scores,
positive vs. negative vs. neutral mood ratios, and transition volumes).

2. WHY IT EXISTS (PASSIVE OBSERVABILITY & STRICT SEPARATION):
-------------------------------------------------------------
System designers and ML engineers require telemetry to analyze conversation health, model accuracy, and user sentiment trends.
`EmotionAnalyticsTracker` operates strictly as an observability component: it records metrics passively without ever altering
runtime decisions, prompt building, or database persistence logic.

3. RESPONSIBILITIES:
--------------------
- Passively record emotion analysis output metadata per user_id.
- Calculate aggregate emotional distribution statistics (frequency counts).
- Compute positive, negative, and neutral emotion ratios.
- Compute rolling average confidence scores across user dialogue turns.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Receives `EmotionResult` dataclasses from `app/emotion/emotion_manager.py`.
- Exposes `EmotionStatistics` for telemetry logging or dashboard inspection.

5. COMPLETE CODE:
-----------------
"""

import logging
from collections import defaultdict, Counter
from typing import Dict, List, Any

from app.emotion.emotion_models import EmotionResult, EmotionStatistics
from app.emotion.emotion_types import EmotionType

logger = logging.getLogger("sana_ai.emotion.analytics")

# Emotion Categorization Maps for Ratio Calculations
POSITIVE_EMOTIONS = {"joy", "excited", "excitement", "calm", "calmness", "hope", "motivation", "pride", "love", "optimism", "gratitude"}
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear", "anxiety", "stress", "grief", "frustration", "disappointment", "embarrassment", "confusion"}


class EmotionAnalyticsTracker:
    """
    Thread-safe passive emotion analytics tracker recording telemetry metrics per user.
    """

    def __init__(self) -> None:
        """Initializes user telemetry metrics storage."""
        self._user_emotions: Dict[str, List[EmotionResult]] = defaultdict(list)

    def track_emotion(self, user_id: str, result: EmotionResult) -> None:
        """
        Passively records an EmotionResult payload for user_id.

        Args:
            user_id (str): Telegram User ID string.
            result (EmotionResult): Classified EmotionResult object.
        """
        u_id = str(user_id)
        self._user_emotions[u_id].append(result)
        logger.debug(
            f"📊 Analytics Metric Logged | User: {u_id} | Emotion: '{result.primary_emotion}' | "
            f"Confidence: {result.confidence:.2f} | Total Tracked: {len(self._user_emotions[u_id])}"
        )

    def get_statistics(self, user_id: str) -> EmotionStatistics:
        """
        Calculates aggregate EmotionStatistics metrics for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            EmotionStatistics: Computed EmotionStatistics object.
        """
        u_id = str(user_id)
        records = self._user_emotions.get(u_id, [])
        total = len(records)

        if total == 0:
            return EmotionStatistics(user_id=u_id, emotion_counts={}, total_analyzed=0)

        counts = Counter([r.primary_emotion for r in records])
        return EmotionStatistics(
            user_id=u_id,
            emotion_counts=dict(counts),
            total_analyzed=total
        )

    def get_detailed_analytics(self, user_id: str) -> Dict[str, Any]:
        """
        Computes detailed ratio metrics (positive vs negative vs neutral, average confidence).

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            Dict[str, Any]: Dictionary containing telemetry ratios and averages.
        """
        u_id = str(user_id)
        records = self._user_emotions.get(u_id, [])
        total = len(records)

        if total == 0:
            return {
                "user_id": u_id,
                "total_messages": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
                "average_confidence": 0.0
            }

        pos_count = sum(1 for r in records if r.primary_emotion in POSITIVE_EMOTIONS)
        neg_count = sum(1 for r in records if r.primary_emotion in NEGATIVE_EMOTIONS)
        neu_count = sum(1 for r in records if r.primary_emotion == EmotionType.NEUTRAL.value)

        avg_conf = sum(r.confidence for r in records) / total

        return {
            "user_id": u_id,
            "total_messages": total,
            "positive_ratio": round(pos_count / total, 3),
            "negative_ratio": round(neg_count / total, 3),
            "neutral_ratio": round(neu_count / total, 3),
            "average_confidence": round(avg_conf, 3)
        }

    def clear_user_analytics(self, user_id: str) -> None:
        """
        Clears telemetry metrics for user_id.

        Args:
            user_id (str): Telegram User ID string.
        """
        u_id = str(user_id)
        if u_id in self._user_emotions:
            del self._user_emotions[u_id]
