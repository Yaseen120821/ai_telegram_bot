"""
app/emotion/emotion_manager.py - Central Emotion Subsystem Manager Facade
==========================================================================

1. PURPOSE:
-----------
Acts as the central Singleton orchestrator facade for Chapter 7 (Emotional Intelligence System).
Coordinates emotion detection, classification, RAM context timeline tracking, transition analysis, importance scoring,
and selective long-term memory persistence.

2. WHY IT EXISTS (FACADE PATTERN):
----------------------------------
Provides a clean, unified API for external modules (`ConversationManager`, Telegram handlers).
Hides internal details of PyTorch inference, Softmax math, transition history tracking, and Chapter 6 SQLite memory bridges.

3. RESPONSIBILITIES:
--------------------
- Process user message text and return verified `EmotionResult`.
- Provide high-level context getters (`get_current_emotion`, `get_previous_emotion`, `get_latest_transition`, `get_emotion_history`).
- Provide emotion storage decision methods (`should_store_emotion`, `store_emotion`).
- Generate conversational summaries (`get_emotional_summary`).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Orchestrates `EmotionDetector`, `EmotionClassifier`, `EmotionContextTracker`, and `EmotionMemory`.
- Interfaced by Telegram bot handlers and conversation managers.

5. COMPLETE CODE:
-----------------
"""

import logging
import threading
from typing import Optional, List

from app.emotion.emotion_detector import EmotionDetector
from app.emotion.emotion_classifier import EmotionClassifier
from app.emotion.emotion_context import EmotionContextTracker
from app.emotion.emotion_memory import EmotionMemory
from app.emotion.emotion_models import (
    EmotionResult,
    EmotionContext,
    PredictionResult,
    EmotionTransition,
    EmotionSummary
)
from app.emotion.emotion_utils import EmotionUtils
from app.memory import MemoryItem

logger = logging.getLogger("sana_ai.emotion.manager")


class EmotionManager:
    """
    Thread-safe Singleton orchestrator for SANA AI Emotional Intelligence System.
    """
    _instance: Optional["EmotionManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor enforcing Singleton pattern."""
        if EmotionManager._instance is not None:
            raise RuntimeError(
                "EmotionManager is a Singleton! Use `EmotionManager.get_instance()` instead."
            )

        self.detector: EmotionDetector = EmotionDetector.get_instance()
        self.classifier: EmotionClassifier = EmotionClassifier(detector=self.detector)
        self.context_tracker: EmotionContextTracker = EmotionContextTracker()
        self.emotion_memory: EmotionMemory = EmotionMemory()
        from app.emotion.emotion_analytics import EmotionAnalyticsTracker
        self.analytics_tracker: EmotionAnalyticsTracker = EmotionAnalyticsTracker()

        logger.info("🎭 EmotionManager Singleton initialized successfully.")

    @classmethod
    def get_instance(cls) -> "EmotionManager":
        """
        Thread-safe accessor for the shared EmotionManager Singleton instance.

        Returns:
            EmotionManager: Shared Singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # PUBLIC API METHODS
    # ------------------------------------------------------------------

    def detect_emotion(self, user_text: str) -> PredictionResult:
        """
        Runs raw emotion detection inference on user text.

        Args:
            user_text (str): Input text message statement.

        Returns:
            PredictionResult: PredictionResult dataclass instance.
        """
        return self.detector.predict_emotion(user_text)

    def process_emotion(self, user_id: str, user_text: str) -> EmotionResult:
        """
        Primary pipeline method: classifies emotion, updates ephemeral RAM context, tracks transitions,
        passively logs analytics telemetry, and selectively persists high-importance milestones to SQLite.
        Includes robust try-except error recovery blocks to guarantee graceful degradation.

        Args:
            user_id (str): Telegram User ID string.
            user_text (str): Incoming user statement text.

        Returns:
            EmotionResult: Finalized EmotionResult dataclass object.
        """
        import time
        from app.emotion.emotion_config import PERFORMANCE_THRESHOLD_MS
        from app.emotion.emotion_types import EmotionType, EmotionIntensity

        start_t = time.time()
        u_id = str(user_id)

        try:
            # 1. Classify emotion
            result: EmotionResult = self.classifier.classify_emotion(user_text)

            # 2. Update RAM context timeline and transitions
            self.context_tracker.add_emotion(u_id, result)

            # 3. Passively record telemetry metrics
            self.analytics_tracker.track_emotion(u_id, result)

            # 4. Selectively store high-importance milestones to SQLite
            self.emotion_memory.store_emotion(u_id, result, user_text)

            elapsed_ms = (time.time() - start_t) * 1000.0
            if elapsed_ms > PERFORMANCE_THRESHOLD_MS:
                logger.warning(
                    f"⚠️ Emotion Processing Performance Threshold Exceeded | "
                    f"User: {u_id} | Elapsed: {elapsed_ms:.2f}ms > SLA Threshold {PERFORMANCE_THRESHOLD_MS}ms"
                )

            return result
        except Exception as err:
            logger.error(
                f"❌ Error encountered during emotion processing for User {u_id} ({err}). "
                f"Applying Graceful Degradation strategy: defaulting to NEUTRAL emotion.",
                exc_info=True
            )
            # Graceful Fallback Result
            fallback_res = EmotionResult(
                primary_emotion=EmotionType.NEUTRAL.value,
                intensity=EmotionIntensity.LOW.value,
                confidence=0.95,
                empathy_directive=""
            )
            try:
                self.context_tracker.add_emotion(u_id, fallback_res)
            except Exception:
                pass
            return fallback_res

    def analyze_user_emotion(self, user_id: str, user_text: str) -> EmotionResult:
        """Alias method for process_emotion for backward compatibility."""
        return self.process_emotion(user_id, user_text)

    def get_current_emotion(self, user_id: str) -> str:
        """
        Retrieves the current active primary emotion string for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            str: Primary emotion string.
        """
        ctx = self.context_tracker.get_user_context(str(user_id))
        return ctx.current_emotion

    def get_previous_emotion(self, user_id: str) -> Optional[str]:
        """
        Retrieves the previous turn's emotion string for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            Optional[str]: Preceding emotion string or None.
        """
        ctx = self.context_tracker.get_user_context(str(user_id))
        return ctx.previous_emotion

    def get_latest_transition(self, user_id: str) -> Optional[EmotionTransition]:
        """
        Retrieves the most recent EmotionTransition record for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            Optional[EmotionTransition]: Latest transition or None.
        """
        return self.context_tracker.get_latest_transition(str(user_id))

    def get_emotion_history(self, user_id: str) -> List[EmotionResult]:
        """
        Retrieves the sliding window list of recent EmotionResult records for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            List[EmotionResult]: List of EmotionResult objects.
        """
        ctx = self.context_tracker.get_user_context(str(user_id))
        return ctx.history

    def should_store_emotion(self, user_id: str, result: EmotionResult, text: str) -> bool:
        """
        Checks whether an EmotionResult meets the importance threshold for SQLite persistence.

        Args:
            user_id (str): Telegram User ID string.
            result (EmotionResult): EmotionResult object.
            text (str): Input trigger user message.

        Returns:
            bool: True if event should be saved to SQLite.
        """
        return self.emotion_memory.should_store_emotion(result, text)

    def store_emotion(self, user_id: str, result: EmotionResult, text: str) -> Optional[MemoryItem]:
        """
        Stores an emotional milestone to SQLite database via Chapter 6 MemoryManager.

        Args:
            user_id (str): Telegram User ID string.
            result (EmotionResult): EmotionResult object.
            text (str): Input trigger message.

        Returns:
            Optional[MemoryItem]: Saved MemoryItem if stored, else None.
        """
        return self.emotion_memory.store_emotion(str(user_id), result, text)

    def get_empathy_directive(self, user_id: str, current_emotion: Optional[str] = None) -> str:
        """
        Retrieves the formatted System Prompt empathy directive for user_id.

        Args:
            user_id (str): Telegram User ID string.
            current_emotion (Optional[str]): Optional explicit emotion string override.

        Returns:
            str: Formatted empathy directive string.
        """
        emotion = current_emotion or self.get_current_emotion(user_id)
        return EmotionUtils.get_empathy_prompt(emotion)

    def get_user_emotion_context(self, user_id: str) -> EmotionContext:
        """
        Retrieves active EmotionContext timeline for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            EmotionContext: EmotionContext dataclass instance.
        """
        return self.context_tracker.get_user_context(str(user_id))

    def get_emotional_summary(self, user_id: str) -> EmotionSummary:
        """
        Generates a high-level EmotionSummary for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            EmotionSummary: Summary object.
        """
        return self.context_tracker.get_emotional_summary(str(user_id))

    def clear_emotion_context(self, user_id: str) -> None:
        """
        Clears stored emotion context timeline for user_id.

        Args:
            user_id (str): Telegram User ID string.
        """
        self.context_tracker.clear_user_context(str(user_id))
        self.analytics_tracker.clear_user_analytics(str(user_id))

    def get_user_analytics(self, user_id: str) -> dict:
        """
        Retrieves detailed emotion telemetry ratios and statistics for user_id.

        Args:
            user_id (str): Telegram User ID string.

        Returns:
            dict: Telemetry metrics dictionary.
        """
        return self.analytics_tracker.get_detailed_analytics(str(user_id))
