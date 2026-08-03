"""
app/emotion/emotion_memory.py - Selective Emotional Memory Persistence Bridge
================================================================================

1. PURPOSE:
-----------
Bridges Chapter 7 (Emotion AI) with Chapter 6 (`MemoryManager`), selectively saving high-importance emotional events
(e.g., severe anxiety, long-term depression, major career milestones) to SQLite disk storage while ignoring transient chatter.

2. WHY IT EXISTS (SELECTIVE PERSISTENCE & IMPORTANCE SCORING):
--------------------------------------------------------------
Storing every minor emotion fills SQLite with noise. `EmotionMemory` evaluates every classified emotion against an
Importance Scale (1 to 10). Only emotional events with `importance >= MIN_MEMORY_IMPORTANCE_THRESHOLD` (6) are written to disk.

3. RESPONSIBILITIES:
--------------------
- Calculate emotion importance scores based on `EMOTION_IMPORTANCE_MAP` (1-10 scale) and intensity level.
- Implement `should_store_emotion(user_id, result, trigger_text)`.
- Format emotional memory items and invoke `MemoryManager.get_instance().save_memory()`.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `EmotionResult` and `EmotionMemoryRecord` from `app/emotion/emotion_models.py`.
- Uses `EMOTION_IMPORTANCE_MAP` and `MIN_MEMORY_IMPORTANCE_THRESHOLD` from `app/emotion/emotion_config.py`.
- Interfaced via `MemoryManager.get_instance()` to Chapter 6 SQLite database.
- Called by `app/emotion/emotion_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import Optional

from app.emotion.emotion_models import EmotionResult, EmotionMemoryRecord
from app.emotion.emotion_types import EmotionIntensity, EmotionType
from app.emotion.emotion_config import EMOTION_IMPORTANCE_MAP, MIN_MEMORY_IMPORTANCE_THRESHOLD
from app.memory import MemoryManager, MemoryItem, MemoryCategory

logger = logging.getLogger("sana_ai.emotion.memory")


class EmotionMemory:
    """
    Selective Emotional Memory Persistence Bridge interfacing with Chapter 6 MemoryManager.
    """

    def calculate_importance(self, result: EmotionResult) -> int:
        """
        Calculates an emotion importance score on a scale from 1 to 10.

        Args:
            result (EmotionResult): EmotionResult object.

        Returns:
            int: Importance score (1 to 10).
        """
        base_importance = EMOTION_IMPORTANCE_MAP.get(result.primary_emotion.lower(), 2)

        # Apply intensity bonus
        if result.intensity == EmotionIntensity.EXTREME.value:
            base_importance += 2
        elif result.intensity == EmotionIntensity.HIGH.value:
            base_importance += 1

        # Clamp strictly between 1 and 10
        return max(1, min(10, base_importance))

    def should_store_emotion(self, result: EmotionResult, trigger_text: str) -> bool:
        """
        Determines whether an emotional event is significant enough to persist to SQLite disk storage.

        Args:
            result (EmotionResult): EmotionResult object.
            trigger_text (str): Input trigger user message.

        Returns:
            bool: True if importance >= MIN_MEMORY_IMPORTANCE_THRESHOLD (6).
        """
        # Neutral emotions are NEVER stored to disk
        if result.primary_emotion == EmotionType.NEUTRAL.value:
            return False

        importance = self.calculate_importance(result)
        return importance >= MIN_MEMORY_IMPORTANCE_THRESHOLD

    def store_emotion(
        self,
        user_id: str,
        result: EmotionResult,
        trigger_text: str
    ) -> Optional[MemoryItem]:
        """
        Persists a high-importance emotional milestone to Chapter 6 SQLite database via MemoryManager.

        Args:
            user_id (str): Telegram User ID string.
            result (EmotionResult): EmotionResult object.
            trigger_text (str): Input trigger message text.

        Returns:
            Optional[MemoryItem]: Saved MemoryItem if stored, else None.
        """
        importance = self.calculate_importance(result)
        should_store = self.should_store_emotion(result, trigger_text)

        rec = EmotionMemoryRecord(
            user_id=str(user_id),
            emotion=result.primary_emotion,
            importance=importance,
            trigger_text=trigger_text,
            stored=should_store
        )

        if not should_store:
            logger.debug(
                f"ℹ️ Transient Emotion Ignored for SQLite [User: {user_id} | Emotion: '{result.primary_emotion}' | "
                f"Importance: {importance}/10 < Threshold {MIN_MEMORY_IMPORTANCE_THRESHOLD}]"
            )
            return None

        memory_key = f"emotional_milestone_{result.primary_emotion}"
        memory_value = (
            f"Expressed {result.intensity} {result.primary_emotion} (Importance: {importance}/10) "
            f"during conversation ('{trigger_text[:50]}...')"
        )

        item = MemoryItem(
            user_id=str(user_id),
            category=MemoryCategory.CUSTOM.value,
            memory_key=memory_key,
            memory_value=memory_value,
            importance=importance,
            confidence=result.confidence,
            source="emotion_memory_bridge"
        )

        try:
            saved_item = MemoryManager.get_instance().save_memory(item)
            logger.info(
                f"💾 Persisted Emotional Milestone to SQLite [User {user_id} | Key: '{memory_key}' | "
                f"Importance: {importance}/10]"
            )
            return saved_item
        except Exception as err:
            logger.error(f"Failed to persist emotional milestone for User {user_id}: {err}")
            return None
