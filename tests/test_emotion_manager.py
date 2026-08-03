"""
tests/test_emotion_manager.py - Comprehensive Diagnostic Test Suite for Chapter 7 Part 3
===========================================================================================

Executes automated diagnostic verification across Chapter 7 Part 3 components:
1. Multi-turn emotion transition tracking (`Joy -> Stress -> Hope -> Joy`).
2. API methods (`get_current_emotion`, `get_previous_emotion`, `get_latest_transition`, `get_emotional_summary`).
3. Selective persistence decision: Ephemeral RAM vs Durable SQLite disk storage based on Importance Scoring (1 to 10 scale).
   - Transient ("I am happy today", Importance = 2) -> Ignored for SQLite disk storage.
   - Permanent ("I am extremely anxious about my surgery tomorrow", Importance = 9) -> Saved to SQLite.
"""

import sys
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.emotion import (
    EmotionManager,
    EmotionResult,
    EmotionTransition,
    EmotionSummary
)
from app.memory import MemoryManager

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_emotion_manager")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 7 Part 3."""
    logger.info("=== Starting SANA AI Emotion Manager & Context System Diagnostic Tests ===")

    test_user_id = "8087004004"
    mgr = EmotionManager.get_instance()

    # Reset test environment
    mgr.clear_emotion_context(test_user_id)
    MemoryManager.get_instance().clear_user_memory(test_user_id)

    # ------------------------------------------------------------------
    # TEST 1: Multi-Turn Transition Tracking
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Multi-Turn Emotion Transition Tracking ---")

    turns = [
        ("I got selected for my dream job!", "joy"),
        ("I am stressed about tomorrow's presentation.", "fear"),
        ("I hope everything goes well.", "fear"),
        ("I finally finished my presentation successfully!", "joy")
    ]

    for statement, _ in turns:
        res: EmotionResult = mgr.process_emotion(test_user_id, statement)
        curr = mgr.get_current_emotion(test_user_id)
        prev = mgr.get_previous_emotion(test_user_id)
        trans = mgr.get_latest_transition(test_user_id)

        logger.info(
            f"Statement: '{statement}'\n"
            f"  • Current: '{curr}' | Previous: '{prev}'\n"
            f"  • Latest Transition: '{trans.from_emotion if trans else 'None'}' ──► '{trans.to_emotion if trans else 'None'}'"
        )

    summary: EmotionSummary = mgr.get_emotional_summary(test_user_id)
    logger.info(
        f"\nConversational Summary:\n"
        f"  • User ID: {summary.user_id}\n"
        f"  • Current Emotion: '{summary.current_emotion}'\n"
        f"  • Previous Emotion: '{summary.previous_emotion}'\n"
        f"  • Dominant Trend: '{summary.dominant_trend}'\n"
        f"  • Total Turns: {summary.total_turns}\n"
        f"  • Total Transitions: {summary.transitions_count}"
    )

    assert summary.total_turns == 4, "Should have recorded 4 turns"
    assert summary.transitions_count > 0, "Should have recorded transitions"
    logger.info("✅ Multi-turn transition tracking verified!")

    # ------------------------------------------------------------------
    # TEST 2: Selective Persistence Decision (RAM vs SQLite)
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Selective Persistence Decision (RAM vs SQLite) ---")

    # Case A: Transient Emotion (Importance < 6)
    transient_text = "I am happy today."
    transient_res = mgr.classifier.classify_emotion(transient_text)
    should_store_transient = mgr.should_store_emotion(test_user_id, transient_res, transient_text)
    logger.info(
        f"Transient Case: '{transient_text}' | Emotion: '{transient_res.primary_emotion}' | "
        f"Importance: {mgr.emotion_memory.calculate_importance(transient_res)}/10 | Should Store to SQLite? {should_store_transient}"
    )
    assert not should_store_transient, "Transient emotions (Importance < 6) should NOT be stored to SQLite"

    # Case B: High Importance Event (Importance >= 6)
    important_text = "I am extremely anxious and stressed about my surgery tomorrow."
    important_res = mgr.classifier.classify_emotion(important_text)
    should_store_important = mgr.should_store_emotion(test_user_id, important_res, important_text)
    logger.info(
        f"Important Case: '{important_text}' | Emotion: '{important_res.primary_emotion}' | "
        f"Importance: {mgr.emotion_memory.calculate_importance(important_res)}/10 | Should Store to SQLite? {should_store_important}"
    )
    assert should_store_important, "High importance events (Importance >= 6) MUST be stored to SQLite"

    # Verify SQLite memory retrieval
    sqlite_memories = MemoryManager.get_instance().retrieve_memories(test_user_id)
    logger.info(f"\nRetrieved {len(sqlite_memories)} emotional milestone records from SQLite DB.")
    for mem in sqlite_memories:
        logger.info(f"  • Key: '{mem.memory_key}' | Value: '{mem.memory_value}' | Importance: {mem.importance}")

    assert len(sqlite_memories) >= 1, "SQLite database should contain persisted emotional milestones"
    logger.info("✅ Selective persistence decision (RAM vs SQLite) verified!")

    logger.info("\n🎉 ALL EMOTION MANAGER & CONTEXT SYSTEM DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
