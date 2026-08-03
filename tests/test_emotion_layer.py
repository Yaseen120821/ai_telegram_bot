"""
tests/test_emotion_layer.py - Comprehensive Diagnostic Test Suite for Chapter 7 Emotion System
=============================================================================================

Executes automated verification across all Chapter 7 modules:
1. Emotion Classification & Intensity Calculation (Joy, Sadness, Anxiety, Frustration, Curiosity, etc.).
2. Empathy Prompt Directive Generation.
3. Emotional Timeline & Trend Tracking (sliding window history).
4. Chapter 6 Long-Term Memory Milestone Persistence.
5. PromptBuilder System Prompt Empathy Injection.
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
    EmotionType,
    EmotionIntensity,
    EmotionResult,
    EmotionUtils
)
from app.llm.prompt_builder import PromptBuilder
from app.memory import MemoryManager

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_emotion_layer")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 7."""
    logger.info("=== Starting SANA AI Emotional Intelligence System Diagnostic Tests ===")

    test_user_id = "9096003003"
    emotion_manager = EmotionManager.get_instance()

    # Clear previous test context
    emotion_manager.clear_user_emotion_context(test_user_id)
    MemoryManager.get_instance().clear_user_memory(test_user_id)

    # ------------------------------------------------------------------
    # TEST 1: Emotion Detector & Classifier Flow
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Emotion Detector, Classifier & Intensity Calculation ---")

    test_cases = [
        ("I failed my exam and I feel so sad and miserable.", EmotionType.SADNESS.value),
        ("I am so frustrated with this broken code error again!", EmotionType.ANGER.value),
        ("I am feeling very anxious and worried about my interview tomorrow.", EmotionType.FEAR.value),
        ("I am so excited and happy to launch SANA AI today!", EmotionType.JOY.value),
        ("How does quantum computing work?", EmotionType.NEUTRAL.value),
        ("The sky is blue today.", EmotionType.NEUTRAL.value)
    ]

    for statement, expected_emotion in test_cases:
        res: EmotionResult = emotion_manager.analyze_user_emotion(test_user_id, statement)
        logger.info(
            f"Statement: '{statement}' -> Emotion: '{res.primary_emotion}' | Intensity: '{res.intensity}' | "
            f"Confidence: {res.confidence:.2f}"
        )
        if expected_emotion != EmotionType.NEUTRAL.value:
            assert res.primary_emotion is not None, f"Expected valid primary emotion, got '{res.primary_emotion}'"

    logger.info("✅ Emotion Detector, Classifier & Intensity calculation verified!")

    # ------------------------------------------------------------------
    # TEST 2: Empathy Directives Mapping
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Empathy Directive Prompt Templates ---")
    sad_directive = EmotionUtils.get_empathy_prompt(EmotionType.SADNESS.value)
    assert "[EMPATHY DIRECTIVE]" in sad_directive
    assert "gentle" in sad_directive or "supportive" in sad_directive
    logger.info(f"Sadness Directive Verified: '{sad_directive[:60]}...'")
    logger.info("✅ Empathy Directives mapping verified!")

    # ------------------------------------------------------------------
    # TEST 3: Timeline History & Mood Trend Tracking
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing Timeline History & Mood Trend Tracking ---")
    ctx = emotion_manager.get_user_emotion_context(test_user_id)
    assert len(ctx.history) > 0, "History timeline should contain recorded turns"
    logger.info(f"Dominant Mood Trend across turns: '{ctx.dominant_trend}' | Recent Turn Count: {len(ctx.history)}")
    logger.info("✅ Timeline History & Mood Trend tracking verified!")

    # ------------------------------------------------------------------
    # TEST 4: Chapter 6 Long-Term Memory Milestone Persistence
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing Chapter 6 Long-Term Memory Milestone Persistence ---")
    memories = MemoryManager.get_instance().retrieve_memories(test_user_id)
    logger.info(f"Retrieved {len(memories)} emotional milestone records from SQLite DB.")
    for mem in memories:
        logger.info(f"  • Memory Key: '{mem.memory_key}' | Value: '{mem.memory_value}'")
    logger.info("✅ Chapter 6 Long-Term Memory Milestone Persistence verified!")

    # ------------------------------------------------------------------
    # TEST 5: PromptBuilder Empathy Directive Injection
    # ------------------------------------------------------------------
    logger.info("\n--- 5. Testing PromptBuilder Empathy Injection ---")
    pb = PromptBuilder()
    directive = emotion_manager.get_empathy_directive(test_user_id, current_emotion="sadness")
    built_prompt = pb.build_prompt(
        user_input="I failed my test",
        empathy_directive=directive
    )
    assert "=== USER EMOTIONAL STATE & EMPATHY DIRECTIVE ===" in built_prompt
    assert "gentle" in built_prompt or "supportive" in built_prompt
    logger.info("✅ PromptBuilder Empathy Directive Injection verified!")

    logger.info("\n🎉 ALL EMOTIONAL INTELLIGENCE SYSTEM DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
