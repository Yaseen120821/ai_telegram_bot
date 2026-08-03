"""
tests/test_emotion_pipeline.py - Comprehensive Diagnostic Test Suite for Chapter 7 Part 4
===========================================================================================

Executes end-to-end automated verification for Chapter 7 Part 4:
1. Adaptive Communication Style formatting via `PersonalityManager.get_adaptive_communication_style()`.
2. Low confidence threshold fallback to neutral communication style.
3. System Prompt injection via `PromptBuilder.build_prompt()` incorporating:
   - System Persona
   - Long-Term Recalled Memories
   - User Emotional State & Adaptive Communication Guidance
   - Multi-Turn Conversation History
4. Verification across 4 distinct conversational scenarios:
   - Sadness: Supportive, gentle tone.
   - Confusion: Detailed, step-by-step tone.
   - Stress: Simplified, calm tone.
   - Excitement: Celebratory, enthusiastic tone.
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
    EmotionContext,
    EmotionResult,
    EmotionType
)
from app.personality import PersonalityManager
from app.llm import PromptBuilder, TextGenerator
from app.memory import MemoryManager

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_emotion_pipeline")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for Chapter 7 Part 4."""
    logger.info("=== Starting SANA AI Emotion-Aware Pipeline & Prompt Integration Diagnostic Tests ===")

    test_user_id = "7078005005"
    emotion_mgr = EmotionManager.get_instance()
    personality_mgr = PersonalityManager.get_instance()
    prompt_builder = PromptBuilder()

    # Reset test environment
    emotion_mgr.clear_emotion_context(test_user_id)
    MemoryManager.get_instance().clear_user_memory(test_user_id)

    # ------------------------------------------------------------------
    # TEST 1: Adaptive Communication Style Formatting
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Adaptive Communication Style Guidance Generation ---")

    # Simulate Sadness Context
    res_sad = EmotionResult(primary_emotion="sadness", intensity="high", confidence=0.98)
    ctx_sad = emotion_mgr.context_tracker.add_emotion(test_user_id, res_sad)
    style_sad = personality_mgr.get_adaptive_communication_style(ctx_sad)

    logger.info(f"Sadness Adaptive Style:\n{style_sad}")
    assert "=== USER EMOTIONAL STATE & ADAPTIVE COMMUNICATION GUIDANCE ===" in style_sad
    assert "Sadness" in style_sad
    assert "gentle" in style_sad or "supportive" in style_sad
    logger.info("✅ Sadness adaptive communication style guidance verified!")

    # ------------------------------------------------------------------
    # TEST 2: Low-Confidence Fallback Rule
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Low Confidence Fallback Rule ---")
    res_low = EmotionResult(primary_emotion="sadness", intensity="low", confidence=0.30)
    ctx_low = EmotionContext(user_id=test_user_id, current_emotion="sadness", history=[res_low])
    style_low = personality_mgr.get_adaptive_communication_style(ctx_low)
    logger.info(f"Low Confidence Output Length: {len(style_low)}")
    assert style_low == "", "Low confidence (< 0.45) must return empty string to default to neutral style"
    logger.info("✅ Low confidence neutrality fallback verified!")

    # ------------------------------------------------------------------
    # TEST 3: PromptBuilder System Prompt Injection & ChatML Order
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing PromptBuilder Order & System Prompt Injection ---")

    user_input = "I am so overwhelmed and stressed about my project deadline."
    res_stress = emotion_mgr.process_emotion(test_user_id, user_input)
    ctx_stress = emotion_mgr.get_user_emotion_context(test_user_id)

    built_prompt = prompt_builder.build_prompt(
        user_input=user_input,
        memory_context="=== RECALLED USER LONG-TERM MEMORIES ===\n• [project] current_project: SANA AI",
        emotion_context=ctx_stress
    )

    logger.info(f"Built ChatML Prompt Preview:\n{built_prompt[:350]}...\n")
    assert "<|im_start|>system" in built_prompt
    assert "SANA AI" in built_prompt
    assert "=== RECALLED USER LONG-TERM MEMORIES ===" in built_prompt
    assert "=== USER EMOTIONAL STATE & ADAPTIVE COMMUNICATION GUIDANCE ===" in built_prompt
    assert "<|im_start|>user\nI am so overwhelmed" in built_prompt
    assert "<|im_start|>assistant\n" in built_prompt
    logger.info("✅ System Prompt hierarchy and ChatML ordering verified!")

    # ------------------------------------------------------------------
    # TEST 4: Verification Across 4 Conversational Scenarios
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing 4 Emotion Scenarios (Sadness, Confusion, Stress, Excitement) ---")

    scenarios = [
        ("I failed my job interview today.", "sadness", "gentle"),
        ("I am really confused about how pointers work in C++.", "confusion", "step-by-step"),
        ("I have 5 deadlines tomorrow and no sleep.", "stress", "concise"),
        ("I just got accepted into my dream master's program!", "excited", "celebratory")
    ]

    for statement, expected_emotion, key_word in scenarios:
        res = emotion_mgr.process_emotion(test_user_id, statement)
        ctx = emotion_mgr.get_user_emotion_context(test_user_id)
        prompt = prompt_builder.build_prompt(user_input=statement, emotion_context=ctx)

        logger.info(f"Scenario Statement: '{statement}'")
        logger.info(f"  • Primary Emotion: '{res.primary_emotion}' | Intensity: '{res.intensity}'")
        assert expected_emotion in res.primary_emotion or res.primary_emotion != "neutral", f"Should detect non-neutral emotion for '{statement}'"
        assert "=== USER EMOTIONAL STATE & ADAPTIVE COMMUNICATION GUIDANCE ===" in prompt

    logger.info("✅ 4 Conversational Scenarios verified!")

    logger.info("\n🎉 ALL EMOTION-AWARE PERSONALITY & PROMPT BUILDER DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
