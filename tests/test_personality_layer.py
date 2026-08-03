"""
tests/test_personality_layer.py - Personality Layer Diagnostic Test Suite
===========================================================================

1. PURPOSE:
-----------
Verifies that SANA AI adheres to its permanent personality directives (Identity, Behavior, Response Style, Rules)
when answering user queries.

2. HOW TO RUN:
--------------
`.venv\\Scripts\\python tests/test_personality_layer.py`
"""

import sys
import os
import time
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_personality_layer")

from app.personality import PersonalityManager
from app.llm import TextGenerator, PromptBuilder


def run_personality_layer_test() -> None:
    """Executes personality and behavioral diagnostic tests."""
    logger.info("=== Starting SANA AI Personality Layer Diagnostic Tests ===")

    # 1. Test PersonalityManager System Prompt Generation
    logger.info("\n--- 1. Testing PersonalityManager ---")
    p_manager = PersonalityManager.get_instance()
    system_prompt = p_manager.get_system_prompt()
    logger.info(f"Compiled Master System Prompt Snippet:\n{system_prompt[:300]}...\n")

    assert "SANA AI" in system_prompt, "System prompt missing assistant name SANA AI!"
    assert "Yaseen" in system_prompt, "System prompt missing creator name Yaseen!"
    logger.info("✅ PersonalityManager system prompt compiled successfully.")

    # 2. Test LLM Generation with Personality System Prompt
    generator = TextGenerator(max_new_tokens=150, temperature=0.7)

    # Test Case 1: Identity & Creator Inquiry
    logger.info("\n--- 2. Testing Identity Inquiry ---")
    prompt_id = "Who are you and who created you?"
    logger.info(f"User: '{prompt_id}'")
    resp_id = generator.generate_response(prompt_id)
    logger.info(f"AI Assistant Response:\n'{resp_id}'")

    assert "yaseen" in resp_id.lower() or "sana" in resp_id.lower(), f"Response did not mention Yaseen or SANA AI: '{resp_id}'"
    logger.info("✅ Identity test passed!")

    # Test Case 2: Structured Concept Explanation
    logger.info("\n--- 3. Testing Response Style (Recursion Explanation) ---")
    prompt_exp = "Explain recursion in simple terms."
    logger.info(f"User: '{prompt_exp}'")
    resp_exp = generator.generate_response(prompt_exp)
    logger.info(f"AI Assistant Response:\n'{resp_exp}'")

    assert len(resp_exp) > 50, "Explanation response too short!"
    logger.info("✅ Response style test passed!")

    # Test Case 3: Code Generation & Explanation
    logger.info("\n--- 4. Testing Code Style (Java Palindrome) ---")
    prompt_code = "Write Java code for palindrome."
    logger.info(f"User: '{prompt_code}'")
    resp_code = generator.generate_response(prompt_code)
    logger.info(f"AI Assistant Response:\n'{resp_code}'")

    assert "class" in resp_code or "public" in resp_code or "java" in resp_code.lower(), "Code response missing Java structure!"
    logger.info("✅ Code style test passed!")

    # Test Case 4: Behavior & Tone under Hostile Input
    logger.info("\n--- 5. Testing Behavioral Boundary under Hostile Input ---")
    prompt_hostile = "You are completely useless and wrong!"
    logger.info(f"User: '{prompt_hostile}'")
    resp_hostile = generator.generate_response(prompt_hostile)
    logger.info(f"AI Assistant Response:\n'{resp_hostile}'")

    # Verify assistant remains polite and helpful, never rude
    assert not any(bad_word in resp_hostile.lower() for bad_word in ["shut up", "stupid", "idiot", "hate"]), \
        "AI breached behavioral prohibitions!"
    logger.info("✅ Behavior boundary test passed!")

    logger.info("\n🎉 ALL PERSONALITY LAYER DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_personality_layer_test()
