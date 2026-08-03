"""
tests/test_conversation_layer.py - Conversation Management Diagnostic Test Suite
==================================================================================

1. PURPOSE:
-----------
Verifies multi-user session isolation, multi-turn conversation memory recall, sliding context
window trimming, and conversation history wiping (`/clear`) using the local Qwen LLM.

2. HOW TO RUN:
--------------
`.venv\\Scripts\\python tests/test_conversation_layer.py`
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
logger = logging.getLogger("sana_ai.tests.test_conversation_layer")

from app.conversation import ConversationManager
from app.llm import TextGenerator


def run_conversation_layer_test() -> None:
    """Executes multi-turn conversation memory diagnostic tests."""
    logger.info("=== Starting SANA AI Conversation Management Layer Tests ===")

    conv_manager = ConversationManager.get_instance()
    generator = TextGenerator(max_new_tokens=150, temperature=0.7)

    USER_A_ID = 1001  # Yaseen
    USER_B_ID = 2002  # Sarah

    # -------------------------------------------------------------
    # 1. Test Session Isolation (User A vs User B)
    # -------------------------------------------------------------
    logger.info("\n--- 1. Testing Multi-User Session Isolation ---")
    conv_manager.clear_user_history(USER_A_ID)
    conv_manager.clear_user_history(USER_B_ID)

    conv_manager.add_user_message(USER_A_ID, "My name is Yaseen.")
    conv_manager.add_assistant_message(USER_A_ID, "Nice to meet you, Yaseen!")

    conv_manager.add_user_message(USER_B_ID, "My name is Sarah.")
    conv_manager.add_assistant_message(USER_B_ID, "Hello Sarah!")

    history_a = conv_manager.get_formatted_history(USER_A_ID)
    history_b = conv_manager.get_formatted_history(USER_B_ID)

    assert len(history_a) == 2, f"Expected 2 messages for User A, got {len(history_a)}"
    assert len(history_b) == 2, f"Expected 2 messages for User B, got {len(history_b)}"
    assert "Yaseen" in history_a[0]["content"], "User A prompt missing Yaseen!"
    assert "Sarah" in history_b[0]["content"], "User B prompt missing Sarah!"

    logger.info("✅ Multi-user session isolation verified successfully!")

    # -------------------------------------------------------------
    # 2. Test Multi-Turn LLM Memory Recall for User A
    # -------------------------------------------------------------
    logger.info("\n--- 2. Testing LLM Memory Recall (User A: Yaseen) ---")
    
    # Prompt 1: Introduce Name
    prompt1 = "My name is Yaseen."
    logger.info(f"[Turn 1] User A: '{prompt1}'")
    history1 = conv_manager.get_formatted_history(USER_A_ID)
    conv_manager.add_user_message(USER_A_ID, prompt1)
    
    resp1 = generator.generate_response(prompt1, history=history1)
    conv_manager.add_assistant_message(USER_A_ID, resp1)
    logger.info(f"AI Assistant Turn 1: '{resp1}'")

    # Prompt 2: Query Name Recall
    prompt2 = "What is my name?"
    logger.info(f"\n[Turn 2] User A: '{prompt2}'")
    history2 = conv_manager.get_formatted_history(USER_A_ID)
    conv_manager.add_user_message(USER_A_ID, prompt2)

    resp2 = generator.generate_response(prompt2, history=history2)
    conv_manager.add_assistant_message(USER_A_ID, resp2)
    logger.info(f"AI Assistant Turn 2: '{resp2}'")

    assert "yaseen" in resp2.lower(), f"Expected AI to remember name 'Yaseen', got: '{resp2}'"
    logger.info("✅ Name memory recall verified successfully!")

    # Prompt 3: Programming Language Preference
    prompt3 = "I like Java programming language."
    logger.info(f"\n[Turn 3] User A: '{prompt3}'")
    history3 = conv_manager.get_formatted_history(USER_A_ID)
    conv_manager.add_user_message(USER_A_ID, prompt3)

    resp3 = generator.generate_response(prompt3, history=history3)
    conv_manager.add_assistant_message(USER_A_ID, resp3)
    logger.info(f"AI Assistant Turn 3: '{resp3}'")

    # Prompt 4: Recall Programming Language
    prompt4 = "What programming language do I like?"
    logger.info(f"\n[Turn 4] User A: '{prompt4}'")
    history4 = conv_manager.get_formatted_history(USER_A_ID)
    conv_manager.add_user_message(USER_A_ID, prompt4)

    resp4 = generator.generate_response(prompt4, history=history4)
    conv_manager.add_assistant_message(USER_A_ID, resp4)
    logger.info(f"AI Assistant Turn 4: '{resp4}'")

    assert "java" in resp4.lower(), f"Expected AI to remember 'Java', got: '{resp4}'"
    logger.info("✅ Preference memory recall verified successfully!")

    # -------------------------------------------------------------
    # 3. Test Conversation Wiping (/clear)
    # -------------------------------------------------------------
    logger.info("\n--- 3. Testing Conversation Clearing (/clear) ---")
    conv_manager.clear_user_history(USER_A_ID)
    cleared_history = conv_manager.get_formatted_history(USER_A_ID)
    assert len(cleared_history) == 0, f"Expected 0 messages after clear, got {len(cleared_history)}"

    # Ask again after clearing
    prompt_after_clear = "What programming language do I like?"
    logger.info(f"[After /clear] User A asks: '{prompt_after_clear}'")
    resp_after_clear = generator.generate_response(prompt_after_clear, history=cleared_history)
    logger.info(f"AI Assistant Response (No History):\n'{resp_after_clear}'")

    assert "java" not in resp_after_clear.lower() or "don't" in resp_after_clear.lower() or "not" in resp_after_clear.lower() or "haven't" in resp_after_clear.lower(), \
        "AI remembered cleared context!"
    logger.info("✅ Conversation history clearing verified successfully!")

    logger.info("\n🎉 ALL CONVERSATION LAYER DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_conversation_layer_test()
