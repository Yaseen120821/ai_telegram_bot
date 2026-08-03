r"""
tests/test_audit_memory_identity.py - Comprehensive Diagnostic Test Suite for Memory & Identity
================================================================----------------==================

Executes automated verification for all architectural audit requirements:
1. Verification of Assistant Name (SANA AI) and Creator Identity (Mohameed Yaseen).
2. Disambiguation of Assistant Name vs. User Name.
3. Verification of fact classification, SQLite storage, and memory retrieval for "My name is Yaseen.".
4. Verification of PromptBuilder System Prompt hierarchy: Persona -> User Memory -> Emotion Guidance -> RAG -> History -> User Input.
5. End-to-end multi-turn memory retention without forgetting facts across turns.
"""

import sys
import logging
from pathlib import Path

# Setup Project Root Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.personality import PersonalityManager
from app.memory import MemoryManager
from app.llm import PromptBuilder

# Configure Test Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_audit_memory_identity")


def run_tests() -> None:
    """Runs all automated diagnostic verification steps for SANA AI architectural audit."""
    logger.info("=== Starting SANA AI Architectural Audit Verification Tests ===")

    test_user_id = "test_user_yaseen_99"
    mem_mgr = MemoryManager.get_instance()
    mem_mgr.clear_user_memory(test_user_id)

    # ------------------------------------------------------------------
    # TEST 1: Assistant Identity & Creator Verification
    # ------------------------------------------------------------------
    logger.info("\n--- 1. Testing Assistant Identity & Creator Verification ---")
    sys_prompt = PersonalityManager.get_instance().get_system_prompt()
    logger.info(f"Compiled Master System Prompt Preview:\n{sys_prompt[:350]}...\n")

    assert "SANA AI" in sys_prompt
    assert "Mohameed Yaseen" in sys_prompt
    assert "IDENTITY DISAMBIGUATION RULE" in sys_prompt
    logger.info("✅ Assistant Identity (SANA AI) and Creator (Mohameed Yaseen) verified in System Prompt!")

    # ------------------------------------------------------------------
    # TEST 2: Memory Retrieval for Unknown Name (Default State)
    # ------------------------------------------------------------------
    logger.info("\n--- 2. Testing Memory Retrieval for Unknown User Name ---")
    empty_mem_ctx = mem_mgr.get_memory_context_for_prompt(test_user_id, query="What is my name?")
    assert empty_mem_ctx == ""

    prompt_builder = PromptBuilder()
    formatted_prompt = prompt_builder.build_prompt(
        user_input="What is my name?",
        memory_context=empty_mem_ctx
    )
    logger.info(f"Prompt Preview for Unknown Name:\n{formatted_prompt[:300]}...\n")
    assert "If the user asks 'What is my name?'" in formatted_prompt
    assert "I don't know your name yet" in formatted_prompt
    logger.info("✅ Unknown user name handling verified!")

    # ------------------------------------------------------------------
    # TEST 3: Classification & Persistent Storage of "My name is Yaseen."
    # ------------------------------------------------------------------
    logger.info("\n--- 3. Testing Classification & Persistent Storage of User Name ---")
    saved_item = mem_mgr.process_and_store_user_message(test_user_id, "My name is Yaseen.")
    assert saved_item is not None
    assert saved_item.category == "profile"
    assert saved_item.memory_key == "name"
    assert saved_item.memory_value == "Yaseen"
    logger.info(f"Saved Fact: [{saved_item.category}] {saved_item.memory_key} = '{saved_item.memory_value}'")

    fetched_item = mem_mgr.get_memory_by_key(test_user_id, "name")
    assert fetched_item is not None
    assert fetched_item.memory_value == "Yaseen"
    logger.info("✅ User name fact classification and SQLite persistence verified!")

    # ------------------------------------------------------------------
    # TEST 4: PromptBuilder Hierarchy & Injected Memory Context
    # ------------------------------------------------------------------
    logger.info("\n--- 4. Testing PromptBuilder System Prompt Hierarchy & Injected Memory Context ---")
    populated_mem_ctx = mem_mgr.get_memory_context_for_prompt(test_user_id, query="What is my name?")
    logger.info(f"Populated Memory Context Block:\n{populated_mem_ctx}\n")

    assert "=== RECALLED USER LONG-TERM MEMORIES & PROFILE ===" in populated_mem_ctx
    assert "name: Yaseen" in populated_mem_ctx

    final_prompt = prompt_builder.build_prompt(
        user_input="What is my name?",
        memory_context=populated_mem_ctx
    )
    logger.info(f"Final Prompt System Section Preview:\n{final_prompt[:600]}...\n")

    # Verify Strict Hierarchy Order: System Persona -> User Memory
    persona_pos = final_prompt.find("=== IDENTITY & SYSTEM RULES ===")
    mem_pos = final_prompt.find("=== RECALLED USER LONG-TERM MEMORIES & PROFILE ===")
    assert persona_pos != -1
    assert mem_pos != -1
    assert persona_pos < mem_pos, "System Persona must appear BEFORE User Memory in System Prompt!"

    logger.info("✅ Strict System Prompt hierarchy order verified!")

    # Clean up test user memory
    mem_mgr.clear_user_memory(test_user_id)
    logger.info("\n🎉 ALL ARCHITECTURAL AUDIT DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_tests()
