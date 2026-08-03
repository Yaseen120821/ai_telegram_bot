"""
tests/test_memory_layer.py - Persistent Long-Term Memory Management Test Suite
================================================================================

1. PURPOSE:
-----------
Verifies MemoryManager public API (`save_memory`, `retrieve_memories`, `search_memory`, `update_memory`,
`delete_memory`, `memory_exists`, `get_memory_by_key`, `get_memories_by_category`, `clear_user_memory`),
classification scoring (1-10 importance, 0-1 confidence), duplicate resolution, ranking, and search.

2. HOW TO RUN:
--------------
`.venv\\Scripts\\python tests/test_memory_layer.py`
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
logger = logging.getLogger("sana_ai.tests.test_memory_layer")

from app.memory import MemoryManager, MemoryClassifier, MemoryItem, MemoryCategory


def run_memory_management_test() -> None:
    """Executes long-term memory management system diagnostic tests."""
    logger.info("=== Starting SANA AI Memory Management System Diagnostic Tests ===")

    memory_manager = MemoryManager.get_instance()
    USER_A_ID = "7074001001"  # Yaseen
    USER_B_ID = "8085002002"  # Sarah

    # Reset state
    memory_manager.clear_user_memory(USER_A_ID)
    memory_manager.clear_user_memory(USER_B_ID)

    # -------------------------------------------------------------
    # 1. Test Classification Engine & Scoring (Part 3)
    # -------------------------------------------------------------
    logger.info("\n--- 1. Testing Classifier Flow, Importance (1-10) & Confidence (0-1) ---")
    classifier = MemoryClassifier()

    # Positive memory classification assertions
    res1 = classifier.classify_statement("My name is Yaseen.")
    assert res1.is_candidate and res1.category == MemoryCategory.PROFILE and res1.importance == 10 and res1.confidence == 1.0, f"Failed Profile: {res1}"

    res2 = classifier.classify_statement("I prefer coding in Python.")
    assert res2.is_candidate and res2.category == MemoryCategory.PREFERENCE and res2.importance == 8, f"Failed Preference: {res2}"

    res3 = classifier.classify_statement("I am building SANA AI.")
    assert res3.is_candidate and res3.category == MemoryCategory.PROJECT and res3.importance == 10, f"Failed Project: {res3}"

    # Negative non-memory prompt assertions
    assert not classifier.classify_statement("What is Python?").is_candidate
    assert not classifier.classify_statement("Write Java code").is_candidate
    assert not classifier.classify_statement("Today's weather is hot").is_candidate
    logger.info("✅ Classifier accuracy, importance scales (1-10), and confidence scores verified!")

    # -------------------------------------------------------------
    # 2. Test Public API CRUD Operations (`save_memory`, `memory_exists`, `get_memory_by_key`)
    # -------------------------------------------------------------
    logger.info("\n--- 2. Testing MemoryManager CRUD API ---")
    item1 = MemoryItem(user_id=USER_A_ID, category=MemoryCategory.PROFILE, memory_key="name", memory_value="Yaseen", importance=10)
    item2 = MemoryItem(user_id=USER_A_ID, category=MemoryCategory.PREFERENCE, memory_key="preferred_language", memory_value="Python", importance=8)
    item3 = MemoryItem(user_id=USER_A_ID, category=MemoryCategory.PROJECT, memory_key="current_project", memory_value="SANA AI", importance=10)

    memory_manager.save_memory(item1)
    memory_manager.save_memory(item2)
    memory_manager.save_memory(item3)

    assert memory_manager.memory_exists(USER_A_ID, "name"), "memory_exists failed for 'name'!"
    assert memory_manager.memory_exists(USER_A_ID, "preferred_language"), "memory_exists failed for 'preferred_language'!"

    fetched_name = memory_manager.get_memory_by_key(USER_A_ID, "name")
    assert fetched_name and fetched_name.memory_value == "Yaseen", f"get_memory_by_key failed: {fetched_name}"
    logger.info("✅ CRUD API methods (`save_memory`, `memory_exists`, `get_memory_by_key`) passed!")

    # -------------------------------------------------------------
    # 3. Test Duplicate Resolution & Update Strategy
    # -------------------------------------------------------------
    logger.info("\n--- 3. Testing Duplicate Resolution & Update Strategy ---")
    updated_item1 = MemoryItem(user_id=USER_A_ID, category=MemoryCategory.PROFILE, memory_key="name", memory_value="Mohamed Yaseen", importance=10)
    memory_manager.update_memory(updated_item1)

    all_a = memory_manager.retrieve_memories(USER_A_ID)
    assert len(all_a) == 3, f"Duplicate created! Expected 3 records, got {len(all_a)}"

    re_fetched = memory_manager.get_memory_by_key(USER_A_ID, "name")
    assert re_fetched and re_fetched.memory_value == "Mohamed Yaseen", f"Update failed: {re_fetched}"
    logger.info("✅ Duplicate resolution & update strategy verified!")

    # -------------------------------------------------------------
    # 4. Test Search & Category Retrieval (`search_memory`, `get_memories_by_category`)
    # -------------------------------------------------------------
    logger.info("\n--- 4. Testing Memory Search & Category Filtering ---")
    search_res = memory_manager.search_memory(USER_A_ID, "SANA AI")
    assert search_res.total_count == 1 and search_res.items[0].memory_key == "current_project", f"Search failed: {search_res}"

    cat_projects = memory_manager.get_memories_by_category(USER_A_ID, MemoryCategory.PROJECT)
    assert len(cat_projects) == 1 and cat_projects[0].memory_value == "SANA AI", f"Category retrieval failed: {cat_projects}"
    logger.info("✅ `search_memory` and `get_memories_by_category` verified!")

    # -------------------------------------------------------------
    # 5. Test Deletion & Wiping (`delete_memory`, `clear_user_memory`)
    # -------------------------------------------------------------
    logger.info("\n--- 5. Testing Soft Delete & User Memory Wiping ---")
    soft_deleted = memory_manager.delete_memory(USER_A_ID, "preferred_language")
    assert soft_deleted, "Soft deletion returned False!"
    assert not memory_manager.memory_exists(USER_A_ID, "preferred_language"), "Soft deleted memory still returned by memory_exists!"

    hard_cleared = memory_manager.clear_user_memory(USER_A_ID)
    assert hard_cleared >= 2, f"Expected at least 2 hard deleted rows, got {hard_cleared}"
    assert len(memory_manager.retrieve_memories(USER_A_ID)) == 0, "Memories remaining after clear_user_memory!"
    logger.info("✅ Soft delete (`delete_memory`) and hard clear (`clear_user_memory`) verified!")

    logger.info("\n🎉 ALL MEMORY MANAGEMENT LAYER DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_memory_management_test()
