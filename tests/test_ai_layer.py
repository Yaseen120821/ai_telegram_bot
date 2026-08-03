"""
tests/test_ai_layer.py - Standalone Diagnostics & Test Script for AI Layer
=============================================================================

1. PURPOSE:
-----------
Verifies model loading, device auto-selection, prompt building, text generation, and response cleaning
for the `app/llm` package in isolation without starting a Telegram polling session.

2. HOW TO RUN:
--------------
`.venv\\Scripts\\python tests/test_ai_layer.py`
"""

import sys
import os
import time
import logging

# Ensure project root is in Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure basic console logging for diagnostic output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("sana_ai.tests.test_ai_layer")

from app.llm import ModelLoader, TextGenerator, PromptBuilder, ResponseFormatter


def run_ai_layer_test() -> None:
    """Executes verification tests for all components in app.llm package."""
    logger.info("=== Starting SANA AI Layer Diagnostic Tests ===")

    # 1. Test ModelLoader Singleton & Device Selection
    logger.info("\n--- 1. Testing ModelLoader ---")
    loader = ModelLoader.get_instance()
    logger.info(f"Target Device detected: {loader.device.upper()}")
    logger.info(f"Floating-point Precision: {loader.torch_dtype}")

    model, tokenizer = loader.load_model("models/qwen")
    assert model is not None, "Model failed to load!"
    assert tokenizer is not None, "Tokenizer failed to load!"
    logger.info("✅ ModelLoader verified successfully.")

    # 2. Test PromptBuilder ChatML Formatting
    logger.info("\n--- 2. Testing PromptBuilder ---")
    builder = PromptBuilder()
    formatted_prompt = builder.build_prompt("Explain recursion.", tokenizer=tokenizer)
    logger.info(f"Formatted ChatML Prompt snippet:\n{formatted_prompt[:150]}...")
    assert len(formatted_prompt) > 0, "Formatted prompt is empty!"
    logger.info("✅ PromptBuilder verified successfully.")

    # 3. Test TextGenerator & Batch Prompts
    logger.info("\n--- 3. Testing TextGenerator Inference ---")
    generator = TextGenerator(max_new_tokens=150, temperature=0.7)

    test_prompts = [
        "Hello",
        "Who are you?",
        "Explain recursion in one short paragraph.",
        "Write Java code for palindrome.",
        "Tell me a short joke.",
        "How are you?"
    ]

    for idx, prompt in enumerate(test_prompts, start=1):
        logger.info(f"\n[Test Prompt {idx}/{len(test_prompts)}] User: '{prompt}'")
        start_t = time.perf_counter()
        response = generator.generate_response(prompt)
        elapsed_t = time.perf_counter() - start_t

        logger.info(f"AI Response ({elapsed_t:.2f}s):\n{response}")
        assert len(response) > 0, f"Empty response for prompt '{prompt}'"

    logger.info("\n🎉 ALL AI LAYER DIAGNOSTIC TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    run_ai_layer_test()
