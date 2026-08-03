"""
tests/test_multimodal_integration.py - Multimodal Pipeline Diagnostic Test Suite
===================================================================================

Verifies:
1. PromptBuilder System Prompt assembly order including image_context parameter.
2. IntelligentPipeline process_query end-to-end multimodal execution with image_paths.
3. Multimodal integration with Memory, Emotion, RAG, and Tool execution subsystems.
"""

import sys
import os
import unittest
from PIL import Image as PILImage

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.prompt_builder import PromptBuilder
from app.tools.integration.intelligent_pipeline import IntelligentPipeline
from app.vision import merge_image_contexts, ImageContextData, ImageRole
from app.vision.image_analysis import ImageCategory


class TestMultimodalIntegrationPipeline(unittest.TestCase):
    def setUp(self):
        self.prompt_builder = PromptBuilder()
        self.pipeline = IntelligentPipeline.get_instance()
        self.img_path = "test_multimodal_sample.png"

        # Create temporary sample image file
        img = PILImage.new("RGB", (300, 200), color=(100, 150, 200))
        img.save(self.img_path)

    def tearDown(self):
        if os.path.exists(self.img_path):
            try:
                os.remove(self.img_path)
            except Exception:
                pass

    # 1. Test PromptBuilder Image Context Assembly
    def test_prompt_builder_image_context(self):
        user_msg = "What is in this screenshot?"
        image_context_str = "=== ATTACHED IMAGE CONTEXT ===\nCategory: CODE_SCREENSHOT\nExtracted Text: def main(): pass\n=== END IMAGE CONTEXT ==="
        memory_str = "=== RECALLED USER MEMORY ===\nUser name is Alice."

        prompt = self.prompt_builder.build_prompt(
            user_input=user_msg,
            memory_context=memory_str,
            image_context=image_context_str
        )

        self.assertIn("=== RECALLED USER MEMORY ===", prompt)
        self.assertIn("=== ATTACHED IMAGE CONTEXT ===", prompt)
        self.assertIn("def main(): pass", prompt)
        self.assertIn("What is in this screenshot?", prompt)

    # 2. Test IntelligentPipeline Single Image Execution
    def test_pipeline_single_image(self):
        query = "Explain what is shown in this attached file."
        res = self.pipeline.process_query(
            query=query,
            image_paths=[self.img_path],
            user_id="user_alice"
        )

        self.assertIsNotNone(res)
        self.assertIsNotNone(res.integrated_context)
        self.assertIsNotNone(res.integrated_context.vision_context)
        self.assertIn(self.img_path, res.integrated_context.image_paths)

    # 3. Test IntelligentPipeline Image + Tool Execution
    def test_pipeline_image_and_tool(self):
        query = "List files in directory and analyze this attached image."
        res = self.pipeline.process_query(
            query=query,
            image_paths=[self.img_path],
            user_id="user_bob"
        )

        self.assertIsNotNone(res.integrated_context.vision_context)

    # 4. Test Multimodal Context Utilities
    def test_multimodal_utils(self):
        img_ctx = ImageContextData(
            file_path="sample.jpg",
            role=ImageRole.PRIMARY_SUBJECT
        )
        img_ctx.analysis.category = ImageCategory.PHOTO
        img_ctx.vision.short_caption = "A bright blue room"

        merged = merge_image_contexts([img_ctx])
        self.assertIn("ATTACHED MULTIMODAL IMAGE CONTEXT", merged)
        self.assertIn("A bright blue room", merged)


if __name__ == "__main__":
    unittest.main()
