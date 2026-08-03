"""
tests/test_multimodal_audit.py - Enterprise Multimodal Pipeline Diagnostic Test Suite
========================================================================================
Validates end-to-end data flow for Vision AI, EasyOCR, Domain Analysis, PDF Analysis,
Context Assembly, Prompt Builder ChatML formatting, Tool Registry discovery, and IntelligentPipeline execution.
"""

import os
import unittest
from PIL import Image, ImageDraw

from app.vision import VisionManager, ImageType, VisionTask
from app.tools.registry import RegistryManager, RegistryState
from app.tools.integration.intelligent_pipeline import IntelligentPipeline
from app.tools.integration.context_builder import ContextBuilder
from app.tools.integration.integration_models import IntegratedContext, ToolContext
from app.llm.prompt_builder import PromptBuilder
from app.rag.pdf_analyzer import PDFAnalyzer


class TestMultimodalPipelineAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Creates sample test images (Code Screenshot, Diagram, Photo) and PDF in temp directory."""
        cls.test_dir = os.path.join("logs", "temp_media", "test_audit")
        os.makedirs(cls.test_dir, exist_ok=True)

        # 1. Create Code Screenshot Mock Image
        cls.code_img_path = os.path.join(cls.test_dir, "test_code_screenshot.png")
        img = Image.new("RGB", (600, 300), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n-1)", fill=(255, 255, 255))
        img.save(cls.code_img_path)

        # 2. Create Simple Photo Mock Image
        cls.photo_img_path = os.path.join(cls.test_dir, "test_photo.jpg")
        img2 = Image.new("RGB", (400, 400), color=(100, 150, 200))
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([50, 50, 200, 200], fill=(255, 100, 100))
        img2.save(cls.photo_img_path)

        # 3. Create Sample PDF File
        cls.pdf_path = os.path.join(cls.test_dir, "sample_doc.pdf")
        with open(cls.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 sample pdf document header and body text for testing PDFAnalyzer context building.")

    def test_01_tool_registry_auto_initialization(self):
        """Verify Tool Registry auto-initialization, single-instance state, and tool discovery."""
        registry = RegistryManager.get_instance()
        self.assertNotEqual(registry.state, RegistryState.UNINITIALIZED)
        tools = registry.list_tools()
        self.assertGreater(len(tools), 0, "Tool Registry should have discovered and registered candidate tools.")

        # Test duplicate initialize call is a safe no-op
        report = registry.initialize()
        self.assertIsNotNone(report)

    def test_02_vision_manager_multimodal_processing(self):
        """Verify VisionManager generates detailed VisionContext for input images."""
        vm = VisionManager.get_instance()
        v_ctx = vm.process_multimodal_images([self.code_img_path])
        self.assertIsNotNone(v_ctx)
        self.assertIn("ATTACHED IMAGE CONTEXT", v_ctx)
        self.assertTrue(any(k in v_ctx.lower() for k in ["code", "text", "calculate", "def", "screenshot"]))

    def test_03_context_builder_preserves_vision_context(self):
        """Verify ContextBuilder includes vision context in assembled PromptContext."""
        builder = ContextBuilder()
        int_ctx = IntegratedContext(
            raw_query="What is this code doing?",
            user_id="test_user",
            conversation_id="test_sess",
            vision_context="=== ATTACHED IMAGE CONTEXT ===\n[IMAGE #1] Category: CODE_SCREENSHOT\nOCR: def calculate_factorial(n):\n=== END IMAGE CONTEXT ==="
        )
        p_ctx = builder.assemble_prompt_context(integrated_context=int_ctx)
        self.assertIsNotNone(p_ctx.vision_context)
        self.assertIn("def calculate_factorial", p_ctx.vision_context)

    def test_04_prompt_builder_chatml_formatting(self):
        """Verify PromptBuilder formats ChatML correctly with system prompt & vision context."""
        pb = PromptBuilder()
        v_str = "=== ATTACHED IMAGE CONTEXT ===\nDetected python function code screenshot.\n=== END IMAGE CONTEXT ==="
        formatted_prompt = pb.build_prompt(
            user_input="Explain this function",
            image_context=v_str
        )
        self.assertIn("ATTACHED IMAGE CONTEXT", formatted_prompt)
        self.assertIn("Explain this function", formatted_prompt)

    def test_05_pdf_analyzer_context_extraction(self):
        """Verify PDFAnalyzer extracts PDF metadata and formats structured ChatML block."""
        analyzer = PDFAnalyzer.get_instance()
        res = analyzer.analyze_pdf(self.pdf_path)
        self.assertIsNotNone(res)
        self.assertIn("formatted_context", res)
        self.assertIn("ATTACHED PDF DOCUMENT", res["formatted_context"])

    def test_06_followup_conversation_memory(self):
        """Verify follow-up queries carry historical image context in conversation history."""
        pipeline = IntelligentPipeline.get_instance()

        history = [
            {"role": "user", "content": "What is in this uploaded picture?"},
            {"role": "assistant", "content": "The image shows a laptop screen with VS Code displaying project files: app, bot, chat, database, logs, models."}
        ]

        captured_prompt = []
        def mock_llm_callback(prompt: str) -> str:
            captured_prompt.append(prompt)
            return "The project files like app, bot, and chat are located on the left side in the File Explorer pane."

        res = pipeline.process_query(
            query="What is on the left side?",
            history=history,
            generator_callback=mock_llm_callback
        )

        self.assertIsNotNone(res.response_text)
        self.assertIn("left side", res.response_text.lower())
        self.assertTrue(len(captured_prompt) > 0)
        self.assertIn("laptop screen", captured_prompt[0])

    def test_07_intelligent_pipeline_end_to_end(self):
        """Verify IntelligentPipeline processes image query cleanly with mock generator callback."""
        pipeline = IntelligentPipeline.get_instance()

        captured_prompt = []
        def mock_llm_callback(prompt: str) -> str:
            captured_prompt.append(prompt)
            return "This code calculates the factorial of a given integer recursively."

        res = pipeline.process_query(
            query="What is this?",
            image_paths=[self.code_img_path],
            generator_callback=mock_llm_callback
        )

        self.assertIsNotNone(res.response_text)
        self.assertIn("factorial", res.response_text.lower())
        self.assertTrue(len(captured_prompt) > 0)
        self.assertIn("ATTACHED IMAGE CONTEXT", captured_prompt[0])


if __name__ == "__main__":
    unittest.main()
