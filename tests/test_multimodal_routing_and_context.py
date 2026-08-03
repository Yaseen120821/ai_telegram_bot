"""
tests/test_multimodal_routing_and_context.py - Multimodal Routing & Context Injection Validation Suite
=======================================================================================================
Validates all 9 acceptance test scenarios for SANA AI Multimodal Repair:
1. Plain Text Message
2. Image Upload
3. Code Screenshot
4. Architecture Diagram
5. Digital PDF Document
6. Scanned PDF Document
7. Mixed Text + Image Input
8. Image Follow-Up Question
9. PDF Follow-Up Question
"""

import os
import tempfile
import unittest
from PIL import Image, ImageDraw, ImageFont

from app.tools.routing.router_types import RoutingMode, DecisionType
from app.tools.routing.decision_engine import DecisionEngine
from app.tools.integration.intelligent_pipeline import IntelligentPipeline
from app.vision.vision_manager import VisionManager
from app.rag.pdf_analyzer import PDFAnalyzer


class TestMultimodalRoutingAndContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="sana_test_multimodal_")
        
        # 1. Create Synthetic Test Image (Photo)
        cls.photo_path = os.path.join(cls.temp_dir, "test_photo.jpg")
        img1 = Image.new("RGB", (400, 300), color=(73, 109, 137))
        d1 = ImageDraw.Draw(img1)
        d1.text((10, 10), "Test Photo Content", fill=(255, 255, 0))
        img1.save(cls.photo_path)

        # 2. Create Synthetic Code Screenshot Image
        cls.code_img_path = os.path.join(cls.temp_dir, "code_screenshot.png")
        img2 = Image.new("RGB", (600, 400), color=(30, 30, 30))
        d2 = ImageDraw.Draw(img2)
        d2.text((20, 20), "def process_data(x):\n    return x * 2", fill=(0, 255, 0))
        img2.save(cls.code_img_path)

        # 3. Create Synthetic Architecture Diagram Image
        cls.diagram_path = os.path.join(cls.temp_dir, "architecture_diagram.png")
        img3 = Image.new("RGB", (500, 500), color=(240, 240, 240))
        d3 = ImageDraw.Draw(img3)
        d3.rectangle([50, 50, 200, 150], outline="black", fill="white")
        d3.text((60, 80), "Client App", fill="black")
        d3.rectangle([300, 50, 450, 150], outline="black", fill="white")
        d3.text((310, 80), "SANA AI API", fill="black")
        img3.save(cls.diagram_path)

        # 4. Create Mock PDF Path
        cls.pdf_path = os.path.join(cls.temp_dir, "sample_document.pdf")
        with open(cls.pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 header sample pdf text content for unit testing %EOF")

        cls.decision_engine = DecisionEngine()
        cls.pipeline = IntelligentPipeline.get_instance()

    def test_01_text_message_routing(self):
        """Scenario 1: Plain text message should route via DIRECT_RESPONSE or TOOL (not MULTIMODAL)."""
        decision = self.decision_engine.evaluate_request("Hello, who are you?")
        self.assertNotEqual(decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIn(decision.routing_mode, (RoutingMode.DIRECT_RESPONSE, RoutingMode.TOOL))

    def test_02_image_upload_routing_and_context(self):
        """Scenario 2: Image upload automatically triggers VisionManager and selects MULTIMODAL routing."""
        res = self.pipeline.process_query(
            query="What is in this image?",
            image_paths=[self.photo_path],
            user_id="test_user_02"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertEqual(res.integrated_context.decision.decision_type, DecisionType.MULTIMODAL_RESPONSE)
        self.assertIsNotNone(res.integrated_context.vision_context)
        self.assertIn("=== ATTACHED IMAGE CONTEXT", res.integrated_context.vision_context)

    def test_03_code_screenshot_processing(self):
        """Scenario 3: Code screenshot triggers OCR & Vision analysis and MULTIMODAL routing."""
        res = self.pipeline.process_query(
            query="Explain this python code screenshot.",
            image_paths=[self.code_img_path],
            user_id="test_user_03"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIsNotNone(res.integrated_context.vision_context)
        self.assertIn("code_screenshot.png", res.integrated_context.vision_context)

    def test_04_architecture_diagram_processing(self):
        """Scenario 4: Architecture diagram image generates structured context block."""
        res = self.pipeline.process_query(
            query="Analyze this system architecture diagram.",
            image_paths=[self.diagram_path],
            user_id="test_user_04"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIn("architecture_diagram.png", res.integrated_context.vision_context)

    def test_05_digital_pdf_document_processing(self):
        """Scenario 5: Digital PDF document triggers PDFAnalyzer and MULTIMODAL routing."""
        pdf_res = PDFAnalyzer.get_instance().analyze_pdf(self.pdf_path)
        pdf_ctx = pdf_res["formatted_context"]

        res = self.pipeline.process_query(
            query="Summarize the attached PDF file.",
            document_context=pdf_ctx,
            user_id="test_user_05"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIsNotNone(res.integrated_context.vision_context)
        self.assertIn("=== ATTACHED PDF DOCUMENT", res.integrated_context.vision_context)

    def test_06_scanned_pdf_document_ocr(self):
        """Scenario 6: Scanned PDF format produces formatted document context block."""
        scanned_ctx = "=== ATTACHED PDF DOCUMENT: scanned_invoice.pdf ===\n• Extracted Text (OCR):\nInvoice #1004 Total: $250.00\n=== END ATTACHED PDF DOCUMENT ==="
        res = self.pipeline.process_query(
            query="What is the invoice total in this scanned PDF?",
            document_context=scanned_ctx,
            user_id="test_user_06"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIn("Invoice #1004", res.integrated_context.vision_context)

    def test_07_mixed_text_and_image(self):
        """Scenario 7: Mixed text prompt + image upload integrates both prompt and vision context."""
        query_text = "Compare this diagram with standard microservices patterns."
        res = self.pipeline.process_query(
            query=query_text,
            image_paths=[self.diagram_path],
            user_id="test_user_07"
        )
        self.assertEqual(res.integrated_context.raw_query, query_text)
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIsNotNone(res.integrated_context.vision_context)

    def test_08_image_followup_question(self):
        """Scenario 8: Follow-up question preserves past history and vision context propagation."""
        history = [
            {"role": "user", "content": "Analyze this screenshot."},
            {"role": "assistant", "content": "The screenshot shows a Python function `process_data`."}
        ]
        res = self.pipeline.process_query(
            query="Can you optimize that function to be async?",
            image_paths=[self.code_img_path],
            history=history,
            user_id="test_user_08"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIsNotNone(res.integrated_context.vision_context)

    def test_09_pdf_followup_question(self):
        """Scenario 9: Follow-up question regarding a PDF document preserves document context."""
        pdf_ctx = "=== ATTACHED PDF DOCUMENT: report.pdf ===\nRevenue: $1.2M\n=== END ATTACHED PDF DOCUMENT ==="
        history = [
            {"role": "user", "content": "What is the revenue in the PDF?"},
            {"role": "assistant", "content": "The revenue stated in report.pdf is $1.2M."}
        ]
        res = self.pipeline.process_query(
            query="What percentage increase is that compared to last year?",
            document_context=pdf_ctx,
            history=history,
            user_id="test_user_09"
        )
        self.assertEqual(res.integrated_context.decision.routing_mode, RoutingMode.MULTIMODAL)
        self.assertIn("report.pdf", res.integrated_context.vision_context)


if __name__ == "__main__":
    unittest.main()
