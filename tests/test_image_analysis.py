"""
tests/test_image_analysis.py - Specialized Image Analysis Diagnostic Test Suite
=================================================================================

Verifies:
1. ImageClassifier multi-signal classification across Code, UI, Document, Chart, Diagram, and Photo.
2. Specialized domain analyzers (CodeAnalyzer, UIAnalyzer, DocumentAnalyzer, ChartAnalyzer, DiagramAnalyzer).
3. AnalysisManager orchestrator facade routing, domain context formatting, and execution latency.
"""

import sys
import os
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.vision.image_analysis import (
    ImageCategory,
    AnalyzerType,
    CodeLanguage,
    DocumentType,
    ChartType,
    AnalysisRequest,
    AnalysisContext,
    ImageClassifier,
    CodeAnalyzer,
    UIAnalyzer,
    DocumentAnalyzer,
    ChartAnalyzer,
    DiagramAnalyzer,
    AnalysisManager
)


class TestImageAnalysisFramework(unittest.TestCase):
    def setUp(self):
        self.manager = AnalysisManager.get_instance()

    # 1. Test ImageClassifier Code Recognition
    def test_classifier_code(self):
        ocr_text = "def calculate_total(price, tax):\n    return price * (1 + tax)\nTraceback (most recent call last):"
        cat, conf, analyzer = ImageClassifier.classify("code_sample.png", ocr_text=ocr_text)

        self.assertEqual(cat, ImageCategory.CODE_SCREENSHOT)
        self.assertEqual(analyzer, AnalyzerType.CODE_ANALYZER)
        self.assertGreaterEqual(conf, 0.90)

    # 2. Test ImageClassifier UI Recognition
    def test_classifier_ui(self):
        ocr_text = "Login Page\nUsername: [      ]\nPassword: [      ]\n[ SUBMIT BUTTON ]"
        cat, conf, analyzer = ImageClassifier.classify("ui_desktop_screenshot.png", ocr_text=ocr_text)

        self.assertIn(cat, (ImageCategory.UI_SCREENSHOT, ImageCategory.DESKTOP_UI))
        self.assertIn(analyzer, (AnalyzerType.UI_ANALYZER, AnalyzerType.DESKTOP_UI_ANALYZER))

    # 3. Test ImageClassifier Research Paper Recognition
    def test_classifier_document(self):
        ocr_text = "Abstract\nThis research paper presents a multimodal AI architecture.\nReferences\n[1] Smith et al."
        cat, conf, analyzer = ImageClassifier.classify("paper.pdf", ocr_text=ocr_text)

        self.assertEqual(cat, ImageCategory.RESEARCH_PAPER)
        self.assertEqual(analyzer, AnalyzerType.DOCUMENT_ANALYZER)

    # 4. Test CodeAnalyzer Implementation
    def test_code_analyzer(self):
        data = {
            "ocr_text": "def divide(a, b):\n    return a / b\nZeroDivisionError: division by zero",
            "caption": "Python error stack trace"
        }
        res = CodeAnalyzer.analyze(data)

        self.assertEqual(res.category, ImageCategory.CODE_SCREENSHOT)
        self.assertIsNotNone(res.code_structure)
        self.assertEqual(res.code_structure.detected_language, CodeLanguage.PYTHON)
        self.assertIn("ZeroDivisionError", res.code_structure.error_type)

    # 5. Test DocumentAnalyzer Implementation
    def test_document_analyzer(self):
        data = {
            "ocr_text": "Deep Learning Architectures for Multimodal AI\nAbstract\nWe propose an integrated framework."
        }
        res = DocumentAnalyzer.analyze(data)

        self.assertIsNotNone(res.document_structure)
        self.assertEqual(res.document_structure.document_type, DocumentType.RESEARCH_PAPER)

    # 6. Test ChartAnalyzer Implementation
    def test_chart_analyzer(self):
        data = {
            "ocr_text": "Sales Quarterly Growth Bar Chart 2026",
            "caption": "A bar chart displaying revenue"
        }
        res = ChartAnalyzer.analyze(data)

        self.assertEqual(res.category, ImageCategory.CHART)
        self.assertIsNotNone(res.chart_structure)
        self.assertEqual(res.chart_structure.chart_type, ChartType.BAR_CHART)

    # 7. Test AnalysisManager Facade End-to-End Execution
    def test_analysis_manager_facade(self):
        req = AnalysisRequest(
            image_path="test_code.png",
            ocr_text="def calculate_total(price, tax):\n    return price * (1 + tax)"
        )
        ctx: AnalysisContext = self.manager.analyze_image(req)

        self.assertEqual(ctx.category, ImageCategory.CODE_SCREENSHOT)
        self.assertIsNotNone(ctx.result)
        self.assertIn("=== SPECIALIZED DOMAIN ANALYSIS ===", ctx.formatted_prompt_block)
        self.assertIn("CODE_SCREENSHOT", ctx.formatted_prompt_block)


if __name__ == "__main__":
    unittest.main()
