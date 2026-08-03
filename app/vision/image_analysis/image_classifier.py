"""
app/vision/image_analysis/image_classifier.py - Multi-Signal Image Classifier
================================================================================
Classifies visual content into specific domain categories (Code, IDE, UI, Document, Chart,
Diagram, Handwriting, Photo) using OCR text keyword heuristics, metadata signals, and visual captions.
"""

import logging
from typing import Tuple, Dict, Any, List
from app.vision.image_analysis.analysis_types import ImageCategory, AnalyzerType
from app.vision.image_analysis.analysis_utils import match_keywords

logger = logging.getLogger("sana_ai.vision.analysis.classifier")


class ImageClassifier:
    """Classifies visual content across 18 domain categories to route to specialized analyzers."""

    @staticmethod
    def classify(
        file_path: str,
        caption: str = "",
        ocr_text: str = "",
        objects: List[Any] = None
    ) -> Tuple[ImageCategory, float, AnalyzerType]:
        """
        Evaluates file path, caption, OCR text, and object detections to return
        Tuple[ImageCategory, confidence: float, AnalyzerType].
        """
        ocr_lower = ocr_text.lower()
        cap_lower = caption.lower()
        fname_lower = file_path.lower()

        # 1. Code / Terminal / IDE Screenshots
        code_kw = ["def ", "class ", "import ", "function", "var ", "const ", "return ", "exception", "traceback", "syntaxerror", "line "]
        term_kw = ["bash", "zsh", "ps ", "cmd", "exit code", "error:", "fatal:", "failed:"]

        if any(kw in ocr_lower for kw in code_kw) or "code" in fname_lower:
            if any(kw in ocr_lower for kw in term_kw) or "terminal" in fname_lower:
                return ImageCategory.TERMINAL_SCREENSHOT, 0.95, AnalyzerType.CODE_ANALYZER
            return ImageCategory.CODE_SCREENSHOT, 0.95, AnalyzerType.CODE_ANALYZER

        # 2. UI Screenshots (Mobile / Desktop / Website)
        ui_kw = ["menu", "button", "login", "settings", "submit", "dashboard", "navbar", "sidebar"]
        if match_keywords(ocr_text, ui_kw) or "screen" in fname_lower or "ui" in fname_lower:
            if "mobile" in fname_lower or "app" in fname_lower:
                return ImageCategory.MOBILE_UI, 0.90, AnalyzerType.MOBILE_UI_ANALYZER
            elif "web" in fname_lower or "http" in ocr_lower or "www" in ocr_lower:
                return ImageCategory.WEBSITE_UI, 0.90, AnalyzerType.WEBSITE_ANALYZER
            return ImageCategory.UI_SCREENSHOT, 0.90, AnalyzerType.UI_ANALYZER

        # 3. Documents & Research Papers
        doc_kw = ["abstract", "introduction", "references", "journal", "ieee", "arxiv", "fig.", "table 1", "invoice", "total"]
        if match_keywords(ocr_text, doc_kw) or "doc" in fname_lower or "paper" in fname_lower:
            if "abstract" in ocr_lower or "references" in ocr_lower:
                return ImageCategory.RESEARCH_PAPER, 0.92, AnalyzerType.DOCUMENT_ANALYZER
            elif "invoice" in ocr_lower or "total" in ocr_lower:
                return ImageCategory.INVOICE, 0.92, AnalyzerType.DOCUMENT_ANALYZER
            return ImageCategory.DOCUMENT, 0.88, AnalyzerType.DOCUMENT_ANALYZER

        # 4. Charts & Graphs
        chart_kw = ["chart", "graph", "axis", "pie", "bar", "legend", "distribution", "percentage"]
        if match_keywords(ocr_text, chart_kw) or "chart" in fname_lower or "graph" in cap_lower:
            return ImageCategory.CHART, 0.90, AnalyzerType.CHART_ANALYZER

        # 5. Architecture Diagrams & Flowcharts
        diag_kw = ["architecture", "flowchart", "process", "database", "server", "client", "step 1", "decision"]
        if match_keywords(ocr_text, diag_kw) or "diagram" in fname_lower or "flow" in fname_lower:
            if "flow" in fname_lower or "decision" in ocr_lower:
                return ImageCategory.FLOWCHART, 0.88, AnalyzerType.DIAGRAM_ANALYZER
            return ImageCategory.ARCHITECTURE_DIAGRAM, 0.88, AnalyzerType.DIAGRAM_ANALYZER

        # 6. Default Photo / General Image
        return ImageCategory.PHOTO, 0.75, AnalyzerType.PHOTO_ANALYZER
