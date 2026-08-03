"""
app/vision/image_analysis/analysis_manager.py - Main Facade for Image Analysis Framework
========================================================================================
Coordinates multi-signal image classification, routes requests to specialized domain analyzers,
and assembles formatted domain prompt context blocks for PromptBuilder injection.
"""

import time
import os
import logging
from typing import Optional, Dict, Any

from app.vision.image_analysis.analysis_types import ImageCategory, AnalyzerType
from app.vision.image_analysis.analysis_config import get_analysis_config, ImageAnalysisConfig
from app.vision.image_analysis.analysis_models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisContext
)
from app.vision.image_analysis.analysis_utils import format_domain_analysis_block
from app.vision.image_analysis.image_classifier import ImageClassifier
from app.vision.image_analysis.ui_analyzer import UIAnalyzer
from app.vision.image_analysis.mobile_ui_analyzer import MobileUIAnalyzer
from app.vision.image_analysis.desktop_ui_analyzer import DesktopUIAnalyzer
from app.vision.image_analysis.website_analyzer import WebsiteAnalyzer
from app.vision.image_analysis.code_analyzer import CodeAnalyzer
from app.vision.image_analysis.document_analyzer import DocumentAnalyzer
from app.vision.image_analysis.chart_analyzer import ChartAnalyzer
from app.vision.image_analysis.table_analyzer import TableAnalyzer
from app.vision.image_analysis.diagram_analyzer import DiagramAnalyzer
from app.vision.image_analysis.handwriting_analyzer import HandwritingAnalyzer
from app.vision.image_analysis.photo_analyzer import PhotoAnalyzer

logger = logging.getLogger("sana_ai.vision.analysis.manager")


class AnalysisManager:
    """Central orchestrator facade routing classified visual assets to specialized domain analyzers."""

    _instance: Optional["AnalysisManager"] = None

    def __init__(self, config: Optional[ImageAnalysisConfig] = None):
        self.config = config or get_analysis_config()

    @classmethod
    def get_instance(cls) -> "AnalysisManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def analyze_image(self, request: AnalysisRequest) -> AnalysisContext:
        """
        Main entry point for domain-specific visual analysis.
        
        Workflow:
        1. Classify image category if not provided.
        2. Select matching specialized domain analyzer.
        3. Execute domain analysis.
        4. Assemble formatted domain prompt context block.
        5. Return AnalysisContext.
        """
        start_time = time.perf_counter()
        req_id = request.request_id
        img_path = request.image_path

        logger.info(f"AnalysisManager processing request [{req_id}] for image '{img_path}'.")

        # 1. Classify image category if absent
        category = request.category
        confidence = 0.85
        analyzer_type = AnalyzerType.DEFAULT_ANALYZER

        if category is None or category == ImageCategory.UNKNOWN:
            category, confidence, analyzer_type = ImageClassifier.classify(
                file_path=img_path,
                caption=request.caption,
                ocr_text=request.ocr_text,
                objects=request.objects
            )

        # 2. Dispatch to Specialized Analyzer
        req_dict = {
            "image_path": img_path,
            "caption": request.caption,
            "ocr_text": request.ocr_text,
            "objects": request.objects,
            "metadata": request.metadata
        }

        result = self._dispatch_analyzer(category, analyzer_type, req_dict)
        result.confidence = confidence
        result.execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Format Prompt Context Block
        formatted_prompt_block = format_domain_analysis_block(
            result=result,
            file_name=os.path.basename(img_path)
        )

        logger.info(f"AnalysisManager completed request [{req_id}] in {result.execution_time_ms:.2f}ms [Category: {category.value}].")

        return AnalysisContext(
            request_id=req_id,
            file_path=img_path,
            category=category,
            formatted_prompt_block=formatted_prompt_block,
            result=result
        )

    def _dispatch_analyzer(
        self,
        category: ImageCategory,
        analyzer_type: AnalyzerType,
        request_data: Dict[str, Any]
    ) -> AnalysisResult:
        """Routes payload to specialized domain analyzer implementation."""
        if category in (ImageCategory.CODE_SCREENSHOT, ImageCategory.IDE_SCREENSHOT, ImageCategory.TERMINAL_SCREENSHOT):
            return CodeAnalyzer.analyze(request_data)

        elif category in (ImageCategory.DOCUMENT, ImageCategory.RESEARCH_PAPER, ImageCategory.ASSIGNMENT, ImageCategory.INVOICE):
            return DocumentAnalyzer.analyze(request_data)

        elif category == ImageCategory.CHART:
            return ChartAnalyzer.analyze(request_data)

        elif category == ImageCategory.TABLE:
            return TableAnalyzer.analyze(request_data)

        elif category in (ImageCategory.ARCHITECTURE_DIAGRAM, ImageCategory.FLOWCHART):
            return DiagramAnalyzer.analyze(request_data)

        elif category == ImageCategory.HANDWRITING:
            return HandwritingAnalyzer.analyze(request_data)

        elif category == ImageCategory.MOBILE_UI:
            return MobileUIAnalyzer.analyze(request_data)

        elif category == ImageCategory.DESKTOP_UI:
            return DesktopUIAnalyzer.analyze(request_data)

        elif category == ImageCategory.WEBSITE_UI:
            return WebsiteAnalyzer.analyze(request_data)

        elif category == ImageCategory.UI_SCREENSHOT:
            return UIAnalyzer.analyze(request_data)

        return PhotoAnalyzer.analyze(request_data)
