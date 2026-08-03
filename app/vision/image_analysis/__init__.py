"""
app/vision/image_analysis package initializer
===============================================
Exposes public API for Specialized Image Analysis Framework, ImageClassifier,
domain analyzers (Code, UI, Document, Chart, Diagram, Table, Photo), and AnalysisManager.
"""

from app.vision.image_analysis.analysis_types import (
    ImageCategory,
    AnalysisType,
    AnalyzerType,
    ConfidenceLevel,
    ChartType,
    DiagramType,
    CodeLanguage,
    DocumentType
)
from app.vision.image_analysis.analysis_config import (
    ImageAnalysisConfig,
    get_analysis_config,
    set_analysis_config
)
from app.vision.image_analysis.analysis_models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisContext,
    UIStructure,
    CodeStructure,
    DocumentStructure,
    ChartStructure
)
from app.vision.image_analysis.analysis_utils import (
    match_keywords,
    format_domain_analysis_block
)
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
from app.vision.image_analysis.analysis_manager import AnalysisManager

__all__ = [
    # Enums
    "ImageCategory",
    "AnalysisType",
    "AnalyzerType",
    "ConfidenceLevel",
    "ChartType",
    "DiagramType",
    "CodeLanguage",
    "DocumentType",
    # Config
    "ImageAnalysisConfig",
    "get_analysis_config",
    "set_analysis_config",
    # Schemas
    "AnalysisRequest",
    "AnalysisResult",
    "AnalysisContext",
    "UIStructure",
    "CodeStructure",
    "DocumentStructure",
    "ChartStructure",
    # Utils
    "match_keywords",
    "format_domain_analysis_block",
    # Classifier & Analyzers
    "ImageClassifier",
    "UIAnalyzer",
    "MobileUIAnalyzer",
    "DesktopUIAnalyzer",
    "WebsiteAnalyzer",
    "CodeAnalyzer",
    "DocumentAnalyzer",
    "ChartAnalyzer",
    "TableAnalyzer",
    "DiagramAnalyzer",
    "HandwritingAnalyzer",
    "PhotoAnalyzer",
    # Manager Facade
    "AnalysisManager"
]
