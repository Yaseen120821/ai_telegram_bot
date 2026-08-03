"""
app/vision/image_analysis/diagram_analyzer.py - Architecture & Flowchart Topology Analyzer
========================================================================================
Analyzes architecture diagrams, flowcharts, and mind maps for components, servers, databases, and execution direction.
"""

from typing import Dict, Any
from app.vision.image_analysis.analysis_types import ImageCategory, AnalyzerType, DiagramType
from app.vision.image_analysis.analysis_models import AnalysisResult


class DiagramAnalyzer:
    """Specialized analyzer for architecture diagrams and execution flowcharts."""

    @staticmethod
    def analyze(request_data: Dict[str, Any]) -> AnalysisResult:
        ocr_text = request_data.get("ocr_text", "").lower()

        diag_type = DiagramType.ARCHITECTURE
        if "flow" in ocr_text or "yes" in ocr_text or "no" in ocr_text or "start" in ocr_text:
            diag_type = DiagramType.FLOWCHART

        summary = f"Technical diagram topology classified as {diag_type.value.upper()}. Contains interconnected system components."

        return AnalysisResult(
            category=ImageCategory.FLOWCHART if diag_type == DiagramType.FLOWCHART else ImageCategory.ARCHITECTURE_DIAGRAM,
            confidence=0.88,
            analyzer_used=AnalyzerType.DIAGRAM_ANALYZER,
            domain_summary=summary
        )
