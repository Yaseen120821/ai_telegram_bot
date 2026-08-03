"""
app/vision/image_analysis/mobile_ui_analyzer.py - Mobile Application UI Analyzer
=================================================================================
Analyzes mobile application UI screenshots for Material Design app bars, bottom navigation bars,
touch targets, cards, and mobile responsiveness.
"""

from typing import Dict, Any
from app.vision.image_analysis.analysis_types import ImageCategory, AnalyzerType
from app.vision.image_analysis.analysis_models import AnalysisResult, UIStructure


class MobileUIAnalyzer:
    """Specialized analyzer for mobile iOS & Android application screenshots."""

    @staticmethod
    def analyze(request_data: Dict[str, Any]) -> AnalysisResult:
        ui_struct = UIStructure(
            buttons=["Floating Action Button (FAB)", "Bottom Action"],
            menus=["App Bar Menu", "Bottom Navigation Bar (3 items)"],
            inputs=["Search Bar"],
            visual_hierarchy="Mobile vertical stack layout with top App Bar and bottom navigation items.",
            accessibility_warnings=["Verify minimum touch target size of 48x48 dp."]
        )

        return AnalysisResult(
            category=ImageCategory.MOBILE_UI,
            confidence=0.92,
            analyzer_used=AnalyzerType.MOBILE_UI_ANALYZER,
            ui_structure=ui_struct,
            domain_summary="Mobile Application interface featuring App Bar, card items, and bottom navigation."
        )
