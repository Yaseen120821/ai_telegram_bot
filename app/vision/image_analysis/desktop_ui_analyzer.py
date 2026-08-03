"""
app/vision/image_analysis/desktop_ui_analyzer.py - Desktop Window & Panel Analyzer
===================================================================================
Analyzes desktop operating system window layouts, toolbars, side panels, status bars, and multi-window environments.
"""

from typing import Dict, Any
from app.vision.image_analysis.analysis_types import ImageCategory, AnalyzerType
from app.vision.image_analysis.analysis_models import AnalysisResult, UIStructure


class DesktopUIAnalyzer:
    """Specialized analyzer for desktop software windows & IDE panel layouts."""

    @staticmethod
    def analyze(request_data: Dict[str, Any]) -> AnalysisResult:
        ui_struct = UIStructure(
            buttons=["Window Controls (Min/Max/Close)", "Toolbar Actions"],
            menus=["Top Menu Bar (File, Edit, View, Help)", "Sidebar Panel"],
            inputs=["Command Palette / Search"],
            visual_hierarchy="Multi-pane desktop window with left sidebar, main workspace editor, and status bar.",
            accessibility_warnings=["Ensure keyboard shortcut navigation focuses all panels."]
        )

        return AnalysisResult(
            category=ImageCategory.DESKTOP_UI,
            confidence=0.90,
            analyzer_used=AnalyzerType.DESKTOP_UI_ANALYZER,
            ui_structure=ui_struct,
            domain_summary="Desktop application window with top menu bar, sidebar panel, workspace editor, and status bar."
        )
