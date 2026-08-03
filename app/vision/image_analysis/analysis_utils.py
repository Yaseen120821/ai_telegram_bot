"""
app/vision/image_analysis/analysis_utils.py - Analysis Helper & Keyword Utilities
=================================================================================
Provides OCR keyword matching, regex text parsing, and domain prompt context block formatters.
"""

import re
from typing import List, Dict, Any
from app.vision.image_analysis.analysis_models import AnalysisResult


def match_keywords(text: str, keywords: List[str]) -> List[str]:
    """Matches text against list of keywords case-insensitively."""
    if not text:
        return []
    text_lower = text.lower()
    return [kw for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)]


def format_domain_analysis_block(result: AnalysisResult, file_name: str = "") -> str:
    """Constructs domain-specific prompt context block for PromptBuilder."""
    lines = [
        "=== SPECIALIZED DOMAIN ANALYSIS ===",
        f"Target File: {file_name or 'Uploaded Image'}",
        f"Domain Category: {result.category.value.upper()} (Confidence: {int(result.confidence * 100)}%)",
        f"Analyzer: {result.analyzer_used.value}"
    ]

    if result.domain_summary:
        lines.append(f"Domain Summary: {result.domain_summary}")

    # Code Analysis Block
    if result.code_structure:
        cs = result.code_structure
        lines.append(f"Code Language: {cs.detected_language.value.upper()}")
        if cs.error_type:
            lines.append(f"Detected Error: {cs.error_type}")
        if cs.stack_trace:
            lines.append(f"Stack Trace Summary:\n{cs.stack_trace}")
        if cs.suggested_fix:
            lines.append(f"Suggested Fix: {cs.suggested_fix}")

    # UI Analysis Block
    if result.ui_structure:
        ui = result.ui_structure
        if ui.buttons:
            lines.append(f"UI Buttons: {', '.join(ui.buttons)}")
        if ui.menus:
            lines.append(f"UI Menus: {', '.join(ui.menus)}")
        if ui.accessibility_warnings:
            lines.append(f"Accessibility Warnings: {'; '.join(ui.accessibility_warnings)}")

    # Document Analysis Block
    if result.document_structure:
        ds = result.document_structure
        if ds.title:
            lines.append(f"Document Title: {ds.title}")
        if ds.authors:
            lines.append(f"Authors: {', '.join(ds.authors)}")
        if ds.abstract:
            lines.append(f"Abstract Summary: {ds.abstract}")

    # Chart Analysis Block
    if result.chart_structure:
        chs = result.chart_structure
        lines.append(f"Chart Type: {chs.chart_type.value.upper()}")
        if chs.data_trends:
            lines.append(f"Identified Trends: {'; '.join(chs.data_trends)}")

    lines.append("=== END DOMAIN ANALYSIS ===")
    return "\n".join(lines)
