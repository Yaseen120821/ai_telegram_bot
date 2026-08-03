"""
app/vision/image_analysis/analysis_types.py - Image Analysis Enumerations
=========================================================================
Defines standard enums for image categories, specialized analyzer targets,
confidence levels, chart types, diagram topologies, programming languages, and document types.
"""

from enum import Enum


class ImageCategory(str, Enum):
    """Specific visual domain category classification."""
    PHOTO = "photo"
    SCREENSHOT = "screenshot"
    CODE_SCREENSHOT = "code_screenshot"
    IDE_SCREENSHOT = "ide_screenshot"
    TERMINAL_SCREENSHOT = "terminal_screenshot"
    UI_SCREENSHOT = "ui_screenshot"
    MOBILE_UI = "mobile_ui"
    DESKTOP_UI = "desktop_ui"
    WEBSITE_UI = "website_ui"
    DOCUMENT = "document"
    RESEARCH_PAPER = "research_paper"
    ASSIGNMENT = "assignment"
    INVOICE = "invoice"
    CHART = "chart"
    TABLE = "table"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    FLOWCHART = "flowchart"
    HANDWRITING = "handwriting"
    UNKNOWN = "unknown"


class AnalysisType(str, Enum):
    """Operational mode for specialized visual analysis."""
    GENERAL = "general"
    UI_UX = "ui_ux"
    CODE_DEBUG = "code_debug"
    DOCUMENT_PARSING = "document_parsing"
    DATA_TREND = "data_trend"
    ARCHITECTURE_TOPOLOGY = "architecture_topology"
    TEXT_TRANSCRIPTION = "text_transcription"


class AnalyzerType(str, Enum):
    """Identifier for specialized domain analyzer component."""
    UI_ANALYZER = "ui_analyzer"
    CODE_ANALYZER = "code_analyzer"
    DOCUMENT_ANALYZER = "document_analyzer"
    CHART_ANALYZER = "chart_analyzer"
    TABLE_ANALYZER = "table_analyzer"
    DIAGRAM_ANALYZER = "diagram_analyzer"
    HANDWRITING_ANALYZER = "handwriting_analyzer"
    PHOTO_ANALYZER = "photo_analyzer"
    WEBSITE_ANALYZER = "website_analyzer"
    MOBILE_UI_ANALYZER = "mobile_ui_analyzer"
    DESKTOP_UI_ANALYZER = "desktop_ui_analyzer"
    DEFAULT_ANALYZER = "default_analyzer"


class ConfidenceLevel(str, Enum):
    """Confidence level assigned to classification and domain analysis."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class ChartType(str, Enum):
    """Chart and graph visual type classification."""
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    AREA_CHART = "area_chart"
    UNKNOWN = "unknown"


class DiagramType(str, Enum):
    """Technical diagram topology classification."""
    ARCHITECTURE = "architecture"
    FLOWCHART = "flowchart"
    MIND_MAP = "mind_map"
    SEQUENCE_DIAGRAM = "sequence_diagram"
    ER_DIAGRAM = "er_diagram"
    UNKNOWN = "unknown"


class CodeLanguage(str, Enum):
    """Detected programming language in code screenshots."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    HTML_CSS = "html_css"
    SQL = "sql"
    SHELL = "shell"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """Document layout category classification."""
    RESEARCH_PAPER = "research_paper"
    INVOICE = "invoice"
    ASSIGNMENT = "assignment"
    ARTICLE = "article"
    FORM = "form"
    UNKNOWN = "unknown"
