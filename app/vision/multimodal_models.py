"""
app/vision/multimodal_models.py - Multimodal Pipeline Data Models
=================================================================
Defines strongly-typed dataclasses for vision context, OCR context, domain analysis,
integrated multimodal prompt context, and image conversation references.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.vision.multimodal_types import (
    ContextSource,
    ImageRole,
    VisionConfidence,
    ReasoningMode,
    ResponseMode
)
from app.vision.image_analysis.analysis_types import ImageCategory


@dataclass
class VisionContext:
    """Raw visual perception context from Florence-2 vision model."""
    short_caption: str = ""
    detailed_caption: str = ""
    objects_count: int = 0
    confidence: float = 0.90
    latency_ms: float = 0.0


@dataclass
class OCRContext:
    """Raw text extraction context from EasyOCR."""
    raw_text: str = ""
    line_count: int = 0
    word_count: int = 0
    confidence: float = 0.88
    regions_count: int = 0


@dataclass
class AnalysisContextData:
    """Domain-specific analysis output from AnalysisManager."""
    category: ImageCategory = ImageCategory.UNKNOWN
    domain_summary: str = ""
    code_language: Optional[str] = None
    detected_error: Optional[str] = None
    ui_buttons: List[str] = field(default_factory=list)
    document_title: Optional[str] = None
    chart_type: Optional[str] = None


@dataclass
class ImageContextData:
    """Aggregated context object representing one processed image asset."""
    image_id: str = field(default_factory=lambda: f"img_{uuid.uuid4().hex[:12]}")
    file_path: str = ""
    role: ImageRole = ImageRole.PRIMARY_SUBJECT
    vision: VisionContext = field(default_factory=VisionContext)
    ocr: OCRContext = field(default_factory=OCRContext)
    analysis: AnalysisContextData = field(default_factory=AnalysisContextData)
    formatted_context_block: str = ""
    processed_timestamp: float = field(default_factory=time.time)


@dataclass
class MultimodalContext:
    """Master multimodal context payload passed to ContextBuilder & PromptBuilder."""
    query: str = ""
    user_id: str = "default_user"
    conversation_id: str = "default_session"
    reasoning_mode: ReasoningMode = ReasoningMode.TEXT_ONLY
    memory_context: Optional[str] = None
    emotion_context: Optional[Any] = None
    rag_context: Optional[str] = None
    tool_context: Optional[Any] = None
    images: List[ImageContextData] = field(default_factory=list)
    formatted_vision_block: str = ""
    total_token_estimate: int = 0


@dataclass
class IntegratedPrompt:
    """Final assembled ChatML prompt payload ready for Qwen tokenizer."""
    prompt_text: str = ""
    estimated_tokens: int = 0
    budget_used_pct: float = 0.0
    section_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class ImageConversationReference:
    """Lightweight memory reference stored in conversation history for follow-up questions."""
    image_id: str
    file_path: str
    category: ImageCategory
    summary: str
    ocr_snippet: str
    timestamp: float = field(default_factory=time.time)
