"""
app/tools/routing/router_models.py - Data Models for Decision & Routing Subsystem
==================================================================================
Defines strongly-typed dataclasses for extracted parameters, intent results,
candidate scoring, confidence ratings, routing decisions, and statistics.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.tools.base_tool import BaseTool
from app.tools.routing.router_types import IntentType, RoutingMode, DecisionType, ConfidenceLevel, ParameterType


@dataclass
class ExtractedParameter:
    """Represents a key-value parameter extracted from user query text."""
    name: str
    value: Any
    param_type: ParameterType = ParameterType.STRING
    is_required: bool = False
    confidence: float = 1.0


@dataclass
class IntentResult:
    """Structure produced by IntentDetector following user query analysis."""
    primary_intent: IntentType
    raw_query: str
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    intent_confidence: float = 0.0
    matched_rule: Optional[str] = None
    goal_summary: str = ""


@dataclass
class ToolCandidate:
    """Scored tool candidate evaluated during routing."""
    tool: BaseTool
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ConfidenceScore:
    """Structured confidence evaluation container."""
    numeric_score: float
    level: ConfidenceLevel
    explanation: str = ""


@dataclass
class RoutingDecision:
    """Comprehensive decision payload produced by DecisionEngine."""
    routing_mode: RoutingMode
    decision_type: DecisionType
    selected_tool: Optional[BaseTool] = None
    candidates: List[ToolCandidate] = field(default_factory=list)
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(0.0, ConfidenceLevel.NONE))
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    fallback_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def should_call_tool(self) -> bool:
        return self.routing_mode in (RoutingMode.TOOL, RoutingMode.TOOL_AND_RAG, RoutingMode.MULTIPLE_TOOLS) and self.selected_tool is not None


@dataclass
class RoutingStatistics:
    """Operational telemetry stats for routing decisions."""
    total_queries: int = 0
    rule_matched_count: int = 0
    llm_fallback_count: int = 0
    direct_response_count: int = 0
    tool_selected_count: int = 0
    rag_selected_count: int = 0
