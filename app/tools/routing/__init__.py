"""
app/tools/routing package initializer - SANA AI Decision & Routing Subsystem
=============================================================================
Exposes public API for IntentDetector, RuleEngine, ConfidenceEngine, DecisionEngine, and ToolRouter.
"""

from app.tools.routing.router_types import (
    IntentType,
    RoutingMode,
    DecisionType,
    ConfidenceLevel,
    ParameterType
)
from app.tools.routing.router_config import ToolRouterConfig, get_router_config, set_router_config
from app.tools.routing.router_models import (
    ExtractedParameter,
    IntentResult,
    ToolCandidate,
    ConfidenceScore,
    RoutingDecision,
    RoutingStatistics
)
from app.tools.routing.router_utils import (
    normalize_query_text,
    extract_filepath_from_text,
    extract_arithmetic_expression
)
from app.tools.routing.rule_engine import RuleEngine
from app.tools.routing.intent_detector import IntentDetector
from app.tools.routing.confidence_engine import ConfidenceEngine
from app.tools.routing.decision_engine import DecisionEngine
from app.tools.routing.tool_router import ToolRouter

__all__ = [
    # Enums
    "IntentType",
    "RoutingMode",
    "DecisionType",
    "ConfidenceLevel",
    "ParameterType",
    # Config
    "ToolRouterConfig",
    "get_router_config",
    "set_router_config",
    # Models
    "ExtractedParameter",
    "IntentResult",
    "ToolCandidate",
    "ConfidenceScore",
    "RoutingDecision",
    "RoutingStatistics",
    # Utils
    "normalize_query_text",
    "extract_filepath_from_text",
    "extract_arithmetic_expression",
    # Core Classes
    "RuleEngine",
    "IntentDetector",
    "ConfidenceEngine",
    "DecisionEngine",
    "ToolRouter"
]
