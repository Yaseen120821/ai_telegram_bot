"""
app/tools/tool_router.py - Top-Level Re-export Alias for app.tools.routing.tool_router
=====================================================================================
Maintains backward compatibility by re-exporting ToolRouter, RoutingDecision, and candidates.
"""

from app.tools.routing.router_types import IntentType, RoutingMode, DecisionType, ConfidenceLevel
from app.tools.routing.router_models import ToolCandidate, RoutingDecision, ConfidenceScore, ExtractedParameter
from app.tools.routing.router_config import ToolRouterConfig, get_router_config, set_router_config
from app.tools.routing.rule_engine import RuleEngine
from app.tools.routing.intent_detector import IntentDetector
from app.tools.routing.confidence_engine import ConfidenceEngine
from app.tools.routing.decision_engine import DecisionEngine
from app.tools.routing.tool_router import ToolRouter

# Alias for backward compatibility with Part 1 legacy test signatures
ToolRoutingCandidate = ToolCandidate

__all__ = [
    "IntentType",
    "RoutingMode",
    "DecisionType",
    "ConfidenceLevel",
    "ToolCandidate",
    "ToolRoutingCandidate",
    "RoutingDecision",
    "ConfidenceScore",
    "ExtractedParameter",
    "ToolRouterConfig",
    "get_router_config",
    "set_router_config",
    "RuleEngine",
    "IntentDetector",
    "ConfidenceEngine",
    "DecisionEngine",
    "ToolRouter"
]
