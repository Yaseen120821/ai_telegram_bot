"""
app/tools/routing/router_config.py - Centralized Configuration for Decision Engine
==================================================================================
Manages confidence thresholds, rule priorities, candidate limits, fallback settings,
and feature flags for intent detection and tool routing.
"""

import os
from dataclasses import dataclass, field
from app.tools.routing.router_types import ConfidenceLevel


@dataclass
class ToolRouterConfig:
    """Central configuration for SANA AI Decision and Routing Engine."""

    # Confidence Thresholds
    high_confidence_threshold: float = 0.80
    medium_confidence_threshold: float = 0.60
    
    # Candidate limits & candidate scoring
    max_candidate_tools: int = 3
    enable_rule_engine: bool = True
    enable_llm_fallback: bool = True
    
    # RAG Routing Thresholds
    rag_keyword_trigger_threshold: float = 0.70
    
    # Logging & Debugging
    log_routing_decisions: bool = True
    cache_intent_patterns: bool = True
    
    @classmethod
    def from_env(cls) -> "ToolRouterConfig":
        """Instantiates configuration from environment variables with default thresholds."""
        return cls(
            high_confidence_threshold=float(os.getenv("SANA_ROUTER_HIGH_THRESH", "0.80")),
            medium_confidence_threshold=float(os.getenv("SANA_ROUTER_MED_THRESH", "0.60")),
            max_candidate_tools=int(os.getenv("SANA_ROUTER_MAX_CANDIDATES", "3")),
            enable_rule_engine=os.getenv("SANA_ROUTER_ENABLE_RULES", "true").lower() == "true",
            enable_llm_fallback=os.getenv("SANA_ROUTER_LLM_FALLBACK", "true").lower() == "true",
        )


_router_config_instance: ToolRouterConfig = ToolRouterConfig.from_env()


def get_router_config() -> ToolRouterConfig:
    """Returns global router configuration singleton."""
    return _router_config_instance


def set_router_config(config: ToolRouterConfig) -> None:
    """Overrides global router configuration."""
    global _router_config_instance
    _router_config_instance = config
