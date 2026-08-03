"""
app/tools/tool_config.py - Centralized Configuration for Tool Calling Framework
=============================================================================
Manages dynamic settings, default thresholds, timeouts, security policies,
and feature flags for tool registration, routing, and execution.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from app.tools.tool_types import PermissionLevel


@dataclass
class ToolFrameworkConfig:
    """Central configuration class for SANA AI Tool Calling Framework."""

    # Default execution limits
    default_timeout_seconds: float = 10.0
    max_retries: int = 2
    max_concurrent_tools: int = 5
    
    # Global state settings
    enabled_by_default: bool = True
    disabled_tools: List[str] = field(default_factory=list)
    
    # Security & Permission Defaults
    default_permission_level: PermissionLevel = PermissionLevel.SAFE
    require_user_confirmation: bool = True
    auto_approve_safe_tools: bool = True
    admin_mode_enabled: bool = False
    
    # Intent Routing Defaults
    routing_confidence_threshold: float = 0.65
    enable_fuzzy_matching: bool = True
    max_candidate_tools: int = 3
    
    # Metrics & Logging
    log_execution_payloads: bool = True
    enable_metrics_collection: bool = True
    
    @classmethod
    def from_env(cls) -> "ToolFrameworkConfig":
        """Instantiates configuration from environment variables with safe defaults."""
        return cls(
            default_timeout_seconds=float(os.getenv("SANA_TOOL_TIMEOUT", "10.0")),
            max_retries=int(os.getenv("SANA_TOOL_MAX_RETRIES", "2")),
            max_concurrent_tools=int(os.getenv("SANA_TOOL_MAX_CONCURRENT", "5")),
            enabled_by_default=os.getenv("SANA_TOOL_ENABLED_DEFAULT", "true").lower() == "true",
            require_user_confirmation=os.getenv("SANA_TOOL_REQUIRE_CONFIRMATION", "true").lower() == "true",
            auto_approve_safe_tools=os.getenv("SANA_TOOL_AUTO_APPROVE_SAFE", "true").lower() == "true",
            routing_confidence_threshold=float(os.getenv("SANA_TOOL_ROUTING_THRESHOLD", "0.65")),
        )


# Global singleton instance of configuration
_config_instance: ToolFrameworkConfig = ToolFrameworkConfig.from_env()


def get_tool_config() -> ToolFrameworkConfig:
    """Returns global tool framework configuration singleton."""
    return _config_instance


def set_tool_config(config: ToolFrameworkConfig) -> None:
    """Overrides global tool framework configuration."""
    global _config_instance
    _config_instance = config
