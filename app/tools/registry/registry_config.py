"""
app/tools/registry/registry_config.py - Centralized Configuration for Tool Registry
===================================================================================
Manages discovery paths, auto-registration modes, metadata validation policies,
tool caps, and feature flags for the tool discovery framework.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from app.tools.registry.registry_types import DiscoveryMode


@dataclass
class ToolRegistryConfig:
    """Central configuration settings for SANA AI Tool Registry and Dynamic Discovery."""

    discovery_mode: DiscoveryMode = DiscoveryMode.HYBRID
    auto_register: bool = True
    max_tools: int = 500
    allow_experimental: bool = False
    strict_metadata_validation: bool = True
    lazy_discovery: bool = False
    
    # Paths relative to project root or absolute paths for dynamic tool scanner
    discovery_paths: List[str] = field(default_factory=lambda: [
        "app/tools/system",
        "app/tools/filesystem",
        "app/tools/internet",
        "app/tools/productivity",
        "app/tools/developer",
        "app/tools/ai",
        "app/tools/communication",
        "app/tools/automation",
    ])
    
    # Excluded files/dirs from auto discovery
    excluded_filenames: List[str] = field(default_factory=lambda: [
        "__init__.py",
        "base_tool.py",
        "tool_types.py",
        "tool_models.py",
    ])
    
    # Telemetry and logging configuration
    log_discovery_details: bool = True
    cache_metadata: bool = True

    @classmethod
    def from_env(cls) -> "ToolRegistryConfig":
        """Instantiates config from environment variables with sensible defaults."""
        mode_str = os.getenv("SANA_REGISTRY_MODE", "hybrid").lower()
        mode_map = {m.value: m for m in DiscoveryMode}
        selected_mode = mode_map.get(mode_str, DiscoveryMode.HYBRID)

        return cls(
            discovery_mode=selected_mode,
            auto_register=os.getenv("SANA_REGISTRY_AUTO_REGISTER", "true").lower() == "true",
            max_tools=int(os.getenv("SANA_REGISTRY_MAX_TOOLS", "500")),
            allow_experimental=os.getenv("SANA_REGISTRY_ALLOW_EXPERIMENTAL", "false").lower() == "true",
            strict_metadata_validation=os.getenv("SANA_REGISTRY_STRICT_VAL", "true").lower() == "true",
        )


_registry_config_instance: ToolRegistryConfig = ToolRegistryConfig.from_env()


def get_registry_config() -> ToolRegistryConfig:
    """Returns global singleton registry configuration instance."""
    return _registry_config_instance


def set_registry_config(config: ToolRegistryConfig) -> None:
    """Overrides global registry configuration."""
    global _registry_config_instance
    _registry_config_instance = config
