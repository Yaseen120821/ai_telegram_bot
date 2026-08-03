"""
app/tools/registry/registry_models.py - Data Models for Tool Registry
======================================================================
Defines strongly-typed dataclasses for tool registrations, metadata inspection,
search results, registry telemetry statistics, health status, and startup reports.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel, ToolState
from app.tools.registry.registry_types import RegistrationStatus, ToolVisibility, RegistryState, RegistryError


@dataclass
class EnhancedToolMetadata:
    """Comprehensive tool metadata record stored in the registry cache."""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "SANA AI Core Team"
    permission_level: PermissionLevel = PermissionLevel.SAFE
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    tags: List[str] = field(default_factory=list)
    state: ToolState = ToolState.ENABLED
    visibility: ToolVisibility = ToolVisibility.PUBLIC
    is_experimental: bool = False


@dataclass
class ToolRegistration:
    """Record of a tool registration entry managed by RegistryManager."""
    tool_id: str
    tool_name: str
    tool_class: Type[BaseTool]
    instance: Optional[BaseTool] = None
    metadata: Optional[EnhancedToolMetadata] = None
    status: RegistrationStatus = RegistrationStatus.SUCCESS
    error_message: Optional[str] = None
    registered_at: float = field(default_factory=time.time)


@dataclass
class ToolSearchResult:
    """Container holding results of a multi-criteria tool search query."""
    query: str
    matched_tools: List[BaseTool] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0


@dataclass
class RegistryStatistics:
    """Operational telemetry breakdown of all tools in the registry."""
    total_registered: int = 0
    enabled_count: int = 0
    disabled_count: int = 0
    deprecated_count: int = 0
    experimental_count: int = 0
    category_breakdown: Dict[str, int] = field(default_factory=dict)
    permission_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class RegistryHealth:
    """Health diagnostic status of the registry sub-system."""
    state: RegistryState = RegistryState.UNINITIALIZED
    is_healthy: bool = True
    loaded_tools_count: int = 0
    failed_imports_count: int = 0
    warning_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class RegistryReport:
    """Comprehensive diagnostic startup and discovery audit report."""
    report_id: str = field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    total_scanned: int = 0
    total_registered: int = 0
    total_failed: int = 0
    registrations: List[ToolRegistration] = field(default_factory=list)
    health: Optional[RegistryHealth] = None
