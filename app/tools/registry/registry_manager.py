"""
app/tools/registry/registry_manager.py - Main Facade for Tool Registry Subsystem
================================================================================
Coordinates ToolDiscovery, ToolLoader, MetadataManager, and RegistryValidator
to deliver dynamic registration, state management, searching, and health reporting.
"""

import time
import logging
from typing import Dict, List, Optional, Set, Type, Any
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel, ToolState
from app.tools.registry.registry_types import RegistryState, RegistrationStatus, ToolVisibility
from app.tools.registry.registry_config import get_registry_config, ToolRegistryConfig
from app.tools.registry.registry_models import (
    ToolRegistration,
    EnhancedToolMetadata,
    ToolSearchResult,
    RegistryStatistics,
    RegistryHealth,
    RegistryReport
)
from app.tools.registry.metadata_manager import MetadataManager
from app.tools.registry.tool_discovery import ToolDiscovery
from app.tools.registry.tool_loader import ToolLoader
from app.tools.registry.registry_validator import RegistryValidator
from app.tools.tool_exceptions import ToolNotFoundException, DuplicateToolException

logger = logging.getLogger("sana_ai.tools.registry.manager")


class RegistryManager:
    """
    Central Orchestrator Facade for Tool Registration & Discovery.
    
    Architectural Strict Boundaries:
    - Never executes tools.
    - Never communicates directly with Telegram UI.
    - Never formats LLM system prompt templates.
    """

    _instance: Optional["RegistryManager"] = None

    def __init__(self, config: Optional[ToolRegistryConfig] = None):
        self.config = config or get_registry_config()
        self._tools: Dict[str, BaseTool] = {}
        self._registrations: Dict[str, ToolRegistration] = {}
        self.metadata_manager = MetadataManager()
        self.discovery = ToolDiscovery(config=self.config)
        self.loader = ToolLoader()
        self.validator = RegistryValidator()
        self.state: RegistryState = RegistryState.UNINITIALIZED
        self.last_startup_report: Optional[RegistryReport] = None

    @classmethod
    def get_instance(cls) -> "RegistryManager":
        """Returns RegistryManager singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        if cls._instance.state == RegistryState.UNINITIALIZED:
            try:
                cls._instance.initialize()
            except Exception as err:
                logger.warning(f"RegistryManager auto-initialization notice: {err}")
        return cls._instance

    def initialize(self) -> RegistryReport:
        """
        Executes dynamic discovery, loads discovered tool classes,
        registers tools into memory cache, and builds startup health report.
        """
        if self.state == RegistryState.READY and self.last_startup_report:
            logger.debug("Tool Registry already initialized and READY.")
            return self.last_startup_report

        self.state = RegistryState.DISCOVERING
        logger.info("Initializing Tool Registry...")

        # 1. Discover tool classes dynamically
        discovered_classes = self.discovery.discover_tools()

        # 2. Load & Validate discovered tools
        self.state = RegistryState.LOADING
        registration_records: List[ToolRegistration] = []

        for cls in discovered_classes:
            reg_record = self.loader.load_tool_class(cls, self._tools)
            registration_records.append(reg_record)
            if reg_record.status == RegistrationStatus.SUCCESS and reg_record.instance:
                self._apply_registration(reg_record)

        # 3. Build Startup Audit Report
        report = self.loader.generate_startup_report(registration_records)
        self.last_startup_report = report
        self.state = RegistryState.READY
        logger.info(f"Tool Registry initialization finished cleanly. State: READY ({len(self._tools)} tools registered).")
        return report

    def register_tool(self, tool: BaseTool) -> ToolRegistration:
        """Manually registers an instantiated BaseTool object."""
        self.validator.validate_tool_instance(tool, self._tools)
        tool_name = tool.name
        
        reg_record = self.loader.load_tool_class(type(tool), self._tools)
        # Override with manual instance
        reg_record.instance = tool
        reg_record.status = RegistrationStatus.SUCCESS
        reg_record.error_message = None

        self._apply_registration(reg_record)
        logger.info(f"Manually registered tool '{tool_name}' [Category: {tool.category.value}].")
        return reg_record

    def unregister_tool(self, tool_name: str) -> None:
        """Removes a tool from registry and metadata cache."""
        if tool_name not in self._tools:
            raise ToolNotFoundException(tool_name)
        
        del self._tools[tool_name]
        if tool_name in self._registrations:
            del self._registrations[tool_name]
        self.metadata_manager.remove_tool(tool_name)
        logger.info(f"Unregistered tool '{tool_name}'.")

    def get_tool(self, tool_name: str) -> BaseTool:
        """Retrieves registered tool instance by name."""
        if tool_name not in self._tools:
            raise ToolNotFoundException(tool_name)
        return self._tools[tool_name]

    def has_tool(self, tool_name: str) -> bool:
        """Checks if tool name exists in registry."""
        return tool_name in self._tools

    def enable_tool(self, tool_name: str) -> None:
        """Enables a tool for execution."""
        tool = self.get_tool(tool_name)
        tool.metadata.state = ToolState.ENABLED
        meta = self.metadata_manager.get_metadata(tool_name)
        if meta:
            meta.state = ToolState.ENABLED
        logger.info(f"Enabled tool '{tool_name}'.")

    def disable_tool(self, tool_name: str) -> None:
        """Disables a tool from execution."""
        tool = self.get_tool(tool_name)
        tool.metadata.state = ToolState.DISABLED
        meta = self.metadata_manager.get_metadata(tool_name)
        if meta:
            meta.state = ToolState.DISABLED
        logger.info(f"Disabled tool '{tool_name}'.")

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        only_enabled: bool = True,
        permission_level: Optional[PermissionLevel] = None
    ) -> List[BaseTool]:
        """Filters and lists registered tools."""
        result = []
        for tool in self._tools.values():
            if only_enabled and not tool.is_enabled:
                continue
            if category is not None and tool.category != category:
                continue
            if permission_level is not None and tool.permission_level != permission_level:
                continue
            result.append(tool)
        return result

    def search_tools(
        self,
        query: Optional[str] = None,
        category: Optional[ToolCategory] = None,
        permission_level: Optional[PermissionLevel] = None,
        tag: Optional[str] = None,
        only_enabled: bool = True
    ) -> ToolSearchResult:
        """
        Multi-criteria tool search across metadata indexes.
        """
        start_time = time.perf_counter()
        candidates: Set[str] = set(self._tools.keys())

        # Filter by Category Index
        if category:
            cat_matches = self.metadata_manager.search_by_category(category)
            candidates.intersection_update(cat_matches)

        # Filter by Permission Index
        if permission_level:
            perm_matches = self.metadata_manager.search_by_permission(permission_level)
            candidates.intersection_update(perm_matches)

        # Filter by Tag Index
        if tag:
            tag_matches = self.metadata_manager.search_by_tag(tag)
            candidates.intersection_update(tag_matches)

        # Query text keyword filter
        matched_tools: List[BaseTool] = []
        q_lower = query.lower().strip() if query else ""

        for tool_name in candidates:
            tool = self._tools[tool_name]
            if only_enabled and not tool.is_enabled:
                continue
            
            if q_lower:
                in_name = q_lower in tool.name.lower()
                in_desc = q_lower in tool.description.lower()
                in_tags = any(q_lower in t.lower() for t in tool.metadata.tags)
                if not (in_name or in_desc or in_tags):
                    continue
            
            matched_tools.append(tool)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0

        return ToolSearchResult(
            query=query or "",
            matched_tools=matched_tools,
            total_found=len(matched_tools),
            search_time_ms=elapsed_ms
        )

    def get_registry_statistics(self) -> RegistryStatistics:
        """Computes current operational statistics of the registry."""
        total = len(self._tools)
        enabled = sum(1 for t in self._tools.values() if t.is_enabled)
        disabled = total - enabled
        deprecated = sum(1 for t in self._tools.values() if t.metadata.state == ToolState.DEPRECATED)
        experimental = sum(1 for reg in self._registrations.values() if reg.metadata and reg.metadata.is_experimental)

        cat_breakdown: Dict[str, int] = {}
        perm_breakdown: Dict[str, int] = {}

        for tool in self._tools.values():
            c_val = tool.category.value
            p_val = tool.permission_level.value
            cat_breakdown[c_val] = cat_breakdown.get(c_val, 0) + 1
            perm_breakdown[p_val] = perm_breakdown.get(p_val, 0) + 1

        return RegistryStatistics(
            total_registered=total,
            enabled_count=enabled,
            disabled_count=disabled,
            deprecated_count=deprecated,
            experimental_count=experimental,
            category_breakdown=cat_breakdown,
            permission_breakdown=perm_breakdown
        )

    def get_registry_health(self) -> RegistryHealth:
        """Fetches health state of the registry."""
        failed_count = sum(1 for r in self._registrations.values() if r.status != RegistrationStatus.SUCCESS)
        return RegistryHealth(
            state=self.state,
            is_healthy=failed_count == 0 and self.state == RegistryState.READY,
            loaded_tools_count=len(self._tools),
            failed_imports_count=failed_count,
            errors=[r.error_message for r in self._registrations.values() if r.error_message]
        )

    def clear(self) -> None:
        """Clears registry (for unit test isolation)."""
        self._tools.clear()
        self._registrations.clear()
        self.metadata_manager.clear()
        self.state = RegistryState.UNINITIALIZED

    def _apply_registration(self, record: ToolRegistration) -> None:
        """Internal helper inserting validated registration record."""
        tool = record.instance
        if tool:
            self._tools[tool.name] = tool
            self._registrations[tool.name] = record
            self.metadata_manager.index_tool(tool)
