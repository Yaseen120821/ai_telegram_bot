"""
app/tools/tool_registry.py - Centralized Tool Registration & Discovery Repository
=================================================================================
Provides dynamic registration, discovery, enabling/disabling, metadata inspection,
and schema generation for all tools within the SANA AI environment.
"""

import logging
from typing import Dict, List, Optional, Any
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, ToolState
from app.tools.tool_models import ToolMetadata, ToolStatistics
from app.tools.tool_validator import ToolValidator
from app.tools.tool_exceptions import DuplicateToolException, ToolNotFoundException

logger = logging.getLogger("sana_ai.tools.registry")


class ToolRegistry:
    """
    Central registry maintaining active tool singletons, lookup indexes,
    category mappings, and usage statistics.
    """

    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._stats: Dict[str, ToolStatistics] = {}

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Singleton instance getter."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """
        Registers a tool into the registry.
        Validates metadata and ensures no duplicate name collisions exist.
        """
        from app.tools.registry import RegistryManager
        ToolValidator.validate_tool_metadata(tool.metadata)

        tool_name = tool.name
        if tool_name in self._tools:
            logger.error(f"Duplicate tool registration attempted for '{tool_name}'.")
            raise DuplicateToolException(tool_name)

        self._tools[tool_name] = tool
        self._stats[tool_name] = ToolStatistics(tool_name=tool_name)
        
        # Sync with RegistryManager singleton
        reg_mgr = RegistryManager.get_instance()
        if not reg_mgr.has_tool(tool_name):
            reg_mgr.register_tool(tool)

        logger.info(f"Registered tool '{tool_name}' [Category: {tool.category.value}, Perm: {tool.permission_level.value}]")

    def unregister(self, tool_name: str) -> None:
        """Removes a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            if tool_name in self._stats:
                del self._stats[tool_name]
            logger.info(f"Unregistered tool '{tool_name}'.")
        else:
            raise ToolNotFoundException(tool_name)

    def get_tool(self, tool_name: str) -> BaseTool:
        """Retrieves a registered tool by name."""
        if tool_name not in self._tools:
            raise ToolNotFoundException(tool_name)
        return self._tools[tool_name]

    def has_tool(self, tool_name: str) -> bool:
        """Checks if a tool is registered."""
        return tool_name in self._tools

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        only_enabled: bool = True
    ) -> List[BaseTool]:
        """Returns list of registered tools, optionally filtered by category or state."""
        result = []
        for tool in self._tools.values():
            if only_enabled and not tool.is_enabled:
                continue
            if category is not None and tool.category != category:
                continue
            result.append(tool)
        return result

    def enable_tool(self, tool_name: str) -> None:
        """Enables a disabled tool."""
        tool = self.get_tool(tool_name)
        tool.metadata.state = ToolState.ENABLED
        logger.info(f"Enabled tool '{tool_name}'.")

    def disable_tool(self, tool_name: str) -> None:
        """Disables a tool from being executed."""
        tool = self.get_tool(tool_name)
        tool.metadata.state = ToolState.DISABLED
        logger.info(f"Disabled tool '{tool_name}'.")

    def get_tool_statistics(self, tool_name: str) -> ToolStatistics:
        """Fetches telemetry statistics for a specific tool."""
        if tool_name not in self._stats:
            raise ToolNotFoundException(tool_name)
        return self._stats[tool_name]

    def get_all_tool_schemas(self, only_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        Generates schema representations for all active tools suitable for LLM system prompt inclusion.
        """
        schemas = []
        for tool in self.list_tools(only_enabled=only_enabled):
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema
            })
        return schemas

    def clear(self) -> None:
        """Clears all registered tools (used primarily for test cleanup)."""
        from app.tools.registry import RegistryManager
        self._tools.clear()
        self._stats.clear()
        RegistryManager.get_instance().clear()
