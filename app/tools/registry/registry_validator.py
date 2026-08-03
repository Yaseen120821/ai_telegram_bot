"""
app/tools/registry/registry_validator.py - Registry Validation Suite
=====================================================================
Validates tool classes, metadata, name collisions, inheritance contracts,
and schema parameters prior to insertion into the active ToolRegistry.
"""

import logging
from typing import Type, Any, Optional, Dict
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.tool_exceptions import ToolValidationException, DuplicateToolException
from app.tools.registry.registry_utils import is_basetool_subclass, parse_semver

logger = logging.getLogger("sana_ai.tools.registry.validator")


class RegistryValidator:
    """Performs rigorous pre-registration checks on candidate tools."""

    @staticmethod
    def validate_tool_class(tool_cls: Type[BaseTool]) -> None:
        """Verifies that candidate class cleanly inherits from BaseTool and is not abstract."""
        if not is_basetool_subclass(tool_cls):
            raise ToolValidationException(
                f"Class '{getattr(tool_cls, '__name__', str(tool_cls))}' must inherit from BaseTool and implement abstract methods."
            )

    @staticmethod
    def validate_tool_instance(tool: BaseTool, existing_names: Dict[str, Any]) -> None:
        """
        Validates an instantiated tool's metadata, name uniqueness, category, and parameters.
        """
        if not isinstance(tool, BaseTool):
            raise ToolValidationException(f"Object of type '{type(tool).__name__}' is not an instance of BaseTool.")

        # 1. Unique Name Check
        tool_name = tool.name
        if not tool_name or not tool_name.strip():
            raise ToolValidationException("Tool name cannot be empty or whitespace.")

        if tool_name in existing_names:
            logger.error(f"Duplicate registration attempt detected for tool '{tool_name}'.")
            raise DuplicateToolException(tool_name)

        # 2. Description completeness
        if not tool.description or not tool.description.strip():
            raise ToolValidationException(f"Tool '{tool_name}' must have a non-empty description.")

        if len(tool.description.strip()) < 10:
            raise ToolValidationException(f"Tool '{tool_name}' description is too short (min 10 characters).")

        # 3. Category validation
        if not isinstance(tool.category, ToolCategory):
            raise ToolValidationException(f"Tool '{tool_name}' category '{tool.category}' is invalid.")

        # 4. Permission level validation
        if not isinstance(tool.permission_level, PermissionLevel):
            raise ToolValidationException(f"Tool '{tool_name}' permission level '{tool.permission_level}' is invalid.")

        # 5. Schema validation
        schema = tool.parameters_schema
        if not isinstance(schema, dict):
            raise ToolValidationException(f"Tool '{tool_name}' parameters_schema must be a dictionary.")

        # 6. Version string format
        semver = parse_semver(tool.metadata.version)
        if semver == (0, 0, 0):
            logger.warning(f"Tool '{tool_name}' has non-standard version format: '{tool.metadata.version}'.")
