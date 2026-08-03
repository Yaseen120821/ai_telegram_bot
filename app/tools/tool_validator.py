"""
app/tools/tool_validator.py - Parameter & Schema Validation for Tool Framework
=============================================================================
Validates tool requests, parameters, schema specifications, and metadata
to ensure strict type safety and parameter correctness prior to tool execution.
"""

import logging
from typing import Dict, Any, List, Tuple
from app.tools.tool_models import ToolMetadata, ToolRequest
from app.tools.tool_exceptions import ToolValidationException

logger = logging.getLogger("sana_ai.tools.validator")


class ToolValidator:
    """Validates tool definitions, invocation requests, and runtime parameters."""

    @staticmethod
    def validate_tool_metadata(metadata: ToolMetadata) -> None:
        """Validates tool registration metadata."""
        if not metadata.name or not metadata.name.strip():
            raise ToolValidationException("Tool name cannot be empty.")
        if not metadata.description or not metadata.description.strip():
            raise ToolValidationException(f"Tool '{metadata.name}' must have a non-empty description.")
        if not isinstance(metadata.parameters_schema, dict):
            raise ToolValidationException(f"Tool '{metadata.name}' parameters schema must be a dictionary.")

    @staticmethod
    def validate_request(request: ToolRequest) -> None:
        """Validates a ToolRequest instance structure."""
        if not request.tool_name or not request.tool_name.strip():
            raise ToolValidationException("ToolRequest must specify a valid 'tool_name'.")
        if not isinstance(request.parameters, dict):
            raise ToolValidationException("ToolRequest 'parameters' must be a dictionary.", tool_name=request.tool_name)

    @staticmethod
    def validate_parameters(tool_name: str, parameters: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates supplied parameter dictionary against tool schema requirements.
        Checks required fields and type constraints.
        Returns normalized parameters dictionary.
        """
        if not schema:
            return parameters

        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # 1. Check for missing required parameters
        missing = [field for field in required_fields if field not in parameters]
        if missing:
            raise ToolValidationException(
                f"Missing required parameter(s) for tool '{tool_name}': {', '.join(missing)}",
                tool_name=tool_name,
                errors={"missing_parameters": missing}
            )

        # 2. Type validation
        type_errors = {}
        for param_name, param_value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type:
                    valid = ToolValidator._check_param_type(param_value, expected_type)
                    if not valid:
                        type_errors[param_name] = f"Expected type '{expected_type}', got '{type(param_value).__name__}'"

        if type_errors:
            raise ToolValidationException(
                f"Parameter type validation failed for tool '{tool_name}'.",
                tool_name=tool_name,
                errors=type_errors
            )

        return parameters

    @staticmethod
    def _check_param_type(value: Any, expected_type: str) -> bool:
        """Helper to match JSON schema primitive types with Python types."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type in ("number", "float"):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type in ("integer", "int"):
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type in ("boolean", "bool"):
            return isinstance(value, bool)
        elif expected_type in ("array", "list"):
            return isinstance(value, list)
        elif expected_type in ("object", "dict"):
            return isinstance(value, dict)
        return True
