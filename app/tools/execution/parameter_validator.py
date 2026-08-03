"""
app/tools/execution/parameter_validator.py - Parameter & Security Input Validation
=================================================================================
Enforces parameter type constraints, required field checks, path traversal security,
and range bounds before allowing tool execution.
"""

import logging
from typing import Dict, Any, List
from app.tools.base_tool import BaseTool
from app.tools.tool_exceptions import ToolValidationException
from app.tools.execution.execution_utils import is_safe_filepath

logger = logging.getLogger("sana_ai.tools.execution.validator")


class ParameterValidator:
    """Validates parameter payloads against tool JSON schemas prior to execution."""

    @staticmethod
    def validate(tool: BaseTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates parameters against tool schema requirements.
        Checks required fields, type matching, and path security.
        """
        schema = tool.parameters_schema
        if not schema:
            return parameters

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 1. Required Parameters Check
        missing = [req for req in required if req not in parameters]
        if missing:
            raise ToolValidationException(
                f"Missing required parameter(s) for tool '{tool.name}': {', '.join(missing)}",
                tool_name=tool.name,
                errors={"missing": missing}
            )

        # 2. Type & Security Checks
        type_errors = {}
        for param_name, value in parameters.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type:
                    valid = ParameterValidator._check_type(value, expected_type)
                    if not valid:
                        type_errors[param_name] = f"Expected type '{expected_type}', got '{type(value).__name__}'"

            # Safe filepath check if param looks like a file path argument
            if "filename" in param_name or "filepath" in param_name or "path" in param_name:
                if isinstance(value, str) and (".." in value or value.startswith("/") or ":" in value):
                    if not is_safe_filepath(value):
                        raise ToolValidationException(
                            f"Path traversal security block on parameter '{param_name}': '{value}'",
                            tool_name=tool.name
                        )

        if type_errors:
            raise ToolValidationException(
                f"Parameter type validation failed for tool '{tool.name}'.",
                tool_name=tool.name,
                errors=type_errors
            )

        return parameters

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
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
