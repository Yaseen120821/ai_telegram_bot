"""
app/tools/base_tool.py - Abstract Base Tool Interface
===================================================
Defines the foundational abstract BaseTool class for SANA AI's Tool Calling Framework.
All executable tools in the system must inherit from BaseTool and implement its contract.
"""

import abc
import logging
from typing import Dict, Any, Optional, List
from app.tools.tool_types import ToolCategory, PermissionLevel, ToolState, ToolPriority
from app.tools.tool_models import ToolMetadata, ExecutionContext, ExecutionResult
from app.tools.tool_utils import measure_execution_time, normalize_tool_output
from app.tools.tool_exceptions import ToolValidationException, ToolExecutionException


class BaseTool(abc.ABC):
    """
    Abstract Base Class for all SANA AI Tools.
    
    Enforces interface-based design, providing standardized metadata, validation,
    execution wrapper, timeout handling, and formatted results across all categories.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        permission_level: PermissionLevel = PermissionLevel.SAFE,
        version: str = "1.0.0",
        parameters_schema: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 10.0,
        priority: ToolPriority = ToolPriority.NORMAL,
        tags: Optional[List[str]] = None
    ):
        from typing import List
        self.metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            permission_level=permission_level,
            state=ToolState.ENABLED,
            priority=priority,
            version=version,
            parameters_schema=parameters_schema or {"type": "object", "properties": {}},
            timeout_seconds=timeout_seconds,
            tags=tags or []
        )
        self.logger = logging.getLogger(f"sana_ai.tools.{category.value}.{name}")

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Human and LLM readable summary of what the tool does."""
        return self.metadata.description

    @property
    def category(self) -> ToolCategory:
        """Tool category taxonomy."""
        return self.metadata.category

    @property
    def permission_level(self) -> PermissionLevel:
        """Security authorization level."""
        return self.metadata.permission_level

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema defining arguments expected by this tool."""
        return self.metadata.parameters_schema

    @property
    def is_enabled(self) -> bool:
        """Checks if the tool is currently enabled."""
        return self.metadata.state == ToolState.ENABLED

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates input arguments against schema before execution.
        Subclasses can override to add custom parameter checks.
        """
        from app.tools.tool_validator import ToolValidator
        return ToolValidator.validate_parameters(self.name, parameters, self.parameters_schema)

    def execute_wrapper(self, parameters: Dict[str, Any], context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        Public wrapper enforcing validation, timing, error handling, and standard result format.
        Delegates underlying action to `_run()`.
        """
        from app.tools.tool_types import ExecutionStatus
        
        # 1. Execute concrete business logic with timing & error handling
        with measure_execution_time() as timer:
            try:
                validated_params = self.validate_parameters(parameters)
                raw_output = self._run(validated_params, context)
                status = ExecutionStatus.SUCCESS
                error_msg = None
                raw_exc = None
            except ToolValidationException as val_exc:
                self.logger.warning(f"Validation error in tool '{self.name}': {val_exc}")
                raw_output = None
                status = ExecutionStatus.VALIDATION_ERROR
                error_msg = str(val_exc)
                raw_exc = val_exc
            except Exception as e:
                self.logger.error(f"Execution failed in tool '{self.name}': {e}", exc_info=True)
                raw_output = None
                status = ExecutionStatus.FAILURE
                error_msg = str(e)
                raw_exc = e

        formatted = self.format_result(raw_output) if status == ExecutionStatus.SUCCESS else error_msg

        return ExecutionResult(
            tool_name=self.name,
            status=status,
            output=formatted,
            error_message=error_msg,
            execution_time_ms=timer["elapsed_ms"],
            raw_exception=raw_exc
        )

    @abc.abstractmethod
    def _run(self, parameters: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        """
        Abstract core execution method.
        Subclasses MUST implement specific business logic here.
        """
        pass

    def format_result(self, raw_result: Any) -> str:
        """
        Formats raw python outputs into a string representation for prompt injection.
        """
        return normalize_tool_output(raw_result)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes tool metadata to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "permission_level": self.permission_level.value,
            "state": self.metadata.state.value,
            "version": self.metadata.version,
            "parameters_schema": self.parameters_schema,
            "timeout_seconds": self.metadata.timeout_seconds
        }
