"""
app/tools/tool_exceptions.py - Custom Exceptions for Tool Calling Framework
===========================================================================
Defines specific, descriptive exception classes for domain-specific errors
within the SANA AI tool execution pipeline.
"""


from typing import Optional

class ToolException(Exception):
    """Base exception class for all Tool Framework errors."""
    def __init__(self, message: str, tool_name: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}


class ToolNotFoundException(ToolException):
    """Raised when a requested tool is not registered in the ToolRegistry."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' was not found in the registry.", tool_name=tool_name)


class DuplicateToolException(ToolException):
    """Raised when attempting to register a tool with an already registered name."""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is already registered.", tool_name=tool_name)


class ToolValidationException(ToolException):
    """Raised when parameter validation or schema validation fails."""
    def __init__(self, message: str, tool_name: str = None, errors: dict = None):
        super().__init__(message, tool_name=tool_name, details=errors)
        self.errors = errors or {}


class PermissionDeniedException(ToolException):
    """Raised when tool execution is denied by PermissionManager."""
    def __init__(self, message: str, tool_name: str = "unknown", permission_level: str = "restricted", reason: str = None):
        msg = message or f"Permission denied for tool '{tool_name}' (Level: {permission_level})."
        if reason and reason not in msg:
            msg += f" Reason: {reason}"
        super().__init__(msg, tool_name=tool_name, details={"permission_level": permission_level, "reason": reason})
        self.permission_level = permission_level
        self.reason = reason


class ToolTimeoutException(ToolException):
    """Raised when tool execution exceeds the specified timeout threshold."""
    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(
            f"Execution of tool '{tool_name}' timed out after {timeout_seconds} seconds.",
            tool_name=tool_name,
            details={"timeout_seconds": timeout_seconds}
        )
        self.timeout_seconds = timeout_seconds


class ToolExecutionException(ToolException):
    """Raised when an unhandled runtime error occurs during tool execution."""
    def __init__(self, message: str, tool_name: str = "unknown", original_exception: Optional[Exception] = None):
        msg = message or (f"Error executing tool '{tool_name}': {str(original_exception)}" if original_exception else "Tool execution error")
        super().__init__(
            msg,
            tool_name=tool_name,
            details={"original_error": str(original_exception) if original_exception else msg, "error_type": type(original_exception).__name__ if original_exception else "ToolExecutionException"}
        )
        self.original_exception = original_exception


class ToolDisabledException(ToolException):
    """Raised when attempting to execute a tool that is currently disabled."""
    def __init__(self, tool_name: str, state: str = "disabled"):
        super().__init__(
            f"Tool '{tool_name}' is currently {state} and cannot be executed.",
            tool_name=tool_name,
            details={"state": state}
        )
