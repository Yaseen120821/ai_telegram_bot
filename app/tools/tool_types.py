"""
app/tools/tool_types.py - Enumeration Types for SANA AI Tool Calling Framework
=============================================================================
Defines standard status, category, permission, state, and priority enums
used across all components of the Tool Calling Framework.
"""

from enum import Enum, auto


class ToolCategory(str, Enum):
    """Categories classifying tools for discovery, authorization, and modular grouping."""
    SYSTEM = "system"
    FILESYSTEM = "filesystem"
    INTERNET = "internet"
    PRODUCTIVITY = "productivity"
    DEVELOPER = "developer"
    UTILITY = "utility"
    AI = "ai"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    CUSTOM = "custom"


class PermissionLevel(str, Enum):
    """Security permission levels defining authorization requirements for tool execution."""
    SAFE = "safe"                          # Read-only or completely safe tools (no side effects)
    CONFIRMATION_REQUIRED = "confirmation" # Requires explicit user confirmation before execution
    RESTRICTED = "restricted"             # Restricted actions requiring elevated privileges
    ADMINISTRATOR = "administrator"       # Critical system administration capabilities


class ExecutionStatus(str, Enum):
    """Execution state machine statuses for tracking tool execution lifecycles."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION_ERROR = "validation_error"
    CANCELLED = "cancelled"


class ToolState(str, Enum):
    """Lifecycle state of registered tools."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"


class ToolPriority(str, Enum):
    """Priority level for tool selection and routing ranking."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
