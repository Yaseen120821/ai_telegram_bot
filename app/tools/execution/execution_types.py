"""
app/tools/execution/execution_types.py - Enumeration Types for Execution Subsystem
===================================================================================
Defines standard enums for execution modes, statuses, permission decisions,
retry strategies, timeout policies, and failure classifications.
"""

from enum import Enum


class ExecutionMode(str, Enum):
    """Operational mode for tool execution."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SIMULATION = "simulation"
    BATCH = "batch"


class ExecutionStatus(str, Enum):
    """Lifecycle state machine for tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    VALIDATION_ERROR = "validation_error"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class PermissionDecision(str, Enum):
    """Detailed categorization of security permission decisions."""
    ALLOWED = "allowed"
    DENIED_CONFIRMATION_REQUIRED = "denied_confirmation_required"
    DENIED_INSUFFICIENT_ROLE = "denied_insufficient_role"
    DENIED_BLACK_LISTED = "denied_blacklisted"


class RetryPolicy(str, Enum):
    """Strategy for retrying transient tool execution failures."""
    NONE = "none"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"


class TimeoutPolicy(str, Enum):
    """Handling strategy when tool execution time exceeds threshold."""
    STRICT_ABORT = "strict_abort"
    LOG_WARNING_ONLY = "log_warning_only"
    CANCEL_AND_FALLBACK = "cancel_and_fallback"


class FailureReason(str, Enum):
    """Detailed error cause classification."""
    NONE = "none"
    UNKNOWN_TOOL = "unknown_tool"
    DISABLED_TOOL = "disabled_tool"
    PARAMETER_INVALID = "parameter_invalid"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    UNHANDLED_EXCEPTION = "unhandled_exception"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
