"""
app/tools/tool_models.py - Data Models for Tool Calling Framework
=================================================================
Defines strongly-typed dataclasses for tool requests, responses, metadata,
execution contexts, permission decisions, and operational statistics.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.tools.tool_types import ToolCategory, PermissionLevel, ExecutionStatus, ToolState, ToolPriority


@dataclass
class ToolMetadata:
    """Metadata describing tool properties, capabilities, and schema."""
    name: str
    description: str
    category: ToolCategory
    permission_level: PermissionLevel = PermissionLevel.SAFE
    state: ToolState = ToolState.ENABLED
    priority: ToolPriority = ToolPriority.NORMAL
    version: str = "1.0.0"
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timeout_seconds: float = 10.0


@dataclass
class ExecutionContext:
    """Runtime context passed down during tool execution."""
    user_id: str
    session_id: str = ""
    message_id: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    user_permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionResult:
    """Result of security permission check performed by PermissionManager."""
    is_allowed: bool
    permission_level: PermissionLevel
    requires_confirmation: bool = False
    user_confirmed: bool = False
    reason: str = "Permission check passed."


@dataclass
class ToolRequest:
    """Structured request payload issued to invoke a tool."""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "default_user"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Optional[ExecutionContext] = None
    priority: ToolPriority = ToolPriority.NORMAL
    user_confirmed: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionResult:
    """Raw result produced by executing a tool function."""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    tool_name: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    retry_count: int = 0
    raw_exception: Optional[Exception] = None


@dataclass
class ToolResponse:
    """Unified response container returned to the system caller."""
    request_id: str
    tool_name: str
    status: ExecutionStatus
    result: Optional[ExecutionResult] = None
    permission_result: Optional[PermissionResult] = None
    formatted_output: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolStatistics:
    """Performance and telemetry metrics for a tool."""
    tool_name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time_ms: float = 0.0
    last_called_at: Optional[float] = None

    @property
    def avg_execution_time_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.call_count

    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count
