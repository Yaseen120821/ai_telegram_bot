"""
app/tools/execution/execution_models.py - Data Models for Execution Pipeline
=============================================================================
Defines strongly-typed dataclasses for execution requests, correlation contexts,
execution results, retry/timeout records, errors, and performance telemetry.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.tools.tool_types import PermissionLevel, ToolPriority
from app.tools.execution.execution_types import ExecutionMode, ExecutionStatus, FailureReason


@dataclass
class ExecutionRequest:
    """Incoming request payload instructing ExecutionManager to run a tool."""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "default_user"
    conversation_id: str = "default_session"
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    user_confirmed: bool = False
    priority: ToolPriority = ToolPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Comprehensive correlation context passed down during tool execution."""
    execution_id: str = field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    correlation_id: str = field(default_factory=lambda: f"corr_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    user_id: str = "default_user"
    conversation_id: str = "default_session"
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.SAFE
    mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Standardized output produced by ToolExecutor and returned to PromptBuilder/Caller."""
    result_id: str = field(default_factory=lambda: f"res_{uuid.uuid4().hex[:12]}")
    request_id: str = ""
    tool_name: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    retry_count: int = 0
    formatted_output: str = ""
    failure_reason: FailureReason = FailureReason.NONE
    raw_exception: Optional[Exception] = None


@dataclass
class ExecutionError:
    """Structured container detailing execution runtime errors."""
    error_type: str
    message: str
    tool_name: str
    trace: str = ""
    is_retryable: bool = False


@dataclass
class RetryResult:
    """Audit record for retry loop attempts."""
    total_attempts: int = 0
    final_status: ExecutionStatus = ExecutionStatus.PENDING
    total_delay_ms: float = 0.0
    succeeded: bool = False


@dataclass
class TimeoutResult:
    """Audit record for timeout enforcement."""
    tool_name: str
    timeout_seconds: float
    elapsed_ms: float
    aborted: bool = True


@dataclass
class ExecutionStatistics:
    """Operational telemetry breakdown of executed tools."""
    total_executed: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    permission_denied_count: int = 0
    total_time_ms: float = 0.0

    @property
    def avg_time_ms(self) -> float:
        if self.total_executed == 0:
            return 0.0
        return self.total_time_ms / self.total_executed
