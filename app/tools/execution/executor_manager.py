"""
app/tools/execution/executor_manager.py - Main Facade for Execution Subsystem
=============================================================================
Coordinates PermissionChecker, ParameterValidator, ExecutionContextFactory,
TimeoutManager, RetryManager, and ResultFormatter to execute tools safely.
"""

import time
import logging
from typing import Optional, Dict, Any
from app.tools.base_tool import BaseTool
from app.tools.registry import RegistryManager
from app.tools.execution.execution_types import ExecutionStatus, PermissionDecision, FailureReason
from app.tools.execution.execution_config import get_execution_config, ToolExecutionConfig
from app.tools.execution.execution_models import ExecutionRequest, ExecutionContext, ExecutionResult, ExecutionStatistics
from app.tools.execution.permission_checker import PermissionChecker
from app.tools.execution.parameter_validator import ParameterValidator
from app.tools.execution.execution_context import ExecutionContextFactory
from app.tools.execution.timeout_manager import TimeoutManager
from app.tools.execution.retry_manager import RetryManager
from app.tools.execution.result_formatter import ResultFormatter
from app.tools.execution.tool_executor import ToolExecutorWorker
from app.tools.tool_exceptions import ToolNotFoundException, ToolValidationException

logger = logging.getLogger("sana_ai.tools.execution.manager")


class ExecutionManager:
    """
    Facade for safe tool execution pipeline.
    
    Strict Isolation Guarantee:
    - Never communicates directly with Telegram.
    - Never calls Prompt Builder or LLM Generator.
    - Never modifies MemoryManager directly.
    """

    _instance: Optional["ExecutionManager"] = None

    def __init__(
        self,
        registry: Optional[RegistryManager] = None,
        permission_checker: Optional[PermissionChecker] = None,
        retry_manager: Optional[RetryManager] = None,
        timeout_manager: Optional[TimeoutManager] = None
    ):
        self.config = get_execution_config()
        self.registry = registry or RegistryManager.get_instance()
        self.permission_checker = permission_checker or PermissionChecker()
        self.retry_manager = retry_manager or RetryManager()
        self.timeout_manager = timeout_manager or TimeoutManager()
        self.worker = ToolExecutorWorker()
        self.stats = ExecutionStatistics()

    @classmethod
    def get_instance(cls) -> "ExecutionManager":
        """Returns ExecutionManager singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def execute_request(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Main entry point for executing a tool request.
        
        Workflow:
        1. Lookup tool in Registry.
        2. Check tool enabled status.
        3. Check permissions.
        4. Validate parameters.
        5. Create ExecutionContext.
        6. Execute with Timeout & Retry pipeline.
        7. Format & return structured ExecutionResult.
        """
        tool_name = request.tool_name
        req_id = request.request_id

        # 1. Lookup Tool
        if not self.registry.has_tool(tool_name):
            logger.error(f"Execution rejected: Tool '{tool_name}' not found in registry.")
            self._update_stats(ExecutionStatus.FAILURE)
            return ResultFormatter.format_failure(
                req_id, tool_name, ExecutionStatus.FAILURE,
                ToolNotFoundException(tool_name),
                reason=FailureReason.UNKNOWN_TOOL
            )

        tool = self.registry.get_tool(tool_name)

        # 2. Enabled status check
        if not tool.is_enabled:
            logger.warning(f"Execution rejected: Tool '{tool_name}' is disabled.")
            self._update_stats(ExecutionStatus.FAILURE)
            return ResultFormatter.format_failure(
                req_id, tool_name, ExecutionStatus.FAILURE,
                Exception(f"Tool '{tool_name}' is currently disabled."),
                reason=FailureReason.DISABLED_TOOL
            )

        # 3. Create ExecutionContext
        context = ExecutionContextFactory.create_context(request, tool)

        # 4. Check Permissions
        perm_decision, perm_reason = self.permission_checker.evaluate_permission(
            tool, user_confirmed=request.user_confirmed, context=context
        )
        if perm_decision != PermissionDecision.ALLOWED:
            logger.warning(f"Execution denied for tool '{tool_name}': {perm_reason}")
            self._update_stats(ExecutionStatus.PERMISSION_DENIED)
            return ResultFormatter.format_failure(
                req_id, tool_name, ExecutionStatus.PERMISSION_DENIED,
                Exception(perm_reason),
                reason=FailureReason.PERMISSION_DENIED
            )

        # 5. Validate Parameters
        try:
            validated_params = ParameterValidator.validate(tool, request.parameters)
        except ToolValidationException as val_err:
            logger.warning(f"Parameter validation failed for tool '{tool_name}': {val_err}")
            self._update_stats(ExecutionStatus.VALIDATION_ERROR)
            return ResultFormatter.format_failure(
                req_id, tool_name, ExecutionStatus.VALIDATION_ERROR,
                val_err,
                reason=FailureReason.PARAMETER_INVALID
            )

        # 6. Execute with Timeout & Retry pipeline
        start_time = time.perf_counter()

        def _single_execution_attempt():
            t_succeeded, t_out, t_res = self.timeout_manager.execute_with_timeout(
                func=self.worker.run_tool,
                args=(tool, validated_params, context),
                timeout_seconds=tool.metadata.timeout_seconds,
                tool_name=tool_name
            )
            if not t_succeeded:
                return False, None, Exception(f"Execution timed out after {tool.metadata.timeout_seconds}s.")
            # Unpack worker output (succeeded, raw_output, error, duration_ms)
            w_succeeded, w_out, w_err, _ = t_out
            return w_succeeded, w_out, w_err

        succeeded, raw_output, exec_err, retry_res = self.retry_manager.execute_with_retry(
            execution_func=_single_execution_attempt,
            max_retries=self.config.max_retries,
            tool_name=tool_name
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 7. Formulate Result
        if succeeded:
            self._update_stats(ExecutionStatus.SUCCESS, elapsed_ms)
            return ResultFormatter.format_success(
                req_id, tool_name, raw_output, elapsed_ms, retry_count=retry_res.total_attempts - 1
            )
        else:
            status = ExecutionStatus.TIMEOUT if "timed out" in str(exec_err) else ExecutionStatus.FAILURE
            reason = FailureReason.TIMEOUT_EXCEEDED if status == ExecutionStatus.TIMEOUT else FailureReason.UNHANDLED_EXCEPTION
            self._update_stats(status, elapsed_ms)
            return ResultFormatter.format_failure(
                req_id, tool_name, status, exec_err, elapsed_ms,
                retry_count=retry_res.total_attempts - 1, reason=reason
            )

    def get_statistics(self) -> ExecutionStatistics:
        """Returns operational telemetry statistics."""
        return self.stats

    def clear_statistics(self) -> None:
        """Resets telemetry metrics."""
        self.stats = ExecutionStatistics()

    def _update_stats(self, status: ExecutionStatus, time_ms: float = 0.0) -> None:
        """Updates internal telemetry statistics."""
        self.stats.total_executed += 1
        self.stats.total_time_ms += time_ms
        if status == ExecutionStatus.SUCCESS:
            self.stats.success_count += 1
        elif status == ExecutionStatus.TIMEOUT:
            self.stats.timeout_count += 1
        elif status == ExecutionStatus.PERMISSION_DENIED:
            self.stats.permission_denied_count += 1
        else:
            self.stats.failure_count += 1
