"""
app/tools/tool_executor.py - Core Tool Executor Integration & Compatibility Layer
================================================================================
Bridges Part 1 legacy ToolExecutor calls to Part 4 ExecutionManager pipeline.
"""

import logging
from typing import Optional
from app.tools.tool_models import ToolRequest, ToolResponse, PermissionResult, ExecutionResult as LegacyExecutionResult
from app.tools.tool_types import ExecutionStatus
from app.tools.execution.execution_types import ExecutionMode, FailureReason
from app.tools.execution.execution_models import ExecutionRequest, ExecutionResult
from app.tools.execution.executor_manager import ExecutionManager
from app.tools.registry import RegistryManager
from app.tools.permission_manager import PermissionManager
from app.tools.tool_utils import format_tool_result_for_prompt

logger = logging.getLogger("sana_ai.tools.executor")


class ToolExecutor:
    """
    Adapter bridging Part 1 ToolExecutor calls to the enterprise Part 4 ExecutionManager.
    """

    def __init__(
        self,
        registry: Optional[RegistryManager] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        self.manager = ExecutionManager.get_instance()

    def execute(self, request: ToolRequest) -> ToolResponse:
        """
        Executes ToolRequest via ExecutionManager pipeline.
        """
        exec_req = ExecutionRequest(
            tool_name=request.tool_name,
            parameters=request.parameters,
            user_id=request.user_id,
            request_id=request.request_id,
            user_confirmed=request.user_confirmed,
            priority=request.priority
        )

        exec_res: ExecutionResult = self.manager.execute_request(exec_req)

        # Convert Part 4 ExecutionResult to Part 1 ToolResponse
        perm_res = PermissionResult(
            is_allowed=exec_res.status == ExecutionStatus.SUCCESS,
            permission_level=PermissionManager().check_permission(
                self.manager.registry.get_tool(request.tool_name).metadata,
                user_confirmed=request.user_confirmed
            ).permission_level if self.manager.registry.has_tool(request.tool_name) else None,
            reason=exec_res.error_message or "Execution completed."
        ) if self.manager.registry.has_tool(request.tool_name) else None

        legacy_exec_res = LegacyExecutionResult(
            result_id=exec_res.result_id,
            request_id=exec_res.request_id,
            tool_name=exec_res.tool_name,
            status=exec_res.status,
            output=exec_res.output,
            error_message=exec_res.error_message,
            execution_time_ms=exec_res.execution_time_ms,
            retry_count=exec_res.retry_count,
            raw_exception=exec_res.raw_exception
        )

        return ToolResponse(
            request_id=exec_res.request_id,
            tool_name=exec_res.tool_name,
            status=exec_res.status,
            result=legacy_exec_res,
            permission_result=perm_res,
            formatted_output=exec_res.formatted_output
        )
