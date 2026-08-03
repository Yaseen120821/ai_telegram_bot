"""
app/tools/execution/execution_context.py - Execution Context Factory
===================================================================
Constructs correlation execution contexts containing unique execution IDs,
timestamps, user identifiers, session context, and tool metadata for debugging.
"""

import time
import logging
from typing import Dict, Any, Optional
from app.tools.base_tool import BaseTool
from app.tools.execution.execution_models import ExecutionContext, ExecutionRequest
from app.tools.execution.execution_types import ExecutionMode
from app.tools.execution.execution_utils import generate_execution_id

logger = logging.getLogger("sana_ai.tools.execution.context")


class ExecutionContextFactory:
    """Factory creating correlation contexts for tool execution tracking."""

    @staticmethod
    def create_context(
        request: ExecutionRequest,
        tool: BaseTool,
        mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    ) -> ExecutionContext:
        """Constructs a fully populated ExecutionContext."""
        exec_id = generate_execution_id("exec")
        corr_id = generate_execution_id("corr")

        ctx = ExecutionContext(
            execution_id=exec_id,
            correlation_id=corr_id,
            timestamp=time.time(),
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            tool_name=tool.name,
            parameters=request.parameters,
            permission_level=tool.permission_level,
            mode=mode,
            env_vars={"user_permissions": ",".join(request.metadata.get("user_permissions", []))}
        )

        logger.debug(f"Created ExecutionContext [{exec_id} | Corr: {corr_id}] for tool '{tool.name}'.")
        return ctx
