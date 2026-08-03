"""
app/tools/execution/tool_executor.py - Single Tool Execution Worker
====================================================================
Isolated worker executing concrete BaseTool instances and measuring runtime duration.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from app.tools.base_tool import BaseTool
from app.tools.execution.execution_models import ExecutionContext
from app.tools.execution.execution_utils import measure_execution_duration

logger = logging.getLogger("sana_ai.tools.execution.worker")


class ToolExecutorWorker:
    """
    Isolated execution worker executing `tool._run()`.
    
    Strict Isolation Guarantee:
    - Never modifies MemoryManager.
    - Never modifies ConversationManager.
    - Never builds prompt templates.
    """

    def run_tool(
        self,
        tool: BaseTool,
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None
    ) -> Tuple[bool, Any, Optional[Exception], float]:
        """
        Executes concrete tool logic.
        Returns Tuple[succeeded: bool, raw_output: Any, error: Optional[Exception], duration_ms: float].
        """
        with measure_execution_duration() as timer:
            try:
                logger.info(f"Worker executing tool '{tool.name}'...")
                # Note: Parameters have already been validated by ParameterValidator
                raw_output = tool._run(parameters, context)
                logger.info(f"Worker finished tool '{tool.name}' successfully in {timer['duration_ms']:.2f}ms.")
                return True, raw_output, None, timer["duration_ms"]
            except Exception as exc:
                logger.error(f"Worker tool execution failed for '{tool.name}': {exc}", exc_info=True)
                return False, None, exc, timer["duration_ms"]
