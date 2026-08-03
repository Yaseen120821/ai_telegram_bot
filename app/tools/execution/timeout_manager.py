"""
app/tools/execution/timeout_manager.py - Thread Timeout Enforcer
================================================================
Enforces strict per-tool execution timeouts via ThreadPoolExecutor workers,
preventing slow network calls or infinite loops from blocking the application.
"""

import time
import logging
import concurrent.futures
from typing import Callable, Any, Optional, Tuple
from app.tools.execution.execution_models import TimeoutResult, ExecutionResult
from app.tools.execution.execution_types import ExecutionStatus, FailureReason
from app.tools.execution.execution_config import get_execution_config

logger = logging.getLogger("sana_ai.tools.execution.timeout")


class TimeoutManager:
    """Enforces per-tool execution timeouts."""

    def __init__(self):
        self.config = get_execution_config()

    def execute_with_timeout(
        self,
        func: Callable[..., Any],
        args: tuple = (),
        kwargs: dict = None,
        timeout_seconds: Optional[float] = None,
        tool_name: str = "unknown_tool"
    ) -> Tuple[bool, Any, Optional[TimeoutResult]]:
        """
        Executes a callable inside a thread worker with a strict timeout limit.
        Returns Tuple[succeeded: bool, return_value: Any, timeout_result: Optional[TimeoutResult]].
        """
        kwargs = kwargs or {}
        limit = timeout_seconds if timeout_seconds is not None else self.config.default_timeout_seconds
        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=limit)
                return True, result, None
            except concurrent.futures.TimeoutError:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(f"Execution of tool '{tool_name}' timed out after {limit:.2f}s.")
                t_result = TimeoutResult(
                    tool_name=tool_name,
                    timeout_seconds=limit,
                    elapsed_ms=elapsed_ms,
                    aborted=True
                )
                return False, None, t_result
