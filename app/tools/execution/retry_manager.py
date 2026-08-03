"""
app/tools/execution/retry_manager.py - Exponential Backoff & Retry Pipeline
===========================================================================
Manages automatic retries for transient tool execution failures, distinguishing
retryable errors from fatal permission/validation failures.
"""

import time
import logging
from typing import Callable, Any, Optional, Tuple
from app.tools.execution.execution_models import RetryResult
from app.tools.execution.execution_types import ExecutionStatus, RetryPolicy
from app.tools.execution.execution_config import get_execution_config
from app.tools.tool_exceptions import ToolValidationException, PermissionDeniedException

logger = logging.getLogger("sana_ai.tools.execution.retry")


class RetryManager:
    """Manages execution retries with exponential backoff."""

    def __init__(self):
        self.config = get_execution_config()

    def is_retryable(self, exception: Exception) -> bool:
        """Determines if an exception is transient and candidate for retrying."""
        if isinstance(exception, (ToolValidationException, PermissionDeniedException)):
            return False
        return True

    def execute_with_retry(
        self,
        execution_func: Callable[[], Tuple[bool, Any, Optional[Exception]]],
        max_retries: Optional[int] = None,
        tool_name: str = "unknown_tool"
    ) -> Tuple[bool, Any, Optional[Exception], RetryResult]:
        """
        Executes a callable, retrying up to max_retries on transient failure.
        """
        limit = max_retries if max_retries is not None else self.config.max_retries
        delay = self.config.initial_retry_delay_seconds
        total_delay_ms = 0.0

        for attempt in range(limit + 1):
            succeeded, output, error = execution_func()
            if succeeded:
                return True, output, None, RetryResult(
                    total_attempts=attempt + 1,
                    final_status=ExecutionStatus.SUCCESS,
                    total_delay_ms=total_delay_ms,
                    succeeded=True
                )

            # Check if error is non-retryable or if we reached maximum retries
            if error and not self.is_retryable(error):
                logger.info(f"Non-retryable error encountered for tool '{tool_name}': {error}. Aborting retries.")
                return False, None, error, RetryResult(
                    total_attempts=attempt + 1,
                    final_status=ExecutionStatus.FAILURE,
                    total_delay_ms=total_delay_ms,
                    succeeded=False
                )

            if attempt < limit:
                logger.warning(f"Attempt {attempt + 1} for tool '{tool_name}' failed ({error}). Retrying in {delay:.2f}s...")
                time.sleep(delay)
                total_delay_ms += (delay * 1000.0)
                if self.config.retry_policy == RetryPolicy.EXPONENTIAL_BACKOFF:
                    delay *= 2.0

        return False, None, error, RetryResult(
            total_attempts=limit + 1,
            final_status=ExecutionStatus.FAILURE,
            total_delay_ms=total_delay_ms,
            succeeded=False
        )
