"""
app/tools/execution/result_formatter.py - Result Normalization & Prompt Formatter
=================================================================================
Formats raw Python outputs into standardized ExecutionResult instances and clean
text representations formatted specifically for PromptBuilder / LLM generator consumption.
"""

import json
import logging
from typing import Any, Optional
from app.tools.execution.execution_models import ExecutionResult
from app.tools.execution.execution_types import ExecutionStatus, FailureReason
from app.tools.execution.execution_utils import sanitize_error_message

logger = logging.getLogger("sana_ai.tools.execution.formatter")


class ResultFormatter:
    """Formats raw tool outputs into clean, unified ExecutionResult objects."""

    @staticmethod
    def format_success(
        request_id: str,
        tool_name: str,
        raw_output: Any,
        duration_ms: float,
        retry_count: int = 0
    ) -> ExecutionResult:
        """Constructs ExecutionResult for successful tool execution."""
        formatted_str = ResultFormatter.normalize_output_string(raw_output)
        prompt_block = ResultFormatter.build_prompt_block(tool_name, "SUCCESS", formatted_str)

        return ExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ExecutionStatus.SUCCESS,
            output=raw_output,
            execution_time_ms=duration_ms,
            retry_count=retry_count,
            formatted_output=prompt_block,
            failure_reason=FailureReason.NONE
        )

    @staticmethod
    def format_failure(
        request_id: str,
        tool_name: str,
        status: ExecutionStatus,
        error: Optional[Exception],
        duration_ms: float = 0.0,
        retry_count: int = 0,
        reason: FailureReason = FailureReason.UNHANDLED_EXCEPTION
    ) -> ExecutionResult:
        """Constructs ExecutionResult for failed or rejected tool execution."""
        err_msg = sanitize_error_message(error) if error else "Execution rejected."
        prompt_block = ResultFormatter.build_prompt_block(tool_name, status.value.upper(), f"Error: {err_msg}")

        return ExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=status,
            error_message=err_msg,
            execution_time_ms=duration_ms,
            retry_count=retry_count,
            formatted_output=prompt_block,
            failure_reason=reason,
            raw_exception=error
        )

    @staticmethod
    def normalize_output_string(raw_output: Any) -> str:
        """Normalizes arbitrary python data structures into clean JSON or string text."""
        if raw_output is None:
            return "Task completed cleanly with no return value."
        if isinstance(raw_output, str):
            return raw_output.strip()
        if isinstance(raw_output, (dict, list)):
            try:
                return json.dumps(raw_output, indent=2, ensure_ascii=False)
            except Exception:
                return str(raw_output)
        return str(raw_output)

    @staticmethod
    def build_prompt_block(tool_name: str, status_str: str, content: str) -> str:
        """Constructs standard prompt block for PromptBuilder injection."""
        return (
            f"[TOOL EXECUTION RESULT]\n"
            f"Tool Name: {tool_name}\n"
            f"Status: {status_str}\n"
            f"Output:\n{content}\n"
            f"[END TOOL RESULT]"
        )
