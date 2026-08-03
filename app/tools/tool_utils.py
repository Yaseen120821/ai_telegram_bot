"""
app/tools/tool_utils.py - Utility Helpers for Tool Calling Framework
===================================================================
Provides helper utilities for UUID generation, timing measurements, output
formatting, payload normalization, and metric logging.
"""

import time
import uuid
import json
import logging
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger("sana_ai.tools.utils")


def generate_unique_id(prefix: str = "req") -> str:
    """Generates a prefixed unique identifier string."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def measure_execution_time():
    """Context manager measuring execution time in milliseconds."""
    start_time = time.perf_counter()
    metrics = {"elapsed_ms": 0.0}
    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        metrics["elapsed_ms"] = (end_time - start_time) * 1000.0


def normalize_tool_output(raw_output: Any) -> str:
    """Normalizes arbitrary tool return values into clean, predictable string representation."""
    if raw_output is None:
        return "Task completed with no return value."
    if isinstance(raw_output, str):
        return raw_output.strip()
    if isinstance(raw_output, (dict, list)):
        try:
            return json.dumps(raw_output, indent=2, ensure_ascii=False)
        except Exception:
            return str(raw_output)
    return str(raw_output)


def truncate_output(text: str, max_length: int = 2000) -> str:
    """Truncates output string if it exceeds maximum character threshold."""
    if len(text) <= max_length:
        return text
    truncated_len = len(text) - max_length
    return text[:max_length] + f"\n... [Truncated {truncated_len} characters]"


def format_tool_result_for_prompt(tool_name: str, status: str, output: str) -> str:
    """Formats tool execution output specifically for consumption by PromptBuilder / LLM generator."""
    return (
        f"[TOOL EXECUTION RESULT]\n"
        f"Tool Name: {tool_name}\n"
        f"Status: {status.upper()}\n"
        f"Output:\n{output}\n"
        f"[END TOOL RESULT]"
    )


def log_tool_metrics(tool_name: str, execution_time_ms: float, status: str, error: Optional[str] = None) -> None:
    """Logs tool performance telemetry for monitoring and diagnostic tracking."""
    if status.lower() == "success":
        logger.info(f"📊 [METRIC] Tool '{tool_name}' | Status: SUCCESS | Time: {execution_time_ms:.2f}ms")
    else:
        logger.warning(f"📊 [METRIC] Tool '{tool_name}' | Status: {status.upper()} | Time: {execution_time_ms:.2f}ms | Error: {error}")
