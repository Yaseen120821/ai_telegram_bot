"""
app/tools/execution/execution_utils.py - Timing & Security Utilities
===================================================================
Provides high-precision timing contexts, ID generators, stack trace cleaning,
and path traversal security checking for safe tool execution.
"""

import os
import time
import uuid
import logging
from typing import Any, Dict
from contextlib import contextmanager

logger = logging.getLogger("sana_ai.tools.execution.utils")


def generate_execution_id(prefix: str = "exec") -> str:
    """Generates a prefixed unique correlation identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@contextmanager
def measure_execution_duration():
    """Context manager tracking execution duration in milliseconds."""
    start = time.perf_counter()
    metrics = {"duration_ms": 0.0}
    try:
        yield metrics
    finally:
        end = time.perf_counter()
        metrics["duration_ms"] = (end - start) * 1000.0


def is_safe_filepath(target_path: str, base_dir: str = ".") -> bool:
    """
    Prevents Path Traversal attacks (e.g. '../../etc/passwd').
    Ensures absolute target path is contained within base directory boundaries.
    """
    try:
        base_abs = os.path.abspath(base_dir)
        target_abs = os.path.abspath(os.path.join(base_abs, target_path))
        return target_abs.startswith(base_abs)
    except Exception:
        return False


def sanitize_error_message(error: Exception) -> str:
    """Sanitizes exception messages to prevent sensitive path or token leaks."""
    msg = str(error)
    # Strip user directory paths from tracebacks for privacy
    user_home = os.path.expanduser("~")
    if user_home and user_home in msg:
        msg = msg.replace(user_home, "~")
    return msg.strip()
