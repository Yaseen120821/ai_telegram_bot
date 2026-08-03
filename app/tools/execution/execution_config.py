"""
app/tools/execution/execution_config.py - Centralized Configuration for Execution Pipeline
==========================================================================================
Manages execution limits, timeout thresholds, retry strategies, permission policies,
and formatting options for the Tool Execution Subsystem.
"""

import os
from dataclasses import dataclass
from app.tools.execution.execution_types import RetryPolicy, TimeoutPolicy


@dataclass
class ToolExecutionConfig:
    """Central configuration for SANA AI Tool Execution Subsystem."""

    default_timeout_seconds: float = 10.0
    max_retries: int = 2
    initial_retry_delay_seconds: float = 0.2
    retry_policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    timeout_policy: TimeoutPolicy = TimeoutPolicy.STRICT_ABORT
    max_concurrent_executions: int = 5
    log_execution_payloads: bool = True
    sanitize_output: bool = True
    format_results_for_prompt: bool = True

    @classmethod
    def from_env(cls) -> "ToolExecutionConfig":
        """Instantiates configuration from environment variables with defaults."""
        return cls(
            default_timeout_seconds=float(os.getenv("SANA_EXEC_TIMEOUT", "10.0")),
            max_retries=int(os.getenv("SANA_EXEC_MAX_RETRIES", "2")),
            initial_retry_delay_seconds=float(os.getenv("SANA_EXEC_RETRY_DELAY", "0.2")),
            max_concurrent_executions=int(os.getenv("SANA_EXEC_MAX_CONCURRENT", "5")),
            log_execution_payloads=os.getenv("SANA_EXEC_LOG_PAYLOADS", "true").lower() == "true",
            sanitize_output=os.getenv("SANA_EXEC_SANITIZE", "true").lower() == "true",
        )


_execution_config_instance: ToolExecutionConfig = ToolExecutionConfig.from_env()


def get_execution_config() -> ToolExecutionConfig:
    """Returns global execution configuration singleton."""
    return _execution_config_instance


def set_execution_config(config: ToolExecutionConfig) -> None:
    """Overrides global execution configuration."""
    global _execution_config_instance
    _execution_config_instance = config
