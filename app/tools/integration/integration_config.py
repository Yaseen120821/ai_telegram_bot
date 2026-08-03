"""
app/tools/integration/integration_config.py - Integration Pipeline Configuration
=================================================================================
Manages feature flags, token budgets, context limits, and logging settings for
the SANA AI Intelligent Decision Pipeline.
"""

import os
from dataclasses import dataclass


@dataclass
class PipelineIntegrationConfig:
    """Centralized configuration for Intelligent Decision Pipeline."""

    enable_tool_integration: bool = True
    enable_rag_integration: bool = True
    enable_memory_integration: bool = True
    max_tool_results: int = 3
    max_memory_items: int = 5
    max_retrieved_chunks: int = 4
    prompt_token_budget: int = 4096
    log_pipeline_traces: bool = True

    @classmethod
    def from_env(cls) -> "PipelineIntegrationConfig":
        return cls(
            enable_tool_integration=os.getenv("SANA_ENABLE_TOOLS", "true").lower() == "true",
            enable_rag_integration=os.getenv("SANA_ENABLE_RAG", "true").lower() == "true",
            enable_memory_integration=os.getenv("SANA_ENABLE_MEMORY", "true").lower() == "true",
            max_tool_results=int(os.getenv("SANA_MAX_TOOL_RESULTS", "3")),
            max_memory_items=int(os.getenv("SANA_MAX_MEMORIES", "5")),
            max_retrieved_chunks=int(os.getenv("SANA_MAX_RAG_CHUNKS", "4")),
            prompt_token_budget=int(os.getenv("SANA_PROMPT_TOKEN_BUDGET", "4096")),
            log_pipeline_traces=os.getenv("SANA_LOG_TRACES", "true").lower() == "true",
        )


_pipeline_config_instance: PipelineIntegrationConfig = PipelineIntegrationConfig.from_env()


def get_pipeline_config() -> PipelineIntegrationConfig:
    """Returns global pipeline configuration singleton."""
    return _pipeline_config_instance


def set_pipeline_config(config: PipelineIntegrationConfig) -> None:
    """Overrides global pipeline configuration."""
    global _pipeline_config_instance
    _pipeline_config_instance = config
