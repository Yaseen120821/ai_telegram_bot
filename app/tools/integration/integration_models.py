"""
app/tools/integration/integration_models.py - Integration Pipeline Data Models
================================================================================
Defines strongly-typed dataclasses for tool contexts, integrated multi-source contexts,
prompt contexts, execution summaries, decision traces, and combined responses.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from app.tools.execution.execution_models import ExecutionResult
from app.tools.routing.router_models import RoutingDecision


@dataclass
class ToolContext:
    """Context holding formatted tool execution results for prompt builder injection."""
    results: List[ExecutionResult] = field(default_factory=list)
    formatted_block: str = ""
    total_execution_time_ms: float = 0.0


@dataclass
class DecisionTrace:
    """Audit log recording multi-source routing decisions."""
    decision_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    query: str = ""
    routing_decision: Optional[RoutingDecision] = None
    recalled_memories_count: int = 0
    retrieved_rag_chunks_count: int = 0
    tools_executed_count: int = 0
    emotion_detected: str = "neutral"
    pipeline_execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PromptContext:
    """Token-budgeted prompt payload ready for PromptBuilder."""
    system_prompt: str = ""
    user_input: str = ""
    memory_context: Optional[str] = None
    emotion_directive: Optional[str] = None
    rag_context: Optional[str] = None
    tool_context: Optional[str] = None
    vision_context: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class IntegratedContext:
    """Combined context aggregating Memory, Emotion, RAG, Tool execution, and Vision states."""
    raw_query: str
    user_id: str = "default_user"
    conversation_id: str = "default_session"
    memory_context: Optional[str] = None
    emotion_context: Optional[Any] = None
    rag_context: Optional[str] = None
    tool_context: Optional[ToolContext] = None
    vision_context: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    decision: Optional[RoutingDecision] = None
    trace: DecisionTrace = field(default_factory=DecisionTrace)


@dataclass
class ExecutionSummary:
    """High-level summary of pipeline execution metrics."""
    query: str
    answered_directly: bool = False
    used_memory: bool = False
    used_rag: bool = False
    used_tool: bool = False
    tools_used: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0


@dataclass
class CombinedResponse:
    """Final unified response returned by IntelligentPipeline to Telegram / Client."""
    response_text: str
    summary: ExecutionSummary
    integrated_context: IntegratedContext
    trace: DecisionTrace
