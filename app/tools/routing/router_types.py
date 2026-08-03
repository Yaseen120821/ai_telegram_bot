"""
app/tools/routing/router_types.py - Enumeration Types for Decision & Routing Engine
==================================================================================
Defines enums for intent categories, routing modes, decision types, confidence levels,
and parameter data types within SANA AI's routing subsystem.
"""

from enum import Enum


class IntentType(str, Enum):
    """Classified user intent categories."""
    GET_TIME = "get_time"
    CALCULATE = "calculate"
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    SYSTEM_INFO = "system_info"
    SEARCH_WEB = "search_web"
    FETCH_RAG = "fetch_rag"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class RoutingMode(str, Enum):
    """Operational modes for routing execution."""
    DIRECT_RESPONSE = "direct_response"       # Answer directly via LLM weights
    TOOL = "tool"                             # Execute a single specific tool
    RAG = "rag"                               # Perform vector document retrieval
    TOOL_AND_RAG = "tool_and_rag"             # Execute tool and retrieve RAG document context
    MULTIPLE_TOOLS = "multiple_tools"         # Execute a sequence/chain of tools
    CLARIFICATION_REQUIRED = "clarification"  # Query user for ambiguous missing arguments
    MULTIMODAL = "multimodal"                 # Multimodal image, PDF, document & visual context processing


class DecisionType(str, Enum):
    """High-level routing decision categories."""
    EXECUTE_TOOL = "execute_tool"
    EXECUTE_RAG = "execute_rag"
    DIRECT_LLM = "direct_llm"
    ASK_CLARIFICATION = "ask_clarification"
    MULTIMODAL_RESPONSE = "multimodal_response"
    FALLBACK = "fallback"


class ConfidenceLevel(str, Enum):
    """Categorized confidence score levels."""
    HIGH = "high"      # Score >= 0.80 -> Immediate Tool/RAG execution
    MEDIUM = "medium"  # Score >= 0.60 -> Execute with safety checks or LLM verification
    LOW = "low"        # Score < 0.60  -> Fallback to direct LLM or clarify
    NONE = "none"      # Score == 0.0  -> Completely unhandled intent


class ParameterType(str, Enum):
    """Data types for extracted parameters."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    FILE_PATH = "file_path"
    EXPRESSION = "expression"
    QUERY = "query"
