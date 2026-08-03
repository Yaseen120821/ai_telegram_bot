"""
app/tools/tool_manager.py - High-Level Tool Calling Orchestrator & Facade
========================================================================
Acts as the unified entry point for Phase 2 Tool Calling within SANA AI.
Coordinates ToolRouter, ToolRegistry, PermissionManager, and ToolExecutor.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from app.tools.tool_registry import ToolRegistry
from app.tools.tool_router import ToolRouter, RoutingDecision
from app.tools.permission_manager import PermissionManager
from app.tools.tool_executor import ToolExecutor
from app.tools.base_tool import BaseTool
from app.tools.tool_models import ToolRequest, ToolResponse, ExecutionContext
from app.tools.tool_config import get_tool_config

logger = logging.getLogger("sana_ai.tools.manager")


class ToolManager:
    """
    Facade and orchestrator for SANA AI's Tool Calling Subsystem.
    
    Architectural Position:
    Telegram -> Conversation -> Emotion -> Memory -> RAG -> ToolManager -> Prompt Builder -> Generator -> Qwen
    
    Why Independent:
    ToolManager isolates intent detection, security authorization, and execution dynamics from
    conversation history, emotional context, RAG document search, and prompt template construction.
    """

    _instance: Optional["ToolManager"] = None

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        router: Optional[ToolRouter] = None,
        permission_manager: Optional[PermissionManager] = None,
        executor: Optional[ToolExecutor] = None
    ):
        self.registry = registry or ToolRegistry.get_instance()
        self.permission_manager = permission_manager or PermissionManager()
        self.router = router or ToolRouter(registry=self.registry)
        self.executor = executor or ToolExecutor(
            registry=self.registry,
            permission_manager=self.permission_manager
        )
        self.config = get_tool_config()
        logger.info("ToolManager initialized successfully.")

    @classmethod
    def get_instance(cls) -> "ToolManager":
        """Returns ToolManager singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_tool(self, tool: BaseTool) -> None:
        """Registers a tool into the framework."""
        self.registry.register(tool)

    def route_and_execute(
        self,
        query: str,
        user_id: str = "default_user",
        manual_override: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        user_confirmed: bool = False,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[RoutingDecision, Optional[ToolResponse]]:
        """
        High-level pipeline method:
        1. Routes user query to match tool intent.
        2. If candidate matched, executes tool through ToolExecutor.
        3. Returns routing decision alongside structured tool response.
        """
        # Step 1: Route query
        decision, tool_request = self.router.route_query(
            query=query,
            user_id=user_id,
            manual_override_tool=manual_override,
            parameters=parameters,
            user_confirmed=user_confirmed,
            context=context
        )

        if not decision.should_call_tool or not decision.selected_tool or not tool_request:
            logger.info(f"Query routing result: No tool call executed. Reason: {decision.fallback_reason}")
            return decision, None

        # Step 2: Build ToolRequest
        params = parameters or decision.extracted_parameters
        request = ToolRequest(
            tool_name=decision.selected_tool.name,
            parameters=params,
            user_id=user_id,
            context=context,
            user_confirmed=user_confirmed
        )

        # Step 3: Execute tool
        response = self.executor.execute(request)
        return decision, response

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_id: str = "default_user",
        user_confirmed: bool = False,
        context: Optional[ExecutionContext] = None
    ) -> ToolResponse:
        """Directly executes a tool by name (bypassing routing)."""
        request = ToolRequest(
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id,
            context=context,
            user_confirmed=user_confirmed
        )
        return self.executor.execute(request)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Exposes JSON schemas for all registered and enabled tools."""
        return self.registry.get_all_tool_schemas()
