"""
app/tools/routing/tool_router.py - Enterprise Tool Router & Request Dispatcher
=============================================================================
Consumes RoutingDecision from DecisionEngine, validates tool state & parameters,
checks security authorization via PermissionManager, and builds dispatchable ToolRequest payloads.
"""

import logging
from typing import Optional, Tuple, Dict, Any
from app.tools.base_tool import BaseTool
from app.tools.registry import RegistryManager
from app.tools.permission_manager import PermissionManager
from app.tools.tool_models import ToolRequest, ExecutionContext
from app.tools.routing.decision_engine import DecisionEngine
from app.tools.routing.router_models import RoutingDecision
from app.tools.routing.router_config import get_router_config

logger = logging.getLogger("sana_ai.tools.routing.router")


class ToolRouter:
    """
    Router & Request Dispatcher.
    
    Consumes decisions from DecisionEngine, enforces security checks,
    and formats structured ToolRequest payloads.
    """

    def __init__(
        self,
        registry: Optional[RegistryManager] = None,
        decision_engine: Optional[DecisionEngine] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        self.registry = registry or RegistryManager.get_instance()
        self.decision_engine = decision_engine or DecisionEngine(registry=self.registry)
        self.permission_manager = permission_manager or PermissionManager()
        self.config = get_router_config()

    def route_query(
        self,
        query: str,
        user_id: str = "default_user",
        manual_override_tool: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        user_confirmed: bool = False,
        context: Optional[ExecutionContext] = None
    ) -> Tuple[RoutingDecision, Optional[ToolRequest]]:
        """
        High-level routing entry point.
        
        1. Query DecisionEngine to evaluate routing decision.
        2. If tool decision confirmed, validate tool state & construct ToolRequest.
        3. Return Tuple[RoutingDecision, Optional[ToolRequest]].
        """
        ctx_dict = context if isinstance(context, dict) else (context.metadata if (context and hasattr(context, "metadata")) else None)

        decision: RoutingDecision = self.decision_engine.evaluate_request(
            query=query,
            manual_override_tool=manual_override_tool,
            context=ctx_dict
        )

        if not decision.should_call_tool or not decision.selected_tool:
            logger.info(f"Routing outcome: No tool call dispatched. (Mode: {decision.routing_mode.value})")
            return decision, None

        tool: BaseTool = decision.selected_tool

        # Validate enabled status
        if not tool.is_enabled:
            logger.warning(f"Routing rejected: Tool '{tool.name}' is disabled.")
            decision.fallback_reason = f"Tool '{tool.name}' is disabled."
            return decision, None

        # Build final parameters dictionary (merge decision parameters with explicit overrides)
        final_params = parameters if parameters is not None else decision.extracted_parameters

        request = ToolRequest(
            tool_name=tool.name,
            parameters=final_params,
            user_id=user_id,
            context=context,
            user_confirmed=user_confirmed,
            priority=tool.metadata.priority
        )

        logger.info(f"Tool request successfully built for tool '{tool.name}' [Request ID: {request.request_id}].")
        return decision, request
