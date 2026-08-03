"""
app/tools/routing/decision_engine.py - Core Decision Brain of SANA AI
====================================================================
Consumes user query, analyzes intent, checks registry inventory, evaluates confidence,
and selects optimal RoutingMode (DIRECT_RESPONSE, TOOL, RAG, TOOL_AND_RAG, CLARIFICATION).
"""

import logging
from typing import List, Optional, Dict, Any
from app.tools.base_tool import BaseTool
from app.tools.registry import RegistryManager
from app.tools.routing.router_types import IntentType, RoutingMode, DecisionType, ConfidenceLevel
from app.tools.routing.router_models import (
    RoutingDecision,
    IntentResult,
    ToolCandidate,
    ConfidenceScore,
    ExtractedParameter
)
from app.tools.routing.intent_detector import IntentDetector
from app.tools.routing.confidence_engine import ConfidenceEngine
from app.tools.routing.router_config import get_router_config

logger = logging.getLogger("sana_ai.tools.routing.decision")


class DecisionEngine:
    """
    Central Brain of SANA AI Tool Calling Framework.
    
    Architectural Isolation:
    - Never executes tools.
    - Never builds system prompt text.
    - Never sends messages to Telegram.
    """

    def __init__(
        self,
        registry: Optional[RegistryManager] = None,
        intent_detector: Optional[IntentDetector] = None,
        confidence_engine: Optional[ConfidenceEngine] = None
    ):
        self.registry = registry or RegistryManager.get_instance()
        self.intent_detector = intent_detector or IntentDetector()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.config = get_router_config()

    def evaluate_request(
        self,
        query: str,
        manual_override_tool: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Evaluates user query and returns structured RoutingDecision.
        """
        query_text = query.strip()
        if not query_text:
            return RoutingDecision(
                routing_mode=RoutingMode.DIRECT_RESPONSE,
                decision_type=DecisionType.DIRECT_LLM,
                confidence=ConfidenceScore(0.0, ConfidenceLevel.NONE, "Empty query text."),
                fallback_reason="Empty user query text."
            )

        # 1. Manual Override Handling
        if manual_override_tool:
            if self.registry.has_tool(manual_override_tool):
                tool = self.registry.get_tool(manual_override_tool)
                if tool.is_enabled:
                    logger.info(f"Manual tool override satisfied: '{manual_override_tool}'.")
                    candidate = ToolCandidate(tool=tool, score=1.0, reasoning="Manual Override")
                    return RoutingDecision(
                        routing_mode=RoutingMode.TOOL,
                        decision_type=DecisionType.EXECUTE_TOOL,
                        selected_tool=tool,
                        candidates=[candidate],
                        confidence=ConfidenceScore(1.0, ConfidenceLevel.HIGH, "Manual Override")
                    )

        # 2. Intent Detection
        intent_result: IntentResult = self.intent_detector.detect_intent(query_text, context)

        # Handle Explicit RAG Intent
        if intent_result.primary_intent == IntentType.FETCH_RAG:
            logger.info("Routing query to RAG retrieval pipeline.")
            return RoutingDecision(
                routing_mode=RoutingMode.RAG,
                decision_type=DecisionType.EXECUTE_RAG,
                confidence=ConfidenceScore(intent_result.intent_confidence, ConfidenceLevel.HIGH, "RAG intent matched.")
            )

        # 3. Query Active Tools in Registry
        active_tools = self.registry.list_tools(only_enabled=True)
        if not active_tools:
            logger.info("No tools registered in registry. Falling back to direct LLM response.")
            return RoutingDecision(
                routing_mode=RoutingMode.DIRECT_RESPONSE,
                decision_type=DecisionType.DIRECT_LLM,
                confidence=ConfidenceScore(0.0, ConfidenceLevel.NONE, "No active tools registered."),
                fallback_reason="Registry inventory empty."
            )

        # 4. Score Candidate Tools
        candidates: List[ToolCandidate] = []
        for tool in active_tools:
            candidate = self.confidence_engine.evaluate_candidate(query_text, tool, intent_result)
            if candidate.score > 0.1:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            logger.info("No candidate tools matched query threshold. Routing to Direct LLM.")
            return RoutingDecision(
                routing_mode=RoutingMode.DIRECT_RESPONSE,
                decision_type=DecisionType.DIRECT_LLM,
                confidence=ConfidenceScore(0.0, ConfidenceLevel.NONE, "Query matched no tool signatures.")
            )

        top_candidate = candidates[0]
        confidence_obj = self.confidence_engine.classify_score(top_candidate.score)

        # 5. Evaluate Decision Thresholds
        if confidence_obj.level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM):
            # Check for missing required parameters
            missing_params = self._check_missing_required_params(top_candidate.tool, intent_result.extracted_parameters)
            if missing_params:
                logger.info(f"Candidate tool '{top_candidate.tool.name}' requires missing param(s): {missing_params}.")
                return RoutingDecision(
                    routing_mode=RoutingMode.CLARIFICATION_REQUIRED,
                    decision_type=DecisionType.ASK_CLARIFICATION,
                    selected_tool=top_candidate.tool,
                    candidates=candidates[:self.config.max_candidate_tools],
                    extracted_parameters=intent_result.extracted_parameters,
                    confidence=confidence_obj,
                    needs_clarification=True,
                    clarification_prompt=f"Please specify the required parameter(s): {', '.join(missing_params)}"
                )

            logger.info(f"Tool decision confirmed: '{top_candidate.tool.name}' (Confidence: {top_candidate.score:.2f}, Level: {confidence_obj.level.value}).")
            return RoutingDecision(
                routing_mode=RoutingMode.TOOL,
                decision_type=DecisionType.EXECUTE_TOOL,
                selected_tool=top_candidate.tool,
                candidates=candidates[:self.config.max_candidate_tools],
                extracted_parameters=intent_result.extracted_parameters,
                confidence=confidence_obj
            )

        # Fallback to Direct LLM for Low/None confidence
        logger.info(f"Top tool score ({top_candidate.score:.2f}) below threshold ({self.config.medium_confidence_threshold:.2f}). Direct LLM fallback.")
        return RoutingDecision(
            routing_mode=RoutingMode.DIRECT_RESPONSE,
            decision_type=DecisionType.DIRECT_LLM,
            candidates=candidates[:self.config.max_candidate_tools],
            confidence=confidence_obj,
            fallback_reason=f"Top tool confidence ({top_candidate.score:.2f}) below medium threshold."
        )

    def _check_missing_required_params(self, tool: BaseTool, extracted: Dict[str, Any]) -> List[str]:
        """Checks if any required parameter in tool schema is missing from extracted arguments."""
        schema = tool.parameters_schema
        required = schema.get("required", [])
        return [r for r in required if r not in extracted]
