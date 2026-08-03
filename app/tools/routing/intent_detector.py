"""
app/tools/routing/intent_detector.py - Intent & Parameter Extraction Engine
===========================================================================
Extracts intent classification, parameter maps, context targets, and goal summaries
from incoming user messages using RuleEngine and fallback analysis.
"""

import logging
from typing import Optional, Dict, Any
from app.tools.routing.router_types import IntentType
from app.tools.routing.router_models import IntentResult
from app.tools.routing.rule_engine import RuleEngine
from app.tools.routing.router_config import get_router_config

logger = logging.getLogger("sana_ai.tools.routing.intent")


class IntentDetector:
    """
    Analyzes incoming user queries to classify Intent, Entities, Parameters, Context, and Goal.
    """

    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        self.rule_engine = rule_engine or RuleEngine()
        self.config = get_router_config()

    def detect_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Detects user intent from text.
        
        Steps:
        1. Query RuleEngine for fast sub-millisecond pattern matching.
        2. If rule matched -> return IntentResult.
        3. Fallback -> return UNKNOWN or GENERAL_CHAT intent with baseline confidence.
        """
        query_text = query.strip()
        if not query_text:
            return IntentResult(
                primary_intent=IntentType.UNKNOWN,
                raw_query="",
                intent_confidence=0.0,
                goal_summary="Empty query"
            )

        # Step 1: Rule Engine Check
        if self.config.enable_rule_engine:
            rule_result = self.rule_engine.match_intent(query_text)
            if rule_result:
                logger.info(
                    f"Intent detected via RuleEngine: Intent={rule_result.primary_intent.value} "
                    f"(Confidence: {rule_result.intent_confidence:.2f}, Rule: {rule_result.matched_rule})"
                )
                return rule_result

        # Step 2: Fallback Intent Detection
        logger.debug(f"Rule matching missed for query '{query_text}'. Classifying as GENERAL_CHAT / UNKNOWN fallback.")
        return IntentResult(
            primary_intent=IntentType.GENERAL_CHAT,
            raw_query=query_text,
            extracted_parameters={},
            intent_confidence=0.30,
            goal_summary="General conversational response"
        )
