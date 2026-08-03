"""
app/tools/routing/rule_engine.py - Rule-Based Deterministic Matching Engine
===========================================================================
Provides fast, sub-millisecond intent pattern matching using regular expressions
and keyword rule indexes without requiring LLM inference calls.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.tools.routing.router_types import IntentType, ParameterType
from app.tools.routing.router_models import IntentResult, ExtractedParameter
from app.tools.routing.router_utils import normalize_query_text, extract_filepath_from_text, extract_arithmetic_expression

logger = logging.getLogger("sana_ai.tools.routing.rule_engine")


class RuleEngine:
    """
    Deterministic rule & pattern matching engine.
    
    Advantages:
    - Speed (< 1 millisecond execution)
    - Reliability (Deterministic outcome)
    - Low Cost (Zero LLM token consumption)
    """

    def __init__(self):
        self._rules: List[Dict[str, Any]] = [
            {
                "rule_id": "rule_get_time",
                "intent": IntentType.GET_TIME,
                "patterns": [
                    r"\bwhat\s+time\s+is\s+it\b",
                    r"\bcurrent\s+time\b",
                    r"\btell\s+me\s+the\s+time\b",
                    r"\bwhat\s+is\s+the\s+time\b"
                ],
                "confidence": 0.95
            },
            {
                "rule_id": "rule_calculate",
                "intent": IntentType.CALCULATE,
                "patterns": [
                    r"\bcalculate\b",
                    r"\bwhat\s+is\s+\d+[\s\+\-\*\/\(\)]+\d+\b",
                    r"\bevaluate\s+expression\b"
                ],
                "confidence": 0.92
            },
            {
                "rule_id": "rule_read_file",
                "intent": IntentType.READ_FILE,
                "patterns": [
                    r"\bread\s+(?:file|document|my)?\s*",
                    r"\bopen\s+(?:file|document)\s*",
                    r"\bshow\s+contents\s+of\b",
                    r"\bcat\s+[a-zA-Z0-9_\-\.]+\b"
                ],
                "confidence": 0.88
            },
            {
                "rule_id": "rule_system_info",
                "intent": IntentType.SYSTEM_INFO,
                "patterns": [
                    r"\bsystem\s+status\b",
                    r"\bcpu\s+usage\b",
                    r"\bmemory\s+usage\b",
                    r"\bdiagnostics\b"
                ],
                "confidence": 0.90
            },
            {
                "rule_id": "rule_fetch_rag",
                "intent": IntentType.FETCH_RAG,
                "patterns": [
                    r"\bsearch\s+docs\b",
                    r"\bsearch\s+documentation\b",
                    r"\binternal\s+manual\b",
                    r"\baccording\s+to\s+the\s+docs\b"
                ],
                "confidence": 0.85
            }
        ]

    def match_intent(self, query: str) -> Optional[IntentResult]:
        """
        Evaluates query string against registered regex pattern rules.
        Returns IntentResult if a pattern matches, or None if no rule applies.
        """
        norm_text = normalize_query_text(query)

        for rule in self._rules:
            for pattern in rule["patterns"]:
                if re.search(pattern, norm_text, re.IGNORECASE):
                    intent = rule["intent"]
                    rule_id = rule["rule_id"]
                    conf = rule["confidence"]
                    params = self._extract_rule_parameters(query, intent)

                    logger.debug(f"Rule match success: '{rule_id}' matched query '{query}' (Intent: {intent.value}).")
                    return IntentResult(
                        primary_intent=intent,
                        raw_query=query,
                        extracted_parameters=params,
                        intent_confidence=conf,
                        matched_rule=rule_id,
                        goal_summary=f"Matched rule {rule_id}"
                    )

        return None

    def _extract_rule_parameters(self, query: str, intent: IntentType) -> Dict[str, Any]:
        """Extracts parameters associated with matched intent."""
        params = {}
        if intent == IntentType.CALCULATE:
            expr = extract_arithmetic_expression(query)
            if expr:
                params["expression"] = expr
        elif intent in (IntentType.READ_FILE, IntentType.WRITE_FILE):
            fpath = extract_filepath_from_text(query)
            if fpath:
                params["filename"] = fpath
        return params
