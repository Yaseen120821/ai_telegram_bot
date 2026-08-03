"""
app/tools/routing/confidence_engine.py - Confidence Scoring & Threshold Engine
==============================================================================
Calculates 0.00 to 1.00 numeric confidence ratings, maps scores to ConfidenceLevel
thresholds (HIGH, MEDIUM, LOW, NONE), and explains scoring decisions.
"""

import re
import logging
from typing import List, Optional
from app.tools.base_tool import BaseTool
from app.tools.routing.router_types import ConfidenceLevel
from app.tools.routing.router_models import ConfidenceScore, IntentResult, ToolCandidate
from app.tools.routing.router_config import get_router_config

logger = logging.getLogger("sana_ai.tools.routing.confidence")


class ConfidenceEngine:
    """Calculates and rates confidence levels for candidate tool selections."""

    def __init__(self):
        self.config = get_router_config()

    def evaluate_candidate(
        self,
        query: str,
        tool: BaseTool,
        intent_result: Optional[IntentResult] = None
    ) -> ToolCandidate:
        """
        Computes composite confidence score for matching a tool to a query & intent.
        
        Factors:
        - Intent result match (+0.40 if intent directly maps to tool category)
        - Tool name & token matches (+0.35)
        - Parameter completeness (+0.15)
        - Description & Tag match (+0.25)
        """
        score = 0.0
        matched_words: List[str] = []
        q_lower = query.lower()

        # 1. Direct Intent Alignment & Rule Matching
        if intent_result and intent_result.intent_confidence > 0.5:
            intent_val = intent_result.primary_intent.value
            intent_tokens = intent_val.split("_")
            tool_name_lower = tool.name.lower()
            tool_tags_lower = [t.lower() for t in tool.metadata.tags]
            
            # Check if intent tokens align with tool name, category, or tags using stem/prefix matching
            matches_intent = (
                any((t in tool_name_lower or any(t.startswith(q[:4]) or q.startswith(t[:4]) for q in re.findall(r'\w+', tool_name_lower))) for t in intent_tokens if len(t) > 2) or
                any(t in tool.category.value for t in intent_tokens if len(t) > 2) or
                any((t in tag or any(t.startswith(q[:4]) or q.startswith(t[:4]) for q in re.findall(r'\w+', tag))) for t in intent_tokens for tag in tool_tags_lower if len(t) > 2)
            )
            if matches_intent:
                score += 0.55
                matched_words.append(f"intent:{intent_val}")

        # 2. Tool Name & Token Matches
        if tool.name.lower() in q_lower:
            score += 0.40
            matched_words.append(tool.name)

        # 3. Parameter Completeness
        if intent_result and intent_result.extracted_parameters:
            schema_props = tool.parameters_schema.get("properties", {})
            for p_name in intent_result.extracted_parameters.keys():
                if p_name in schema_props:
                    score += 0.20
                    matched_words.append(f"param:{p_name}")

        # 4. Description & Tags Match
        for tag in tool.metadata.tags:
            if tag.lower() in q_lower:
                score += 0.25
                matched_words.append(f"tag:{tag}")

        final_score = min(score, 1.0)
        explanation = f"Matched factors: {', '.join(matched_words)}" if matched_words else "No factors matched"

        return ToolCandidate(
            tool=tool,
            score=final_score,
            matched_keywords=list(set(matched_words)),
            reasoning=explanation
        )

    def classify_score(self, numeric_score: float) -> ConfidenceScore:
        """Categorizes raw numeric score into ConfidenceLevel enum with explanation."""
        if numeric_score >= self.config.high_confidence_threshold:
            level = ConfidenceLevel.HIGH
            expl = f"High confidence ({numeric_score:.2f} >= {self.config.high_confidence_threshold:.2f})"
        elif numeric_score >= self.config.medium_confidence_threshold:
            level = ConfidenceLevel.MEDIUM
            expl = f"Medium confidence ({numeric_score:.2f} >= {self.config.medium_confidence_threshold:.2f})"
        elif numeric_score > 0.0:
            level = ConfidenceLevel.LOW
            expl = f"Low confidence ({numeric_score:.2f} < {self.config.medium_confidence_threshold:.2f})"
        else:
            level = ConfidenceLevel.NONE
            expl = "Zero confidence match."

        return ConfidenceScore(numeric_score=numeric_score, level=level, explanation=expl)
