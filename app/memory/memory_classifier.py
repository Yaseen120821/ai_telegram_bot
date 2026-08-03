r"""
app/memory/memory_classifier.py - Fact Candidate Classification Engine
========================================================================

1. PURPOSE:
-----------
Analyzes incoming user statements to evaluate whether they contain durable personal facts (name, preferences, goals,
projects, education, skills, career) worth storing in SQLite long-term memory.

2. WHY IT EXISTS (RULE-BASED DESIGN):
--------------------------------------
Rule-based classification executes in <1 millisecond with zero GPU/CPU latency, preventing memory classification
from slowing down the user experience.

3. RESPONSIBILITIES:
--------------------
- Filter out transient queries, greetings, weather chatter, and code generation requests.
- Identify memory category taxonomy (`profile`, `preference`, `goal`, `project`, `education`, `skill`, `career`, `relationship`).
- Extract user profile name assertions (`"my name is"`, `"call me"`, `"i am called"`, `"you can call me"`).
- Assign importance scores (1 to 10 scale) and confidence scores (0.0 to 1.0 scale).
- Return structured `ExtractionResult` objects.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `ExtractionResult` from `app/memory/memory_models.py`.
- Uses `MemoryCategory` and `ImportanceLevel` from `app/memory/memory_types.py`.
- Uses `MemoryUtils` from `app/memory/memory_utils.py`.
- Called by `app/memory/memory_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import re
import logging
from typing import List, Tuple

from app.memory.memory_models import ExtractionResult
from app.memory.memory_types import MemoryCategory, ImportanceLevel
from app.memory.memory_utils import MemoryUtils

logger = logging.getLogger("sana_ai.memory.classifier")


class MemoryClassifier:
    """
    Pattern-matching and heuristic classifier for extracting personal facts from text.
    """

    def __init__(self) -> None:
        """Initializes rule patterns, categories, key templates, importance scores, and confidence scores."""
        # Rules format: (compiled_regex, category, key_template, importance_1_to_10, confidence_0_to_1)
        self.rules: List[Tuple[re.Pattern, str, str, int, float]] = [
            # 1. Profile Patterns (Importance: 10, Confidence: 1.0)
            (re.compile(r"\bmy name is ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "name", 10, 1.0),
            (re.compile(r"\byou can call me ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "name", 10, 1.0),
            (re.compile(r"\bcall me ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "name", 10, 1.0),
            (re.compile(r"\bi am called ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "name", 10, 1.0),
            (re.compile(r"\bi am ([a-zA-Z0-9_\s]+) and i am\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "name", 10, 1.0),
            (re.compile(r"\bi live in ([a-zA-Z0-9_\s,]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "location", 9, 0.95),
            (re.compile(r"\bi am from ([a-zA-Z0-9_\s,]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "hometown", 8, 0.95),
            (re.compile(r"\bmy birthday is ([a-zA-Z0-9_\s,]+)\b", re.IGNORECASE), MemoryCategory.PROFILE.value, "birthday", 10, 1.0),

            # 2. Preference Patterns (Importance: 7-8, Confidence: 0.95)
            (re.compile(r"\bi prefer (coding in |using |working with )?([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.PREFERENCE.value, "preferred_tool", 8, 0.95),
            (re.compile(r"\bi like ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.PREFERENCE.value, "likes", 7, 0.90),
            (re.compile(r"\bi love ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.PREFERENCE.value, "likes", 8, 0.95),
            (re.compile(r"\bmy favorite ([a-zA-Z0-9_]+) is ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.PREFERENCE.value, "favorite_{0}", 8, 0.95),

            # 3. Project Patterns (Importance: 10, Confidence: 1.0)
            (re.compile(r"\bi am building ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROJECT.value, "current_project", 10, 1.0),
            (re.compile(r"\bmy project is ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROJECT.value, "current_project", 10, 1.0),
            (re.compile(r"\bi am working on ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.PROJECT.value, "active_project", 9, 0.95),

            # 4. Goal Patterns (Importance: 9, Confidence: 0.95)
            (re.compile(r"\bi want to become (an? )?([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.GOAL.value, "career_goal", 9, 0.95),
            (re.compile(r"\bmy goal is (to )?([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.GOAL.value, "primary_goal", 9, 0.95),

            # 5. Career & Education Patterns (Importance: 8-9, Confidence: 0.95)
            (re.compile(r"\bi work as (an? )?([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.CAREER.value, "occupation", 9, 0.95),
            (re.compile(r"\bi study ([a-zA-Z0-9_\s]+)\b", re.IGNORECASE), MemoryCategory.EDUCATION.value, "field_of_study", 8, 0.95),

            # 6. Skill Patterns (Importance: 7, Confidence: 0.90)
            (re.compile(r"\bi know ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.SKILL.value, "skill", 7, 0.90),
            (re.compile(r"\bi know how to use ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.SKILL.value, "skill", 7, 0.90),
            (re.compile(r"\bmy favorite database is ([a-zA-Z0-9_\s#+]+)\b", re.IGNORECASE), MemoryCategory.PREFERENCE.value, "favorite_database", 8, 0.95),
        ]

    def classify_statement(self, user_text: str) -> ExtractionResult:
        """
        Analyzes user_text to extract candidate personal facts.

        Args:
            user_text (str): Raw input statement from user.

        Returns:
            ExtractionResult: Extraction result indicating if statement is a memory candidate.
        """
        if MemoryUtils.is_transient_prompt(user_text):
            logger.debug(f"Transient prompt ignored: '{user_text[:30]}...'")
            return ExtractionResult(is_candidate=False)

        cleaned_text = MemoryUtils.clean_user_input(user_text)

        # Match regex patterns against rules
        for pattern, category, key_template, importance, confidence in self.rules:
            match = pattern.search(cleaned_text)
            if match:
                groups = match.groups()
                
                # Handle dynamic key substitution (e.g. favorite_{0})
                if "{0}" in key_template and len(groups) >= 2:
                    memory_key = MemoryUtils.normalize_key_fact(key_template.format(groups[0]))
                    memory_value = MemoryUtils.clean_user_input(groups[1])
                elif len(groups) >= 2:
                    memory_key = MemoryUtils.normalize_key_fact(key_template)
                    memory_value = MemoryUtils.clean_user_input(groups[1])
                else:
                    memory_key = MemoryUtils.normalize_key_fact(key_template)
                    memory_value = MemoryUtils.clean_user_input(groups[0])

                if memory_value:
                    logger.info(
                        f"🎯 Extracted Memory Fact | Category: '{category}' | Key: '{memory_key}' | "
                        f"Value: '{memory_value}' | Importance: {importance} | Confidence: {confidence}"
                    )
                    return ExtractionResult(
                        is_candidate=True,
                        category=category,
                        memory_key=memory_key,
                        memory_value=memory_value,
                        confidence=confidence,
                        importance=importance
                    )

        return ExtractionResult(is_candidate=False)
