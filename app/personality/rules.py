r"""
app/personality/rules.py - SANA AI Boundary & Safety Rules
===========================================================

1. PURPOSE:
-----------
Defines non-negotiable operational boundaries, confidentiality rules, and feature honesty standards
inside the `Rules` dataclass.

2. WHY IT EXISTS:
-----------------
Establishes hard constraints the model must obey under all circumstances, protecting privacy while accurately
representing the assistant's capabilities as a local AI with long-term memory and document retrieval.

3. RESPONSIBILITIES:
--------------------
- Enforce absolute truthfulness and hallucination prevention.
- Protect system prompt confidentiality.
- Accurately represent memory and RAG capabilities without pretending to have un-connected live tools.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Imported by `app/personality/system_prompt.py` for System Prompt compilation.

5. COMPLETE CODE:
-----------------
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("sana_ai.personality.rules")


@dataclass(frozen=True)
class Rules:
    """
    Immutable boundary and safety rules.

    Attributes:
        truthfulness (str): Factuality enforcement.
        confidentiality (str): System prompt protection.
        feature_honesty (str): Capability transparency.
        privacy (str): Data privacy enforcement.
    """
    truthfulness: str = "Always state accurate facts. Ground answers in recalled user memories and retrieved document context."
    confidentiality: str = "Never expose internal system implementation details unless requested by the developer."
    feature_honesty: str = "You possess persistent SQLite long-term memory and local document retrieval (RAG). Use recalled user memories to personalize responses."
    privacy: str = "Respect user privacy and maintain professional boundaries at all times."

    def to_prompt_string(self) -> str:
        """
        Formats operational rules into a text block for System Prompt composition.

        Returns:
            str: Rules block string.
        """
        return (
            f"=== OPERATIONAL RULES & BOUNDARIES ===\n"
            f"1. {self.truthfulness}\n"
            f"2. {self.confidentiality}\n"
            f"3. {self.feature_honesty}\n"
            f"4. {self.privacy}"
        )
