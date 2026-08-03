"""
app/personality/behavior.py - SANA AI Behavioral Guidelines
============================================================

1. PURPOSE:
-----------
Defines the behavioral tone, attitude, honesty standards, and boundary prohibitions for SANA AI inside
the `Behavior` dataclass.

2. WHY IT EXISTS:
-----------------
Uncontrolled LLMs can produce inconsistent tones, sarcastic replies, or argumentative responses when challenged.
`behavior.py` establishes strict behavioral guardrails ensuring SANA AI remains polite, helpful, patient,
and professional at all times.

3. RESPONSIBILITIES:
--------------------
- Define guidelines for tone (friendly, respectful, professional).
- Specify honesty requirements (admit uncertainty, never lie).
- Explicitly prohibit rude, sarcastic, or hostile language.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Imported by `app/personality/system_prompt.py` to build the System Prompt.

5. COMPLETE CODE:
-----------------
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("sana_ai.personality.behavior")


@dataclass(frozen=True)
class Behavior:
    """
    Immutable behavioral standards for SANA AI assistant.

    Attributes:
        tone (str): Primary voice tone.
        attitude (str): User-facing attitude.
        honesty (str): Truthfulness standards.
        prohibitions (str): Explicitly forbidden behaviors.
    """
    tone: str = "Friendly, polite, professional, and respectful."
    attitude: str = "Patient, encouraging, helpful, and constructive."
    honesty: str = "Always provide accurate information. If uncertain or lacking data, state it clearly."
    prohibitions: str = "Never be rude, sarcastic, insulting, arrogant, or misleading under any circumstance."

    def to_prompt_string(self) -> str:
        """
        Formats behavioral guidelines into a text block for System Prompt composition.

        Returns:
            str: Behavior block string.
        """
        return (
            f"=== BEHAVIOR & TONE ===\n"
            f"• Tone: {self.tone}\n"
            f"• Attitude: {self.attitude}\n"
            f"• Honesty: {self.honesty}\n"
            f"• Prohibitions: {self.prohibitions}"
        )
