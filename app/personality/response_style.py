"""
app/personality/response_style.py - SANA AI Response Formatting Style
======================================================================

1. PURPOSE:
-----------
Defines output formatting, structural hierarchy, coding conventions, and explanation style for SANA AI
in the `ResponseStyle` dataclass.

2. WHY IT EXISTS:
-----------------
Raw LLM outputs can vary wildly in structure—sometimes outputting unformatted walls of text or bare code blocks
without explanation. `response_style.py` enforces clean GitHub Markdown formatting, clear headers, bullet lists,
and educational step-by-step explanations.

3. RESPONSIBILITIES:
--------------------
- Enforce Markdown layout guidelines (headings, bullet points, bold emphasis).
- Mandate conceptual explanations prior to presenting code snippets.
- Ensure code blocks specify language identifiers (e.g. ```python, ```java).

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Imported by `app/personality/system_prompt.py` for System Prompt compilation.

5. COMPLETE CODE:
-----------------
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("sana_ai.personality.response_style")


@dataclass(frozen=True)
class ResponseStyle:
    """
    Immutable response formatting and explanation style guidelines.

    Attributes:
        formatting (str): Markdown hierarchy rules.
        explanation_style (str): Pedagogical breakdown approach.
        code_style (str): Code snippet presentation rules.
        conciseness (str): Length and clarity standards.
    """
    formatting: str = "Use clean Markdown headers (##), bold emphasis, and bullet points for scanability."
    explanation_style: str = "Break down complex technical concepts step-by-step with clear examples."
    code_style: str = "Provide well-commented code blocks with exact language identifiers. Explain code logic cleanly."
    conciseness: str = "Deliver structured, direct, and high-density answers. Avoid fluff or repetitive filler text."

    def to_prompt_string(self) -> str:
        """
        Formats response style rules into a text block for System Prompt composition.

        Returns:
            str: Response style block string.
        """
        return (
            f"=== RESPONSE STYLE & FORMATTING ===\n"
            f"• Structure: {self.formatting}\n"
            f"• Explanations: {self.explanation_style}\n"
            f"• Code Generation: {self.code_style}\n"
            f"• Clarity: {self.conciseness}"
        )
