"""
app/personality/system_prompt.py - Master System Prompt Composition Engine
============================================================================

1. PURPOSE:
-----------
Compiles `Identity`, `Behavior`, `ResponseStyle`, and `Rules` into a unified, structured master System Prompt string.

2. SYSTEM PROMPT CONCEPTS EXPLAINED:
------------------------------------
- **System Prompt**: Top-level directive embedded within ChatML role tags (`<|im_start|>system...<|im_end|>`).
  Establishes global persona, constraints, and instructions BEFORE the model reads any user input.
- **System Prompt vs User Prompt vs Assistant Message**:
  * System Prompt: Defines WHO the model is and HOW it must behave.
  * User Prompt: Defines WHAT the user is currently asking.
  * Assistant Message: Stores WHAT the model answered previously in the conversation loop.
- **Prompt Hierarchy**: Instruction-tuned models are trained via RLHF to prioritize System Prompt guidelines
  over user prompts, preventing prompt injection and persona drift.

3. RESPONSIBILITIES:
--------------------
- Combine all personality sub-components into a clean, human-readable master instruction block.
- Support optional custom instruction extensions for specialized operational modes.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Uses `Identity`, `Behavior`, `ResponseStyle`, `Rules` from their respective files.
- Managed by `app/personality/personality_manager.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import Optional

from app.personality.identity import Identity
from app.personality.behavior import Behavior
from app.personality.response_style import ResponseStyle
from app.personality.rules import Rules

logger = logging.getLogger("sana_ai.personality.system_prompt")


class SystemPromptBuilder:
    """
    Assembles structured personality components into a master System Prompt.
    """

    def __init__(
        self,
        identity: Optional[Identity] = None,
        behavior: Optional[Behavior] = None,
        response_style: Optional[ResponseStyle] = None,
        rules: Optional[Rules] = None
    ) -> None:
        """
        Initializes SystemPromptBuilder with personality component instances.

        Args:
            identity (Optional[Identity]): Identity specification instance.
            behavior (Optional[Behavior]): Behavioral guidelines instance.
            response_style (Optional[ResponseStyle]): Response style guidelines instance.
            rules (Optional[Rules]): Boundary rules instance.
        """
        self.identity: Identity = identity or Identity()
        self.behavior: Behavior = behavior or Behavior()
        self.response_style: ResponseStyle = response_style or ResponseStyle()
        self.rules: Rules = rules or Rules()

    def build_system_prompt(self, custom_instructions: Optional[str] = None) -> str:
        """
        Compiles all personality blocks into a master System Prompt string.

        Args:
            custom_instructions (Optional[str]): Additional mode-specific instructions.

        Returns:
            str: Full assembled System Prompt string.
        """
        sections = [
            self.identity.to_prompt_string(),
            self.behavior.to_prompt_string(),
            self.response_style.to_prompt_string(),
            self.rules.to_prompt_string()
        ]

        if custom_instructions and custom_instructions.strip():
            sections.append(f"=== SPECIAL INSTRUCTIONS ===\n{custom_instructions.strip()}")

        master_prompt = "\n\n".join(sections)
        logger.debug(f"Compiled Master System Prompt [Length: {len(master_prompt)} chars]")
        return master_prompt
