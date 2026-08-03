"""
app/personality/templates.py - Prompt Engineering & Composition Engine
========================================================================

1. PURPOSE:
-----------
Provides prompt composition templates and teaches fundamental Prompt Engineering paradigms
(Role Prompting, Instruction Prompting, Zero-Shot, Few-Shot, Chain-of-Thought).

2. PROMPT ENGINEERING PARADIGMS EXPLAINED:
-------------------------------------------
- **Role Prompting**: Assigning a specific identity, expert domain, and persona (e.g. "You are SANA AI...").
- **Instruction Prompting**: Providing explicit instructions regarding output layout, language, and constraints.
- **Zero-Shot Prompting**: Directing the LLM to solve a task without providing demonstration examples.
- **Few-Shot Prompting**: Providing 1-3 exemplar input/output pairs inside the system or user prompt to anchor formatting.
- **Chain-of-Thought (CoT)**: Instructing the model to "think step-by-step" before providing a final answer.
  (Improves reasoning accuracy on math, logic, and code problems).

3. RESPONSIBILITIES:
--------------------
- Provide clean utility methods for string composition and template formatting.
- Format multi-role message arrays cleanly.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Used by `app/personality/personality_manager.py` and `app/llm/prompt_builder.py`.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("sana_ai.personality.templates")


class PromptTemplate:
    """
    Template composition helper for building ChatML role message arrays.
    """

    @staticmethod
    def compose_messages(
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        user_input: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Combines System Prompt, Conversation History, and User Input into a unified ChatML message list.

        Args:
            system_prompt (str): Assembled System Prompt directive.
            history (Optional[List[Dict[str, str]]]): Multi-turn conversation history.
            user_input (Optional[str]): Current incoming user prompt.

        Returns:
            List[Dict[str, str]]: Unified list of role/content dictionaries.
        """
        messages: List[Dict[str, str]] = []

        # 1. System Prompt Block
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        # 2. Multi-turn Conversation History Block
        if history:
            messages.extend(history)

        # 3. Current User Input (if provided and not already last message in history)
        if user_input and user_input.strip():
            if not history or history[-1].get("content", "").strip() != user_input.strip():
                messages.append({"role": "user", "content": user_input.strip()})

        return messages
