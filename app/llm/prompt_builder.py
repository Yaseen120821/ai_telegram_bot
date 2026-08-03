r"""
app/llm/prompt_builder.py - Prompt Construction & ChatML Formatting Engine
==========================================================================

1. PURPOSE:
-----------
Formats raw user messages into structured ChatML prompt strings compatible with Qwen's ChatML
template format (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n`).

2. WHY IT EXISTS:
-----------------
Enforces strict System Prompt assembly order:
System Persona & Identity -> Relevant User Memories -> Emotion Context -> RAG Knowledge -> History -> User Message.

3. RESPONSIBILITIES:
--------------------
- Construct role-based prompt templates for Qwen local model inference.
- Enforce strict priority hierarchy placing User Memories before RAG Knowledge Documents.
- Disambiguate Assistant Name (SANA AI) and Creator (Mohameed Yaseen) from User Identity.

4. COMMUNICATION WITH OTHER MODULES:
------------------------------------
- Called by `app/llm/generator.py` prior to tokenization.

5. COMPLETE CODE:
-----------------
"""

import logging
from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer

logger = logging.getLogger("sana_ai.llm.prompt_builder")

DEFAULT_SYSTEM_PROMPT = (
    "You are SANA AI, a helpful, polite, and intelligent personal AI assistant created and engineered by Mohameed Yaseen. "
    "Provide clear, direct, and concise answers to the user's questions. "
    "Your name is SANA AI. Your creator is Mohameed Yaseen. The human interacting with you is the USER. "
    "Never confuse your name (SANA AI) with the user's name. "
    "If the user asks 'What is my name?', check the recalled user memories below. If unknown, say 'I don't know your name yet.'"
)


class PromptBuilder:
    """
    Constructs role-based prompt templates for Qwen local model inference.
    """

    def __init__(self, default_system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        """
        Initializes the PromptBuilder.

        Args:
            default_system_prompt (str): Base system instructions for SANA AI assistant.
        """
        self.default_system_prompt = default_system_prompt

    def build_prompt(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        empathy_directive: Optional[str] = None,
        emotion_context: Optional[Any] = None,
        rag_context: Optional[str] = None,
        tool_context: Optional[str] = None,
        image_context: Optional[str] = None
    ) -> str:
        """
        Constructs a complete ChatML prompt for Qwen inference following strict priority order:
        Persona -> User Memory -> Emotion Guidance -> RAG Knowledge -> Tool Results -> Image Context -> History -> User Input.
        """
        if not system_prompt:
            try:
                from app.personality import PersonalityManager
                sys_prompt = PersonalityManager.get_instance().get_system_prompt()
            except Exception as p_err:
                logger.warning(f"Could not load PersonalityManager system prompt ({p_err}). Using default.")
                sys_prompt = self.default_system_prompt
        else:
            sys_prompt = system_prompt

        # 1. Append long-term memory context to system prompt (PRIORITY: User facts come first!)
        if memory_context and memory_context.strip():
            sys_prompt = f"{sys_prompt}\n\n{memory_context.strip()}"

        # 2. Append adaptive communication guidance from EmotionContext if present
        if emotion_context is not None:
            try:
                from app.personality import PersonalityManager
                adaptive_style = PersonalityManager.get_instance().get_adaptive_communication_style(emotion_context)
                if adaptive_style and adaptive_style.strip():
                    sys_prompt = f"{sys_prompt}\n\n{adaptive_style.strip()}"
            except Exception as e_err:
                logger.warning(f"Could not append adaptive communication style: {e_err}")
        elif empathy_directive and empathy_directive.strip():
            sys_prompt = f"{sys_prompt}\n\n=== USER EMOTIONAL STATE & EMPATHY DIRECTIVE ===\n{empathy_directive.strip()}"

        # 3. Append RAG Knowledge Context to system prompt
        if rag_context and rag_context.strip():
            sys_prompt = f"{sys_prompt}\n\n{rag_context.strip()}"

        # 4. Append Executed Tool Results Context to system prompt
        if tool_context and tool_context.strip():
            sys_prompt = f"{sys_prompt}\n\n=== EXECUTED TOOL RESULTS ===\n{tool_context.strip()}"

        # 5. Append Multimodal Image Context & Domain Analysis to system prompt
        if image_context and image_context.strip():
            sys_prompt = f"{sys_prompt}\n\n{image_context.strip()}"

        messages: List[Dict[str, str]] = []

        # Add System Prompt
        if sys_prompt and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt.strip()})

        # Add Conversation History
        if history:
            messages.extend(history)

        # Add Current User Input if not already the last message in history
        if not history or history[-1].get("content", "").strip() != user_input.strip():
            messages.append({"role": "user", "content": user_input.strip()})

        # Use Tokenizer Chat Template if available
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.debug(f"Prompt constructed via tokenizer.apply_chat_template [Length: {len(formatted_prompt)} chars]")
                return formatted_prompt
            except Exception as e:
                logger.warning(f"Failed to use tokenizer.apply_chat_template: {e}. Falling back to manual ChatML format.")

        # Fallback Manual ChatML Construction (Qwen standard format)
        formatted_prompt = self._build_chatml_fallback(messages)
        logger.debug(f"Prompt constructed via manual ChatML fallback [Length: {len(formatted_prompt)} chars]")
        return formatted_prompt

    def _build_chatml_fallback(self, messages: List[Dict[str, str]]) -> str:
        """
        Manual fallback ChatML formatting:
        <|im_start|>system
        ...<|im_end|>
        <|im_start|>user
        ...<|im_end|>
        <|im_start|>assistant
        """
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        # Append assistant trigger tag
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)
