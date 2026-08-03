"""
app/tools/integration/context_builder.py - Priority Context Assembly & Token Budgeter
======================================================================================
Assembles multi-source context components (System Persona, User Memories, Emotion,
RAG Documents, Tool Execution Results, Conversation History) into token-budgeted
PromptContext payloads following strict priority ordering.
"""

import logging
from typing import Dict, Any, List, Optional
from app.tools.integration.integration_models import IntegratedContext, PromptContext
from app.tools.integration.integration_config import get_pipeline_config

logger = logging.getLogger("sana_ai.tools.integration.context_builder")


class ContextBuilder:
    """
    Priority Context Assembly Engine.
    
    Priority Hierarchy:
    1. System Persona & Identity
    2. Recalled User Memories (Long-Term Facts)
    3. Emotion Communication Directives
    4. RAG Knowledge Documents
    5. Executed Tool Results
    6. Conversation History
    7. Current User Message
    """

    def __init__(self):
        self.config = get_pipeline_config()

    def assemble_prompt_context(
        self,
        integrated_context: IntegratedContext,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> PromptContext:
        """
        Assembles integrated multi-source context into token-budgeted PromptContext.
        """
        raw_query = integrated_context.raw_query

        # Extract context blocks
        mem_str = integrated_context.memory_context
        emo_directive = None
        if integrated_context.emotion_context:
            emo_directive = f"EMOTIONAL STATE: {getattr(integrated_context.emotion_context, 'primary_emotion', 'neutral')}"

        rag_str = integrated_context.rag_context

        tool_str = None
        if integrated_context.tool_context and integrated_context.tool_context.formatted_block:
            tool_str = integrated_context.tool_context.formatted_block

        vision_str = integrated_context.vision_context

        hist = history or []

        # Calculate rough token estimation (1 token ~ 4 chars)
        total_chars = (
            len(system_prompt or "") +
            len(mem_str or "") +
            len(emo_directive or "") +
            len(rag_str or "") +
            len(tool_str or "") +
            len(vision_str or "") +
            sum(len(m.get("content", "")) for m in hist) +
            len(raw_query)
        )
        estimated_tokens = total_chars // 4

        # Enforce Token Budget Truncation if necessary
        budget = self.config.prompt_token_budget
        if estimated_tokens > budget:
            logger.warning(f"Context estimate ({estimated_tokens} tokens) exceeds budget ({budget}). Truncating conversation history...")
            hist = hist[-3:] # Keep last 3 conversation turns

        p_ctx = PromptContext(
            system_prompt=system_prompt or "",
            user_input=raw_query,
            memory_context=mem_str,
            emotion_directive=emo_directive,
            rag_context=rag_str,
            tool_context=tool_str,
            vision_context=vision_str,
            history=hist,
            token_estimate=estimated_tokens
        )

        logger.debug(f"Assembled PromptContext [Est. Tokens: {estimated_tokens} | History turns: {len(hist)}]")
        return p_ctx
