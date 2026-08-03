r"""
app/tools/integration/intelligent_pipeline.py - Intelligent Decision & Execution Pipeline
========================================================================================
Main facade of SANA AI's reasoning layer. Orchestrates Conversation Manager,
Memory Manager, Emotion Manager, Decision Engine, Tool Execution Engine,
RAG Manager, Prompt Builder, and Generator into a unified intelligence pipeline.
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any
from app.tools.routing.tool_router import ToolRouter
from app.tools.routing.router_models import RoutingDecision, RoutingMode
from app.tools.execution.executor_manager import ExecutionManager
from app.tools.execution.execution_models import ExecutionRequest, ExecutionResult
from app.tools.integration.integration_models import (
    ToolContext,
    IntegratedContext,
    DecisionTrace,
    ExecutionSummary,
    CombinedResponse
)
from app.tools.integration.integration_config import get_pipeline_config
from app.tools.integration.context_builder import ContextBuilder
from app.llm.prompt_builder import PromptBuilder

logger = logging.getLogger("sana_ai.tools.integration.pipeline")


class IntelligentPipeline:
    """
    Unified Intelligent Execution Pipeline for SANA AI.
    
    Decision Architecture:
    User Message -> Memory Retrieval -> Emotion Analysis -> Decision Engine
        ├── Multimodal Vision (if image attachments)
        ├── Tool Execution (if Tool intent)
        ├── RAG Retrieval (if RAG intent)
        └── Direct LLM Response
    -> Context Assembly -> Prompt Builder -> Qwen Generator -> Response
    """

    _instance: Optional["IntelligentPipeline"] = None

    def __init__(
        self,
        router: Optional[ToolRouter] = None,
        execution_manager: Optional[ExecutionManager] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.config = get_pipeline_config()
        self.router = router or ToolRouter()
        self.execution_manager = execution_manager or ExecutionManager.get_instance()
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    @classmethod
    def get_instance(cls) -> "IntelligentPipeline":
        """Returns global IntelligentPipeline singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_query(
        self,
        query: str,
        image_paths: Optional[List[str]] = None,
        user_id: str = "default_user",
        conversation_id: str = "default_session",
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        emotion_context: Optional[Any] = None,
        user_confirmed: bool = False,
        generator_callback: Optional[Any] = None
    ) -> CombinedResponse:
        """
        Processes an incoming query through the complete intelligent decision pipeline.
        Supports multimodal image inputs, memory retrieval, emotion guidance, RAG, and tools.
        """
        start_time = time.perf_counter()
        query_text = query.strip() if query and query.strip() else "Analyze attached media context."
        logger.info(f"⚡ IntelligentPipeline processing query: '{query_text}' [User: {user_id}] [Images: {len(image_paths) if image_paths else 0}]")

        # 0. Process Multimodal Vision & Domain Analysis if image attachments exist
        vision_context_str: Optional[str] = None
        if image_paths:
            logger.info(f"👁️ Pipeline processing {len(image_paths)} attached image(s) via Vision AI...")
            try:
                from app.vision.vision_manager import VisionManager
                vision_context_str = VisionManager.get_instance().process_multimodal_images(image_paths)
            except Exception as v_err:
                logger.error(f"Vision AI processing error: {v_err}")
                vision_context_str = f"=== ATTACHED IMAGE CONTEXT ===\nImages attached ({len(image_paths)} files), but vision processing encountered an error.\n=== END IMAGE CONTEXT ==="

        debug_multimodal = os.getenv("DEBUG_MULTIMODAL", "False").lower() in ("true", "1", "yes")
        if debug_multimodal and vision_context_str:
            logger.info(f"\n==================== [DEBUG MULTIMODAL: VISION CONTEXT] ====================\n{vision_context_str}\n==========================================================================")

        # 1. Routing Decision
        decision, tool_request = self.router.route_query(
            query=query_text,
            user_id=user_id,
            user_confirmed=user_confirmed
        )

        # 2. Tool Execution (if required)
        tool_context_obj: Optional[ToolContext] = None
        executed_tools: List[str] = []
        if decision.should_call_tool and decision.selected_tool:
            tool_name = decision.selected_tool.name
            logger.info(f"🛠️ Pipeline executing tool: '{tool_name}'...")
            exec_req = ExecutionRequest(
                tool_name=tool_name,
                parameters=decision.extracted_parameters,
                user_id=user_id,
                conversation_id=conversation_id,
                user_confirmed=user_confirmed
            )
            exec_res: ExecutionResult = self.execution_manager.execute_request(exec_req)
            executed_tools.append(tool_name)
            tool_context_obj = ToolContext(
                results=[exec_res],
                formatted_block=exec_res.formatted_output,
                total_execution_time_ms=exec_res.execution_time_ms
            )

        # 3. RAG Knowledge Retrieval (if required or always if documents exist)
        rag_context_str: Optional[str] = None
        try:
            from app.rag import RAGManager
            rag_res = RAGManager.get_instance().retrieve(query_text, user_id=str(user_id))
            if rag_res and rag_res.chunks:
                rag_ctx_obj = RAGManager.get_instance().build_context(query_text, rag_res.chunks)
                rag_context_str = rag_ctx_obj.formatted_text
        except Exception as rag_err:
            logger.warning(f"Could not retrieve RAG knowledge context: {rag_err}")

        # 4. Handle Clarification Mode
        if decision.routing_mode == RoutingMode.CLARIFICATION_REQUIRED and decision.clarification_prompt:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            int_ctx = IntegratedContext(
                raw_query=query_text, user_id=user_id, conversation_id=conversation_id,
                memory_context=memory_context, emotion_context=emotion_context,
                rag_context=rag_context_str, tool_context=tool_context_obj,
                vision_context=vision_context_str, image_paths=image_paths or [],
                decision=decision
            )
            trace = DecisionTrace(
                query=query_text, routing_decision=decision,
                pipeline_execution_time_ms=elapsed_ms
            )
            summary = ExecutionSummary(query=query_text, answered_directly=True, total_latency_ms=elapsed_ms)
            return CombinedResponse(
                response_text=decision.clarification_prompt,
                summary=summary, integrated_context=int_ctx, trace=trace
            )

        # 5. Assemble Priority Integrated Context
        int_ctx = IntegratedContext(
            raw_query=query_text,
            user_id=user_id,
            conversation_id=conversation_id,
            memory_context=memory_context,
            emotion_context=emotion_context,
            rag_context=rag_context_str,
            tool_context=tool_context_obj,
            vision_context=vision_context_str,
            image_paths=image_paths or [],
            decision=decision
        )

        # 6. Build Token-Budgeted Prompt Context
        p_ctx = self.context_builder.assemble_prompt_context(
            integrated_context=int_ctx,
            history=history
        )

        if debug_multimodal:
            logger.info(f"\n==================== [DEBUG MULTIMODAL: GENERATOR INPUT] ====================\nUser Input: {query_text}\nVision Context Present: {bool(p_ctx.vision_context)}\nTool Context Present: {bool(p_ctx.tool_context)}\nToken Estimate: {p_ctx.token_estimate}\n==========================================================================")

        # 7. Generate Response via TextGenerator
        if generator_callback:
            formatted_prompt = self.prompt_builder.build_prompt(
                user_input=query_text,
                history=p_ctx.history,
                memory_context=p_ctx.memory_context,
                emotion_context=emotion_context,
                rag_context=p_ctx.rag_context,
                tool_context=p_ctx.tool_context,
                image_context=p_ctx.vision_context
            )
            response_text = generator_callback(formatted_prompt)
        else:
            from app.llm import TextGenerator
            generator = TextGenerator()
            response_text = generator.generate_response(
                user_input=query_text,
                history=p_ctx.history,
                memory_context=p_ctx.memory_context,
                emotion_context=emotion_context,
                rag_context=p_ctx.rag_context,
                tool_context=p_ctx.tool_context,
                image_context=p_ctx.vision_context
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        trace = DecisionTrace(
            query=query_text,
            routing_decision=decision,
            recalled_memories_count=1 if memory_context else 0,
            retrieved_rag_chunks_count=1 if rag_context_str else 0,
            tools_executed_count=len(executed_tools),
            emotion_detected=getattr(emotion_context, 'primary_emotion', 'neutral') if emotion_context else 'neutral',
            pipeline_execution_time_ms=elapsed_ms
        )

        summary = ExecutionSummary(
            query=query_text,
            answered_directly=not bool(executed_tools or rag_context_str),
            used_memory=bool(memory_context),
            used_rag=bool(rag_context_str),
            used_tool=bool(executed_tools),
            tools_used=executed_tools,
            total_latency_ms=elapsed_ms
        )

        logger.info(f"✅ IntelligentPipeline completed in {elapsed_ms:.2f}ms [Tools: {executed_tools}]")

        return CombinedResponse(
            response_text=response_text,
            summary=summary,
            integrated_context=int_ctx,
            trace=trace
        )
