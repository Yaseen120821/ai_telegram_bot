r"""
app/llm/generator.py - LLM Text Generation & Inference Orchestrator
=====================================================================

1. PURPOSE:
-----------
Orchestrates text generation using the Qwen model. Takes user input, constructs formatted prompts,
tokenizes input tensors, executes `model.generate()` with inference hyperparameters, decodes output tokens,
cleans the response, and logs performance metrics (latency and tokens/second).

2. MULTIMODAL & TOOL SUPPORT:
------------------------------
Accepts history, memory_context, empathy_directive, emotion_context, rag_context, tool_context, and image_context,
passing them to `PromptBuilder.build_prompt()` to construct complete ChatML prompts.
"""

import time
import logging
from typing import Optional, Dict, Any, List
import torch

from app.llm.model_loader import ModelLoader
from app.llm.prompt_builder import PromptBuilder
from app.llm.response_formatter import ResponseFormatter

logger = logging.getLogger("sana_ai.llm.generator")


class TextGenerator:
    """
    Inference orchestrator for local Qwen LLM.
    """

    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> None:
        """
        Initializes the TextGenerator with default inference settings.
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.prompt_builder = PromptBuilder()

    def generate_response(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_context: Optional[str] = None,
        empathy_directive: Optional[str] = None,
        emotion_context: Optional[Any] = None,
        rag_context: Optional[str] = None,
        tool_context: Optional[str] = None,
        image_context: Optional[str] = None,
        override_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates an AI response for the provided user message.

        Args:
            user_input (str): Raw text message from user.
            system_prompt (Optional[str]): Optional custom system instructions.
            history (Optional[List[Dict[str, str]]]): Conversation history message list.
            memory_context (Optional[str]): Long-term user memories context block.
            empathy_directive (Optional[str]): Empathy prompt directive based on detected emotion.
            emotion_context (Optional[Any]): Active EmotionContext dataclass instance.
            rag_context (Optional[str]): RAG Knowledge Context block.
            tool_context (Optional[str]): Executed tool output context block.
            image_context (Optional[str]): Multimodal image visual analysis & OCR text block.
            override_params (Optional[Dict[str, Any]]): Optional inference parameter overrides.

        Returns:
            str: Cleaned AI response string.
        """
        prompt_text = user_input.strip() if user_input and user_input.strip() else "Analyze the attached context."

        # 1. Retrieve Singleton Model and Tokenizer
        loader = ModelLoader.get_instance()
        model, tokenizer = loader.get_model_and_tokenizer()

        # 2. Build Formatted Prompt with all multimodal, RAG, Memory, Emotion & Tool contexts
        formatted_prompt = self.prompt_builder.build_prompt(
            user_input=prompt_text,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            history=history,
            memory_context=memory_context,
            empathy_directive=empathy_directive,
            emotion_context=emotion_context,
            rag_context=rag_context,
            tool_context=tool_context,
            image_context=image_context
        )

        # 3. Tokenize Input Text
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(loader.device)
        attention_mask = inputs.attention_mask.to(loader.device) if "attention_mask" in inputs else None

        input_token_count = input_ids.shape[1]
        logger.info(f"⚡ Starting Qwen inference | Input Prompt: {input_token_count} tokens | Device: '{loader.device.upper()}'")

        # 4. Resolve Inference Generation Parameters
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }

        if override_params:
            params.update(override_params)

        start_time = time.perf_counter()

        # 5. Execute PyTorch Generation without Gradient Tracking
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **params
                )

            elapsed_time = time.perf_counter() - start_time

            # 6. Extract Only Newly Generated Tokens (Slice out prompt tokens)
            generated_tokens = output_ids[0][input_token_count:]
            output_token_count = len(generated_tokens)

            # 7. Decode Tokens to String
            raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # 8. Clean and Format Response
            cleaned_response = ResponseFormatter.clean_response(
                raw_text=raw_response,
                original_prompt=formatted_prompt
            )

            # 9. Compute & Log Benchmarks
            tokens_per_sec = output_token_count / elapsed_time if elapsed_time > 0 else 0
            logger.info(
                f"✅ Inference Complete | Time: {elapsed_time:.2f}s | "
                f"Generated: {output_token_count} tokens ({tokens_per_sec:.2f} tok/s)"
            )

            return cleaned_response

        except torch.cuda.OutOfMemoryError as oom_err:
            logger.critical("🚨 CUDA Out Of Memory Error during generation! Clearing cache...", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "⚠️ System Warning: GPU memory limit reached while generating response. Please try a shorter prompt."

        except Exception as exc:
            logger.error(f"❌ Error during Qwen text generation: {exc}", exc_info=True)
            return f"⚠️ An error occurred while generating AI response: {str(exc)}"
