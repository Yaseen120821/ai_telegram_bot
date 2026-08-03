r"""
bot/handlers/echo.py - Telegram AI Chat Router (Text, Photo & Document Multimodal Handler)
========================================================================================

1. PURPOSE:
-----------
Processes incoming plain text messages, photo uploads (`F.photo`), and document uploads (`F.document`)
from Telegram users, routing media through Vision AI (Florence-2 + EasyOCR), PDF Ingestion,
Long-Term Memory, Emotion AI, RAG Knowledge, and the local Qwen LLM.

2. SUPPORTED MEDIA TYPES:
-------------------------
- Plain Text Conversations.
- Photos & Images (PNG, JPG, WEBP, BMP, GIF, TIFF).
- Digital & Scanned PDF Documents (`.pdf`).
- Source Code, Markdown, TXT, DOCX, HTML, JSON, CSV files.
"""

import os
import asyncio
import logging
from typing import Optional, List
from aiogram import Router, F, Bot
from aiogram.types import Message

from app.tools.integration import IntelligentPipeline
from app.conversation import ConversationManager
from app.memory import MemoryManager
from app.emotion import EmotionManager

logger = logging.getLogger("sana_ai.handler.chat")

# Initialize Router for AI chat handler
echo_router = Router(name="echo_router")

# Global singleton managers
_pipeline: IntelligentPipeline = IntelligentPipeline.get_instance()
_conv_manager: ConversationManager = ConversationManager.get_instance()
_memory_manager: MemoryManager = MemoryManager.get_instance()
_emotion_manager: EmotionManager = EmotionManager.get_instance()

TEMP_MEDIA_DIR = os.path.abspath("logs/temp_media")
os.makedirs(TEMP_MEDIA_DIR, exist_ok=True)


@echo_router.message(F.text.startswith("/"))
async def handle_unknown_command(message: Message) -> None:
    """Catch-all handler for unhandled or unknown slash commands."""
    command_text = message.text.split()[0] if message.text else message.text
    logger.warning(f"Unknown command received: '{command_text}' from User ID {message.from_user.id if message.from_user else 'Unknown'}")
    response_text = (
        f"❓ **Unknown Command**: `{command_text}`\n\n"
        "I didn't recognize that command. Type `/help` to see available commands."
    )
    await message.answer(text=response_text, parse_mode="Markdown")


@echo_router.message(F.photo)
async def handle_photo_message(message: Message, bot: Bot) -> None:
    """
    Handles photo uploads (`F.photo`). Downloads image, executes Vision AI & EasyOCR analysis,
    and returns grounded AI response.
    """
    user_id = str(message.from_user.id) if message.from_user else "0"
    user_name = message.from_user.first_name if message.from_user else "User"
    caption_text = message.caption.strip() if message.caption else "Analyze this image and describe its contents."

    logger.info(f"📸 Incoming Photo Update from '{user_name}' (ID: {user_id}) | Caption: '{caption_text}'")

    # 1. Download highest-resolution PhotoSize from Telegram
    try:
        photo_size = message.photo[-1]
        file_info = await bot.get_file(photo_size.file_id)
        ext = os.path.splitext(file_info.file_path)[1] or ".jpg"
        local_filename = f"photo_{user_id}_{message.message_id}{ext}"
        local_path = os.path.join(TEMP_MEDIA_DIR, local_filename)

        await bot.download_file(file_info.file_path, destination=local_path)
        logger.info(f"✅ Photo downloaded successfully to: '{local_path}'")
    except Exception as dl_err:
        logger.error(f"Failed to download photo update: {dl_err}", exc_info=True)
        await message.answer(text="⚠️ Failed to download uploaded photo. Please try again.")
        return

    # 2. Trigger Typing Action
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:
        pass

    # 3. Process via IntelligentPipeline with image path attachment
    await _process_and_reply(
        message=message,
        bot=bot,
        user_id=user_id,
        user_name=user_name,
        prompt_text=caption_text,
        image_paths=[local_path]
    )


@echo_router.message(F.document)
async def handle_document_message(message: Message, bot: Bot) -> None:
    """
    Handles document uploads (`F.document` - PDFs, Images as documents, Code files, TXT, DOCX, CSV, JSON).
    """
    user_id = str(message.from_user.id) if message.from_user else "0"
    user_name = message.from_user.first_name if message.from_user else "User"
    caption_text = message.caption.strip() if message.caption else f"Analyze attached document '{message.document.file_name}'."
    file_name = message.document.file_name or "document.bin"
    mime_type = message.document.mime_type or ""

    logger.info(f"📄 Incoming Document Update from '{user_name}' (ID: {user_id}) | File: '{file_name}' | MIME: '{mime_type}'")

    # 1. Download Document File
    try:
        file_info = await bot.get_file(message.document.file_id)
        local_filename = f"doc_{user_id}_{message.message_id}_{file_name}"
        local_path = os.path.join(TEMP_MEDIA_DIR, local_filename)

        await bot.download_file(file_info.file_path, destination=local_path)
        logger.info(f"✅ Document downloaded successfully to: '{local_path}'")
    except Exception as dl_err:
        logger.error(f"Failed to download document update: {dl_err}", exc_info=True)
        await message.answer(text="⚠️ Failed to download uploaded document. Please try again.")
        return

    # 2. Trigger Typing Action
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:
        pass

    # 3. Differentiate Image Documents vs. PDF & Text Documents
    ext = os.path.splitext(file_name)[1].lower()
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

    if ext in image_exts or mime_type.startswith("image/"):
        logger.info(f"🖼️ Document '{file_name}' classified as Image Document. Routing to Vision AI...")
        await _process_and_reply(
            message=message,
            bot=bot,
            user_id=user_id,
            user_name=user_name,
            prompt_text=caption_text,
            image_paths=[local_path]
        )
        return

    # Process PDF / Text / Code Document
    pdf_context_block = ""
    if ext == ".pdf" or mime_type == "application/pdf":
        logger.info(f"📕 Document '{file_name}' classified as PDF. Routing to PDFAnalyzer...")
        try:
            from app.rag.pdf_analyzer import PDFAnalyzer
            pdf_res = PDFAnalyzer.get_instance().analyze_pdf(local_path)
            pdf_context_block = pdf_res["formatted_context"]
        except Exception as pdf_err:
            logger.error(f"PDFAnalyzer processing error: {pdf_err}")
            pdf_context_block = f"=== ATTACHED PDF DOCUMENT: {file_name} ===\nNotice: Error reading PDF ({pdf_err}).\n=== END ATTACHED PDF DOCUMENT ==="
    else:
        logger.info(f"📄 Document '{file_name}' classified as Text/Code file. Ingesting content...")
        try:
            from app.rag.document_processing.loaders import LoaderFactory
            loader = LoaderFactory.get_loader(ext.lstrip("."))
            raw_doc_text = loader.load(local_path)
            pdf_context_block = f"=== ATTACHED DOCUMENT: {file_name} ===\n{raw_doc_text.strip()}\n=== END ATTACHED DOCUMENT ==="
        except Exception as doc_err:
            logger.error(f"Document loader error: {doc_err}")
            pdf_context_block = f"=== ATTACHED DOCUMENT: {file_name} ===\nNotice: Error reading file ({doc_err}).\n=== END ATTACHED DOCUMENT ==="

    # Also register document into RAGManager for incremental search
    try:
        from app.rag import RAGManager
        RAGManager.get_instance().index_document(local_path)
    except Exception:
        pass

    # Complete pipeline processing with PDF/Document Context
    await _process_and_reply(
        message=message,
        bot=bot,
        user_id=user_id,
        user_name=user_name,
        prompt_text=f"{caption_text}\n\n{pdf_context_block}",
        image_paths=None
    )


@echo_router.message(F.text)
async def handle_ai_chat_message(message: Message, bot: Bot) -> None:
    """Handles plain text messages."""
    user_id = str(message.from_user.id) if message.from_user else "0"
    user_name = message.from_user.first_name if message.from_user else "User"
    user_text = message.text.strip()

    logger.info(f"💬 Incoming User Prompt | User: '{user_name}' (ID: {user_id}) | Prompt: '{user_text}'")

    # Send typing action indicator
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:
        pass

    await _process_and_reply(
        message=message,
        bot=bot,
        user_id=user_id,
        user_name=user_name,
        prompt_text=user_text,
        image_paths=None
    )


async def _process_and_reply(
    message: Message,
    bot: Bot,
    user_id: str,
    user_name: str,
    prompt_text: str,
    image_paths: Optional[List[str]] = None
) -> None:
    """Helper method executing full pipeline, history tracking, offloading, and reply delivery."""
    # 1. Analyze Emotion & retrieve EmotionContext
    _emotion_manager.process_emotion(user_id, prompt_text)
    emotion_context = _emotion_manager.get_user_emotion_context(user_id)

    # 2. Persist Long-Term Memory facts
    try:
        await asyncio.to_thread(_memory_manager.process_and_store_user_message, user_id, prompt_text)
    except Exception:
        pass

    memory_context = _memory_manager.get_memory_context_for_prompt(user_id, query=prompt_text)
    history = _conv_manager.get_formatted_history(user_id)

    # Record user prompt in short-term history
    _conv_manager.add_user_message(user_id, prompt_text)

    # 3. Offload CPU/GPU blocking pipeline processing to worker thread
    try:
        combined_response = await asyncio.to_thread(
            _pipeline.process_query,
            query=prompt_text,
            image_paths=image_paths,
            user_id=user_id,
            conversation_id=str(message.chat.id),
            history=history,
            memory_context=memory_context,
            emotion_context=emotion_context
        )
        ai_response = combined_response.response_text
    except Exception as proc_err:
        logger.error(f"Error during async pipeline execution: {proc_err}", exc_info=True)
        ai_response = "⚠️ I encountered an internal error while processing your request. Please try again."

    # Record assistant answer in history
    _conv_manager.add_assistant_message(user_id, ai_response)

    # Deliver reply to Telegram user
    try:
        await message.answer(text=ai_response, parse_mode="Markdown")
        logger.info(f"📤 Sent AI Response to User ID {user_id} [Length: {len(ai_response)} chars]")
    except Exception as parse_err:
        logger.warning(f"Markdown parsing failed for response ({parse_err}). Falling back to plain text reply.")
        await message.answer(text=ai_response, parse_mode=None)
