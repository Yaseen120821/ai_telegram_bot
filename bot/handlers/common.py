"""
bot/handlers/common.py - Common Command Handlers (/start, /help)
=================================================================

1. PURPOSE:
-----------
Processes baseline user commands (`/start`, `/help`) and quick menu button clicks
("❓ Help", "🤖 About SANA", "⚡ Status").

2. WHY IT EXISTS:
-----------------
Separates system commands from custom feature handlers. In aiogram 3.x, code is
organized into `Router` modules for clean maintenance and scalability.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Exposes: `router = Router(name="common")`.
- Included in: `bot/bot.py` via `dp.include_router(common_router)`.
- Uses: `get_main_reply_keyboard()` from `bot/keyboards/common_keyboards.py`.

4. EXECUTION FLOW:
------------------
1. Telegram message arrives matching `/start` or `CommandStart()`.
2. Router matches criteria and routes execution to `cmd_start()`.
3. `cmd_start()` constructs formatted message and attaches reply keyboard.
4. Returns response asynchronously using `await message.answer(...)`.

5. COMMON MISTAKES:
-------------------
- Blocking the event loop by running heavy synchronous computation inside `async def`.
- Forgetting to register `common_router` with the Dispatcher in `bot/bot.py`.

6. DEBUGGING TIPS:
------------------
- If command doesn't trigger, verify router registration order in `bot/bot.py`.
- Ensure handler filters use exact aiogram `CommandStart()` or `Command("help")`.

7. TESTING INSTRUCTIONS:
------------------------
- Send `/start` -> verify welcome banner & reply buttons.
- Send `/help` -> verify help text menu.
- Click keyboard buttons -> verify corresponding responses.
"""

import logging
from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from bot.keyboards.common_keyboards import get_main_reply_keyboard

logger = logging.getLogger("sana_ai.handler.common")

# Initialize Router for common system commands
common_router = Router(name="common_router")


@common_router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """
    Handles the `/start` command. Sends welcome message & main menu keyboard.

    Args:
        message (Message): Incoming Telegram message object.
    """
    user_name = message.from_user.first_name if message.from_user else "User"
    
    welcome_text = (
        f"👋 **Hello, {user_name}! Welcome to SANA AI** 🤖\n\n"
        "I am your Personal AI Assistant, designed to assist you with intelligent chat, "
        "memory, task automation, and local AI intelligence.\n\n"
        "📌 **Current Status**: Chapter 2 - Telegram Transport Layer Active!\n"
        "💡 Type `/help` or use the menu buttons below to get started."
    )
    
    await message.answer(
        text=welcome_text,
        parse_mode="Markdown",
        reply_markup=get_main_reply_keyboard()
    )


@common_router.message(Command("clear"))
@common_router.message(F.text == "🧹 Clear History")
async def cmd_clear_history(message: Message) -> None:
    """
    Handles the `/clear` command. Clears active conversation history for the requesting user.

    Args:
        message (Message): Incoming Telegram message object.
    """
    user_id = message.from_user.id if message.from_user else 0
    from app.conversation import ConversationManager
    
    cleared = ConversationManager.get_instance().clear_user_history(user_id)
    if cleared:
        reply_text = "🧹 **Conversation History Cleared!**\n\nSANA AI has reset your chat context."
    else:
        reply_text = "ℹ️ You had no active conversation history to clear."

    await message.answer(text=reply_text, parse_mode="Markdown")


@common_router.message(Command("forget"))
@common_router.message(F.text == "🗑️ Forget Memory")
async def cmd_forget_memory(message: Message) -> None:
    """
    Handles the `/forget` command. Permanently deletes stored long-term memory for the requesting user.

    Args:
        message (Message): Incoming Telegram message object.
    """
    user_id = message.from_user.id if message.from_user else 0
    from app.memory import MemoryManager
    
    deleted_count = MemoryManager.get_instance().clear_user_memories(user_id)
    if deleted_count > 0:
        reply_text = f"🗑️ **Long-Term Memory Wiped!**\n\nPermanently deleted {deleted_count} stored facts for your profile from SQLite."
    else:
        reply_text = "ℹ️ You had no stored long-term memory records to delete."

    await message.answer(text=reply_text, parse_mode="Markdown")


@common_router.message(Command("help"))
@common_router.message(F.text == "❓ Help")
async def cmd_help(message: Message) -> None:
    """
    Handles the `/help` command and '❓ Help' keyboard button.

    Args:
        message (Message): Incoming Telegram message object.
    """
    help_text = (
        "📖 **SANA AI - Command & Feature Guide**\n\n"
        "🔹 `/start` - Restart the bot session & show main menu\n"
        "🔹 `/help` - Show this usage guide\n"
        "🔹 `/clear` - Clear active short-term conversation history\n"
        "🔹 `/forget` - Permanently wipe all stored long-term memories from SQLite\n"
        "🔹 Send any text - Chat naturally with SANA AI (Multi-turn & Long-Term Memory active!)\n\n"
        "⚙️ **System Architecture Layer**:\n"
        "• Telegram Bot Layer: **ACTIVE** (aiogram 3.x)\n"
        "• AI Model Layer (Qwen): **ACTIVE** (Local CUDA/CPU)\n"
        "• Short-Term Conversation: **ACTIVE** (In-Memory per-user session)\n"
        "• Long-Term Memory: **ACTIVE** (Durable SQLite Storage)"
    )
    
    await message.answer(
        text=help_text,
        parse_mode="Markdown",
        reply_markup=get_main_reply_keyboard()
    )


@common_router.message(F.text == "🤖 About SANA")
async def btn_about(message: Message) -> None:
    """
    Handles the '🤖 About SANA' reply keyboard button click.
    """
    about_text = (
        "🤖 **About SANA AI**\n\n"
        "SANA AI is an advanced, privacy-focused Personal AI Assistant powered by local LLM "
        "architecture, asynchronous Telegram pipelines, and state-of-the-art Python engineering."
    )
    await message.answer(text=about_text, parse_mode="Markdown")


@common_router.message(F.text == "⚡ Status")
async def btn_status(message: Message) -> None:
    """
    Handles the '⚡ Status' reply keyboard button click.
    """
    status_text = (
        "⚡ **SANA AI System Health Status**\n\n"
        "🟢 **Telegram Connection**: Connected & Polling\n"
        "🟢 **Async Event Loop**: Operational\n"
        "🟢 **Logging Engine**: Active (`logs/bot.log`)\n"
        "⚪ **Local Qwen LLM**: Standby (Chapter 3)"
    )
    await message.answer(text=status_text, parse_mode="Markdown")
