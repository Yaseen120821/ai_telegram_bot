"""
bot/services/command_service.py - Telegram UI Command Registrar Service
=========================================================================

1. PURPOSE:
-----------
Registers bot commands (`/start`, `/help`) directly with Telegram servers using
`Bot.set_my_commands()`. This populates the auto-complete menu when users type `/`.

2. WHY IT EXISTS:
-----------------
Without this service, users have to guess what commands your bot supports.
Setting bot commands gives your bot a professional native UI menu inside Telegram clients.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Called by: `bot/bot.py` inside the `on_startup` hook.
- Uses: `aiogram.Bot` instance and `aiogram.types.BotCommand`.

4. EXECUTION FLOW:
------------------
1. `run_bot()` starts up in `bot/bot.py`.
2. Invokes `setup_bot_commands(bot)`.
3. Calls Telegram API `bot.set_my_commands([BotCommand(...), ...])`.
4. Telegram updates auto-complete list for all users globally.

5. COMMON MISTAKES:
-------------------
- Using uppercase command strings (`/Start` instead of `/start`). Telegram commands MUST be lowercase!
- Forgetting to register this service on startup, leaving the Telegram UI menu empty.

6. DEBUGGING TIPS:
------------------
- If menu commands don't show up in Telegram, restart your Telegram client app.
- Check logs to confirm `setup_bot_commands` executed without network exceptions.

7. TESTING INSTRUCTIONS:
------------------------
- Open your bot in Telegram and type `/` in the message bar.
- Confirm auto-complete popup shows `/start` and `/help` with descriptions.
"""

import logging
from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeDefault

logger = logging.getLogger("sana_ai.service.command")

async def setup_bot_commands(bot: Bot) -> None:
    """
    Registers Telegram UI menu commands with descriptions.

    Args:
        bot (Bot): Active aiogram Bot instance.
    """
    commands = [
        BotCommand(
            command="start",
            description="🚀 Start interacting with SANA AI Assistant"
        ),
        BotCommand(
            command="help",
            description="❓ View help guide, available features & usage info"
        ),
        BotCommand(
            command="clear",
            description="🧹 Clear active conversation history for this session"
        ),
        BotCommand(
            command="forget",
            description="🗑️ Permanently delete stored long-term memory facts"
        ),
    ]

    try:
        await bot.set_my_commands(
            commands=commands,
            scope=BotCommandScopeDefault()
        )
        logger.info("Successfully registered bot commands in Telegram UI menu.")
    except Exception as e:
        logger.error(f"Failed to register Telegram bot commands: {e}", exc_info=True)
