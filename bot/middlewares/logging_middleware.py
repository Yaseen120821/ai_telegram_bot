"""
bot/middlewares/logging_middleware.py - Telemetry & Event Audit Middleware
===========================================================================

1. PURPOSE:
-----------
Intercepts every incoming Telegram message/update before reaching handler functions.
Logs request metadata (User ID, Username, Chat ID, Message Text) and processing duration.

2. WHY IT EXISTS:
-----------------
In production software, observability is essential. Without middleware logging,
you wouldn't know who is using the bot, what commands are failing, or how long
handlers take to respond. Middlewares allow centralized request inspection without
duplicating log lines across every single handler.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Registered in `bot/bot.py` via `dp.message.outer_middleware(StructLoggingMiddleware())`.
- Wraps execution of handlers inside `bot/handlers/common.py` and `bot/handlers/echo.py`.

4. EXECUTION FLOW:
------------------
1. Incoming Telegram `Update` hits `aiogram` Dispatcher.
2. `StructLoggingMiddleware.__call__()` triggers BEFORE the handler.
3. Logs user details and message snippet.
4. Invokes `await handler(event, data)` to run the target handler.
5. Captures execution end time and logs completion latency.

5. COMMON MISTAKES:
-------------------
- Forgetting to `await handler(event, data)`, which freezes update handling completely!
- Swallowing exceptions inside middleware without re-raising or logging them.
- Logging sensitive personal information in unencrypted plain logs.

6. DEBUGGING TIPS:
------------------
- If logs aren't appearing, verify middleware is registered on `dp.message.outer_middleware`.
- Ensure log level in `.env` is set to `INFO` or `DEBUG`.

7. TESTING INSTRUCTIONS:
------------------------
- Send any message to the bot on Telegram.
- Check console output or `logs/bot.log` to see user ID, text, and latency metrics.
"""

import time
import logging
from typing import Callable, Dict, Any, Awaitable
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Message

logger = logging.getLogger("sana_ai.middleware")

class StructLoggingMiddleware(BaseMiddleware):
    """
    Asynchronous Middleware for logging incoming Telegram messages and execution latency.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        """
        Main middleware interceptor method.

        Args:
            handler: Next middleware or event handler coroutine.
            event: Incoming Telegram update event object.
            data: Contextual data dictionary passed by aiogram.

        Returns:
            Any: Result of the downstream handler execution.
        """
        if isinstance(event, Message):
            user = event.from_user
            user_info = f"User[id={user.id}, username=@{user.username or 'N/A'}]" if user else "User[Unknown]"
            chat_info = f"Chat[id={event.chat.id}, type={event.chat.type}]"
            text_snippet = (event.text[:50] + '...') if event.text and len(event.text) > 50 else (event.text or "<Non-Text>")

            logger.info(f"📥 INCOMING MESSAGE | {user_info} | {chat_info} | Text: '{text_snippet}'")
            
            start_time = time.perf_counter()
            try:
                # Delegate control to downstream middleware / handler
                result = await handler(event, data)
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.info(f"📤 PROCESSED MESSAGE | {user_info} | Latency: {execution_time:.2f}ms")
                return result
            except Exception as exc:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(f"❌ HANDLER ERROR | {user_info} | Failed after {execution_time:.2f}ms | Error: {exc}", exc_info=True)
                raise exc
        
        # Pass non-Message events straight through
        return await handler(event, data)
