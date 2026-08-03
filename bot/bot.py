"""
bot/bot.py - Core Telegram Bot Runner & Event Loop Dispatcher
==============================================================

1. PURPOSE:
-----------
Initializes logging, configuration, `aiogram.Bot` instance, `Dispatcher`, middlewares,
routers, commands menu, pre-loads the Qwen LLM into memory during startup, and starts long polling.

2. WHY IT EXISTS:
-----------------
Acts as the central orchestrator and lifecycle manager for the Telegram bot service.
It glues all sub-components (handlers, middlewares, AI engine) into a unified application.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Loads config from `bot/config.py`.
- Pre-loads Qwen LLM via `app/llm/model_loader.py`.
- Registers middlewares from `bot/middlewares/logging_middleware.py`.
- Registers routers from `bot/handlers/common.py` and `bot/handlers/echo.py`.
- Invokes UI command registrar from `bot/services/command_service.py`.
- Called by: `main.py`.

4. EXECUTION FLOW:
------------------
1. `setup_logging()` creates log outputs in console and `logs/bot.log`.
2. `Config.load_from_env()` loads configuration.
3. `Bot` & `Dispatcher` instances are instantiated.
4. Outer logging middleware attached to `dp.message`.
5. `common_router` and `echo_router` registered with `dp`.
6. Startup hook:
   a. Sets Telegram UI commands menu (`/start`, `/help`).
   b. Pre-loads Qwen model resident into memory via `ModelLoader`.
7. `dp.start_polling(bot)` opens asynchronous polling loop.
8. Graceful teardown closes HTTP session on shutdown.
"""

import sys
import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import ErrorEvent

from bot.config import Config
from bot.middlewares.logging_middleware import StructLoggingMiddleware
from bot.services.command_service import setup_bot_commands
from bot.handlers.common import common_router
from bot.handlers.echo import echo_router
from app.llm import ModelLoader

logger = logging.getLogger("sana_ai.bot")


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configures dual-destination logging (Console output + logs/bot.log file).

    Args:
        log_level (str): String representation of log level (DEBUG, INFO, etc.).
    """
    os.makedirs("logs", exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(numeric_level)

    # File Handler (logs/bot.log)
    file_handler = logging.FileHandler("logs/bot.log", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(numeric_level)

    # Root Logger Setup
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Avoid duplicate handlers on re-entry
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging initialized successfully at level '{log_level.upper()}'")


async def global_error_handler(event: ErrorEvent) -> None:
    """
    Global exception handler registered with aiogram Dispatcher.
    Catches unhandled errors across all handlers and logs full stack trace.

    Args:
        event (ErrorEvent): Uncaught error event wrapped by aiogram.
    """
    logger.critical(
        f"🚨 UNHANDLED BOT EXCEPTION | Update ID: {event.update.update_id if event.update else 'N/A'} | Error: {event.exception}",
        exc_info=event.exception
    )


async def main_bot() -> None:
    """
    Main asynchronous bot runner function.
    Initializes Bot, Dispatcher, Routers, Middlewares, pre-loads Qwen LLM, and starts long polling.
    """
    # 1. Load Configuration
    config = Config.load_from_env()

    # 2. Setup Logging
    setup_logging(config.log_level)
    logger.info("Initializing SANA AI Telegram Bot & Local AI Layer...")

    # 3. Instantiate Bot and Dispatcher
    bot = Bot(
        token=config.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
    )
    dp = Dispatcher()

    # 4. Register Outer Middlewares
    dp.message.outer_middleware(StructLoggingMiddleware())

    # 5. Register Routers (Order matters! common_router FIRST, echo_router SECOND)
    dp.include_router(common_router)
    dp.include_router(echo_router)

    # 6. Register Global Error Handler
    dp.errors.register(global_error_handler)

    # 7. Register Startup Hook to Warm-Load Qwen Model & Set Telegram Commands Menu
    @dp.startup()
    async def on_startup(bot: Bot) -> None:
        logger.info("Bot startup hook triggered. Registering menu commands...")
        await setup_bot_commands(bot)
        
        logger.info(f"🧠 Warm-loading Qwen model from '{config.model_path}' into memory...")
        start_time = asyncio.get_event_loop().time()
        
        # Offload blocking model disk read and weight loading to worker thread
        await asyncio.to_thread(ModelLoader.get_instance().load_model, config.model_path)
        
        # Warm-load Tool Registry dynamically
        logger.info("🛠️ Initializing Tool Registry...")
        try:
            from app.tools.registry import RegistryManager
            await asyncio.to_thread(RegistryManager.get_instance().initialize)
        except Exception as reg_err:
            logger.warning(f"Tool Registry initialization notice: {reg_err}")

        # Warm-load Vision AI (Florence-2 + EasyOCR) Subsystem
        logger.info("👁️ Warm-loading Vision AI & EasyOCR models into memory...")
        try:
            from app.vision import VisionManager
            await asyncio.to_thread(VisionManager.get_instance)
        except Exception as vis_err:
            logger.warning(f"Vision AI warm-loading notice: {vis_err}")

        warmup_duration = asyncio.get_event_loop().time() - start_time
        logger.info(f"⚡ All Core Systems (Qwen, Vision AI, Tools) are warm and resident in memory (Warmup time: {warmup_duration:.2f}s)!")
        logger.info("🚀 SANA AI Telegram Bot is now ONLINE and polling for updates!")

    # 8. Start Polling with Graceful Shutdown Cleanup
    try:
        # Drop pending updates to avoid processing stale messages sent while bot was offline
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    except asyncio.CancelledError:
        logger.warning("Polling interrupted by cancellation signal.")
    except Exception as e:
        logger.critical(f"Fatal error during bot polling: {e}", exc_info=True)
    finally:
        logger.info("Shutting down SANA AI Bot... Closing HTTP sessions.")
        await bot.session.close()
        logger.info("SANA AI Bot gracefully stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot manually terminated by user.")
