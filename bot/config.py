"""
bot/config.py - Environment & Application Configuration Loader
===============================================================

1. PURPOSE:
-----------
Loads environment variables from the `.env` file into a strongly-typed Python
dataclass (`Config`). It ensures the Telegram Bot API token and log settings
are loaded safely before any bot services start.

2. WHY IT EXISTS:
-----------------
Hardcoding secrets (like Telegram Bot Tokens) in source code is a severe security
risk. If code is pushed to GitHub or shared, your bot can be hijacked.
`bot/config.py` acts as a single source of truth for runtime configurations.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Reads: `.env` file via `dotenv.load_dotenv()`.
- Imported by: `bot/bot.py`, `bot/services/command_service.py`, `main.py`.

4. EXECUTION FLOW:
------------------
1. `load_dotenv()` looks for `.env` at project root.
2. `Config.load_from_env()` reads environment variables via `os.getenv()`.
3. Validates required variables (`BOT_TOKEN`).
4. Returns an immutable `Config` object to the caller.

5. COMMON MISTAKES:
-------------------
- Committing `.env` with actual bot tokens to version control (Git).
- Forgetting to call `load_dotenv()` before accessing `os.getenv()`.
- Not handling missing `BOT_TOKEN` gracefully.

6. DEBUGGING TIPS:
------------------
- If `BOT_TOKEN` is missing, verify `.env` exists in project root.
- Ensure no trailing quotes or extra spaces exist in `.env` (`BOT_TOKEN=12345:ABC...`).

7. TESTING INSTRUCTIONS:
------------------------
- Run: `.venv\\Scripts\\python -c "from bot.config import Config; print(Config.load_from_env())"`
"""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger("sana_ai.config")

@dataclass(frozen=True)
class Config:
    """
    Immutable Configuration Data Class.

    Attributes:
        bot_token (str): Telegram Bot API Token obtained from @BotFather.
        log_level (str): Logging severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        environment (str): Operational mode (development, production).
        model_path (str): Path to local Qwen LLM directory.
        max_new_tokens (int): Maximum output tokens per response generation.
    """
    bot_token: str
    log_level: str = "INFO"
    environment: str = "development"
    model_path: str = "models/qwen"
    max_new_tokens: int = 512

    @classmethod
    def load_from_env(cls) -> "Config":
        """
        Loads and validates configuration from environment variables.

        Returns:
            Config: Populated configuration object.

        Raises:
            ValueError: If BOT_TOKEN is missing or left as default placeholder.
        """
        # Load environment variables from .env file (with override=True to catch updates)
        from dotenv import find_dotenv
        load_dotenv(find_dotenv(), override=True)

        bot_token = os.getenv("BOT_TOKEN", "").strip()
        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        environment = os.getenv("ENVIRONMENT", "development").strip().lower()
        model_path = os.getenv("MODEL_PATH", "models/qwen").strip()
        
        try:
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512").strip())
        except ValueError:
            max_new_tokens = 512

        # Validate Bot Token existence and placeholder state
        if not bot_token or bot_token == "your_telegram_bot_token_here":
            error_msg = (
                "CRITICAL ERROR: BOT_TOKEN is missing or unconfigured in .env file!\n"
                "Please open .env and set BOT_TOKEN=<your_actual_bot_token> obtained from @BotFather."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Configuration loaded successfully [Environment: {environment}, LogLevel: {log_level}, ModelPath: {model_path}]")
        return cls(
            bot_token=bot_token,
            log_level=log_level,
            environment=environment,
            model_path=model_path,
            max_new_tokens=max_new_tokens
        )
