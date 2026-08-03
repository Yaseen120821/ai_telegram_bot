"""
main.py - Primary Entry Point for SANA AI Assistant
===================================================

1. PURPOSE:
-----------
Serves as the root execution entry point for the SANA AI application.
Initializes runtime environment settings (UTF-8 encoding on Windows), displays
a visual startup banner, loads configuration, and launches the asynchronous Telegram bot loop.

2. WHY IT EXISTS:
-----------------
In standard Python project conventions, `main.py` at root level provides a predictable
command-line interface (`python main.py`). It hides internal package entry details from the user.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Imports and invokes `main_bot()` from `bot/bot.py`.
- Reads configuration via `bot/config.py`.

4. EXECUTION FLOW:
------------------
1. User executes `python main.py`.
2. UTF-8 console stream re-configuration is applied (Windows safety).
3. Rich startup banner is printed to stdout.
4. `asyncio.run(main_bot())` launches event loop and starts Telegram long polling.

5. COMMON MISTAKES:
-------------------
- Launching multiple instances of `main.py` with the same `BOT_TOKEN`, which triggers
  Telegram `409 Conflict: terminated by other getUpdates request` error!

6. DEBUGGING TIPS:
------------------
- If `ValueError` is raised regarding `BOT_TOKEN`, open `.env` and replace placeholder token.

7. TESTING INSTRUCTIONS:
------------------------
- Run: `.venv\\Scripts\\python main.py`
"""

import sys
import asyncio
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from bot.bot import main_bot
from bot.config import Config

console = Console()

def print_sana_banner() -> None:
    """Prints a styled ASCII startup banner for SANA AI."""
    banner_text = Text()
    banner_text.append("=== SANA AI - Personal AI Assistant ===\n", style="bold cyan")
    banner_text.append("Chapter 2: Telegram Bot Layer (aiogram 3.x)\n", style="bold green")
    banner_text.append("Status: Operational | Asynchronous Polling Mode\n", style="magenta")
    banner_text.append("Press Ctrl+C to stop the bot gracefully.\n", style="italic dim")
    
    console.print(Panel(banner_text, title="🤖 SANA AI System Initialization", border_style="cyan"))


def main() -> None:
    """Main application execution wrapper."""
    # Ensure Windows console uses UTF-8 encoding
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
            sys.stdin.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print_sana_banner()

    try:
        # Validate configuration before entering async loop
        Config.load_from_env()
    except ValueError as val_err:
        console.print(Panel(
            f"[bold red]Configuration Error:[/bold red]\n{val_err}",
            title="❌ Setup Required",
            border_style="red"
        ))
        sys.exit(1)

    try:
        asyncio.run(main_bot())
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[bold yellow]SANA AI Bot session closed by user. Goodbye![/bold yellow]")


if __name__ == "__main__":
    main()
