"""
bot/keyboards/common_keyboards.py - Interactive Telegram Keyboards
=====================================================================

1. PURPOSE:
-----------
Constructs custom Telegram Reply Keyboards (buttons displayed below the chat bar)
to improve user experience and enable quick command triggering.

2. WHY IT EXISTS:
-----------------
Users shouldn't be forced to manually type `/start` or `/help` every time.
Keyboards make your bot feel like an interactive application rather than a bare CLI tool.

3. HOW IT COMMUNICATES WITH OTHER FILES:
----------------------------------------
- Imported by: `bot/handlers/common.py`.
- Included in bot responses via `await message.answer(..., reply_markup=get_main_reply_keyboard())`.

4. EXECUTION FLOW:
------------------
1. Handler function (e.g. `cmd_start`) calls `get_main_reply_keyboard()`.
2. `ReplyKeyboardBuilder` formats buttons into rows.
3. Keyboards are serialized to JSON and sent to Telegram API.
4. Telegram client renders buttons in the user's chat window.

5. COMMON MISTAKES:
-------------------
- Making buttons too large by forgetting `resize_keyboard=True`.
- Overcrowding rows with too many buttons, making UI illegible on mobile screens.

6. DEBUGGING TIPS:
------------------
- If buttons don't appear, check if `reply_markup` parameter was passed to `message.answer()`.
- To remove custom keyboards, send `ReplyKeyboardRemove()`.

7. TESTING INSTRUCTIONS:
------------------------
- Send `/start` command in Telegram.
- Verify custom buttons ("❓ Help", "🤖 About SANA", "⚡ Status") appear cleanly at the bottom.
"""

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import ReplyKeyboardBuilder

def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    """
    Constructs the main navigation reply keyboard.

    Returns:
        ReplyKeyboardMarkup: Formatted Telegram keyboard layout with quick action buttons.
    """
    builder = ReplyKeyboardBuilder()
    
    # Add main navigation buttons
    builder.button(text="❓ Help")
    builder.button(text="🤖 About SANA")
    builder.button(text="⚡ Status")
    
    # Adjust layout matrix: 2 buttons in row 1, 1 button in row 2
    builder.adjust(2, 1)
    
    # Render with resize_keyboard=True so buttons adjust neatly on mobile screens
    return builder.as_markup(
        resize_keyboard=True,
        input_field_placeholder="Choose an option or type a message..."
    )
