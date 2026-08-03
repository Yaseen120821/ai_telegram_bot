# SANA AI - Personal AI Assistant (Chapter 2)

Welcome to **Chapter 2** of **SANA AI**! In this chapter, we built a production-grade, asynchronous Telegram Bot using **aiogram 3.x**, **python-dotenv**, and **logging**.

---

## 📁 Project Architecture & Directory Structure

```
PersonalAI_Bot/
│
├── .venv/                         # Preserved Python virtual environment
│
├── app/                           # Business Domain Layer (Future Chapters)
│   ├── llm/                       # Qwen model wrappers
│   ├── memory/                    # Short-term & long-term memory
│   ├── emotion/                   # Emotion detection engine
│   ├── rag/                       # Vector search & RAG
│   ├── database/                  # SQLite/PostgreSQL connectors
│   └── utils/                     # Shared utilities
│
├── bot/                           # Telegram Bot Transport Layer
│   ├── handlers/                  # Message & command routing
│   │   ├── common.py              # /start, /help, menu handlers
│   │   └── echo.py                # Echo text & unknown command handlers
│   ├── keyboards/                 # Interactive Reply Keyboards
│   │   └── common_keyboards.py    # Main menu layout
│   ├── middlewares/               # Telemetry & Logging
│   │   └── logging_middleware.py  # Request logger & execution latency
│   ├── services/                  # UI services
│   │   └── command_service.py     # Telegram auto-complete command setter
│   ├── config.py                  # Environment config loader (.env)
│   └── bot.py                     # Dispatcher, polling loop & shutdown hooks
│
├── models/                        # Preserved local Qwen model weights
│   └── qwen/
│
├── scripts/                       # Preserved download & interactive chat scripts
├── tests/                         # Preserved test suite
├── logs/                          # Runtime log directory (bot.log)
│
├── .env                           # Secret variables (BOT_TOKEN)
├── .env.example                   # Environment configuration template
├── requirements.txt               # Dependencies (aiogram, python-dotenv, torch, transformers)
├── main.py                        # Root execution entry point
└── README.md                      # Documentation & execution guide
```

---

## 🚀 How to Run Chapter 2

### Step 1: Configure Your Telegram Bot Token
1. Open Telegram and search for `@BotFather`.
2. Create a new bot using `/newbot` and follow the instructions.
3. Copy your API Token (e.g. `1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ`).
4. Open the `.env` file in project root and paste your token:
   ```env
   BOT_TOKEN=1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ
   LOG_LEVEL=INFO
   ENVIRONMENT=development
   ```

### Step 2: Start the Bot
Run the following command in your terminal:
```bash
.venv\Scripts\python main.py
```

---

## 🛠️ Bot Features Implemented

| Command / Feature | Description |
| :--- | :--- |
| `/start` | Welcomes user, introduces SANA AI, renders interactive menu keyboard |
| `/help` | Displays command guide & architecture status |
| `❓ Help` | Reply Keyboard button to trigger help menu |
| `🤖 About SANA` | Reply Keyboard button to show bot bio |
| `⚡ Status` | Reply Keyboard button to check system health |
| Text Echo | Echoes back user text messages with clean formatting |
| Unknown Command | Handles unconfigured slash commands gracefully |
| Audit Logging | Logs request telemetry, user IDs, and latency to console & `logs/bot.log` |
| Graceful Shutdown | Handles `Ctrl+C` / SIGINT signals safely without hanging HTTP sessions |

---

## 📚 Key Software Engineering Concepts Learned

1. **Decoupled Architecture**: Transport layer (`bot/`) is strictly separated from domain logic (`app/`).
2. **Asynchronous Programming (`async`/`await`)**: Uses Python's `asyncio` event loop for non-blocking I/O operations.
3. **aiogram 3.x Router Pattern**: Modular event routing using `Router` and `Dispatcher`.
4. **Middlewares**: Centralized request interception for logging, metrics, and error handling.
5. **Environment Security**: Safe secrets management using `.env` and `python-dotenv`.
