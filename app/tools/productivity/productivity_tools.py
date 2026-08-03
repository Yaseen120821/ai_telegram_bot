"""
app/tools/productivity/productivity_tools.py - Productivity & Task Management Tools
==================================================================================
Provides local productivity tools: Notes, Todo, Reminder, Task List, Clipboard,
Timer, Stopwatch, and Simple Calendar tools.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel

logger = logging.getLogger("sana_ai.tools.productivity")

# In-memory storage stores for local productivity state
_NOTES_STORE: Dict[str, str] = {}
_TODOS_STORE: List[Dict[str, Any]] = []
_REMINDERS_STORE: List[Dict[str, Any]] = []
_CLIPBOARD_STORE: str = ""
_STOPWATCH_START: Optional[float] = None


class NotesTool(BaseTool):
    """Manages personal notes (create, view, delete)."""
    def __init__(self):
        super().__init__(
            name="manage_notes",
            description="Creates, retrieves, or lists quick personal text notes.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "get", "list", "delete"]},
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["action"]
            },
            tags=["notes", "note", "notepad", "memo"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        action = parameters["action"].lower()
        title = parameters.get("title", "Untitled")

        if action == "add":
            _NOTES_STORE[title] = parameters.get("content", "")
            return {"action": "add", "title": title, "status": "saved"}
        elif action == "get":
            content = _NOTES_STORE.get(title, "Note not found.")
            return {"title": title, "content": content}
        elif action == "list":
            return {"total_notes": len(_NOTES_STORE), "titles": list(_NOTES_STORE.keys())}
        elif action == "delete":
            if title in _NOTES_STORE:
                del _NOTES_STORE[title]
                return {"title": title, "status": "deleted"}
            return {"title": title, "status": "not_found"}
        return {"error": f"Unknown action '{action}'"}


class TodoTool(BaseTool):
    """Manages a personal TODO item list."""
    def __init__(self):
        super().__init__(
            name="manage_todos",
            description="Adds, lists, or completes todo checklist items.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "list", "complete"]},
                    "item": {"type": "string"},
                    "index": {"type": "integer"}
                },
                "required": ["action"]
            },
            tags=["todo", "todolist", "checklist"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        action = parameters["action"].lower()

        if action == "add":
            item = parameters.get("item", "New task")
            _TODOS_STORE.append({"task": item, "completed": False})
            return {"action": "add", "task": item, "status": "added"}
        elif action == "list":
            return {"total": len(_TODOS_STORE), "todos": _TODOS_STORE}
        elif action == "complete":
            idx = parameters.get("index", 1) - 1
            if 0 <= idx < len(_TODOS_STORE):
                _TODOS_STORE[idx]["completed"] = True
                return {"index": idx + 1, "task": _TODOS_STORE[idx]["task"], "status": "completed"}
            return {"error": "Invalid todo item index"}
        return {"error": f"Unknown action '{action}'"}


class ReminderTool(BaseTool):
    """Schedules local reminders."""
    def __init__(self):
        super().__init__(
            name="manage_reminders",
            description="Sets or lists local reminder notifications.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set", "list"]},
                    "reminder_text": {"type": "string"},
                    "delay_seconds": {"type": "integer"}
                },
                "required": ["action"]
            },
            tags=["reminder", "remind", "alarm"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        action = parameters["action"].lower()
        if action == "set":
            text = parameters.get("reminder_text", "Reminder")
            delay = parameters.get("delay_seconds", 60)
            trigger_t = time.time() + delay
            _REMINDERS_STORE.append({"text": text, "trigger_time": trigger_t})
            return {"action": "set", "reminder": text, "trigger_in_seconds": delay}
        elif action == "list":
            return {"total": len(_REMINDERS_STORE), "reminders": _REMINDERS_STORE}
        return {"error": f"Unknown action '{action}'"}


class TaskListTool(BaseTool):
    """Manages categorized task lists."""
    def __init__(self):
        super().__init__(
            name="manage_task_list",
            description="Views summary metrics of current active task lists.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["task_list", "tasks", "summary"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        completed = sum(1 for t in _TODOS_STORE if t.get("completed"))
        pending = len(_TODOS_STORE) - completed
        return {
            "total_notes": len(_NOTES_STORE),
            "total_todos": len(_TODOS_STORE),
            "pending_todos": pending,
            "completed_todos": completed,
            "active_reminders": len(_REMINDERS_STORE)
        }


class ClipboardTool(BaseTool):
    """Reads or writes simulated in-memory text clipboard."""
    def __init__(self):
        super().__init__(
            name="manage_clipboard",
            description="Reads from or copies text into the application clipboard.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["copy", "paste"]},
                    "text": {"type": "string"}
                },
                "required": ["action"]
            },
            tags=["clipboard", "copy", "paste"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        global _CLIPBOARD_STORE
        action = parameters["action"].lower()

        if action == "copy":
            _CLIPBOARD_STORE = parameters.get("text", "")
            return {"action": "copy", "status": "copied", "length": len(_CLIPBOARD_STORE)}
        elif action == "paste":
            return {"action": "paste", "clipboard_content": _CLIPBOARD_STORE}
        return {"error": f"Unknown action '{action}'"}


class TimerTool(BaseTool):
    """Simulates starting a countdown timer."""
    def __init__(self):
        super().__init__(
            name="start_timer",
            description="Starts a countdown timer for a given duration in seconds.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "duration_seconds": {"type": "integer"}
                },
                "required": ["duration_seconds"]
            },
            tags=["timer", "countdown", "alarm"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        sec = parameters["duration_seconds"]
        return {"duration_seconds": sec, "status": "started", "message": f"Timer set for {sec} seconds."}


class StopwatchTool(BaseTool):
    """Starts, stops, or reads a stopwatch."""
    def __init__(self):
        super().__init__(
            name="manage_stopwatch",
            description="Starts, stops, or reads a stopwatch timer.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "read", "reset"]}
                },
                "required": ["action"]
            },
            tags=["stopwatch", "timer", "elapsed"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        global _STOPWATCH_START
        action = parameters["action"].lower()

        if action == "start":
            _STOPWATCH_START = time.time()
            return {"action": "start", "status": "running"}
        elif action == "read":
            if _STOPWATCH_START is None:
                return {"elapsed_seconds": 0.0, "status": "stopped"}
            elapsed = time.time() - _STOPWATCH_START
            return {"elapsed_seconds": round(elapsed, 2), "status": "running"}
        elif action == "reset":
            _STOPWATCH_START = None
            return {"action": "reset", "status": "cleared"}
        return {"error": f"Unknown action '{action}'"}


class SimpleCalendarTool(BaseTool):
    """Displays monthly calendar grid for month and year."""
    def __init__(self):
        super().__init__(
            name="simple_calendar",
            description="Generates an ASCII calendar grid for a specific month and year.",
            category=ToolCategory.PRODUCTIVITY,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {"type": "integer"}
                }
            },
            tags=["calendar", "month", "grid"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        import calendar
        now = time.localtime()
        yr = parameters.get("year", now.tm_year)
        mo = parameters.get("month", now.tm_mon)

        cal_str = calendar.month(yr, mo)
        return {"year": yr, "month": mo, "calendar_grid": cal_str}
