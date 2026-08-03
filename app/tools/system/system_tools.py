"""
app/tools/system/system_tools.py - System & Environment Inspection Tools
========================================================================
Provides read-only system tools: CurrentTime, CurrentDate, CurrentDateTime,
Timezone, SystemInfo, and Uptime tools.
"""

import time
import os
import platform
import datetime
from typing import Dict, Any, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel

_START_TIME = time.time()


class CurrentTimeTool(BaseTool):
    """Returns current system local time."""
    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Fetches the current local system time (HH:MM:SS AM/PM).",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["time", "clock", "current_time", "system"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> str:
        return datetime.datetime.now().strftime("%I:%M:%S %p")


class CurrentDateTool(BaseTool):
    """Returns current system local date."""
    def __init__(self):
        super().__init__(
            name="get_current_date",
            description="Fetches the current local system date (YYYY-MM-DD, DayOfWeek).",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["date", "calendar", "current_date", "today"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d (%A)")


class CurrentDateTimeTool(BaseTool):
    """Returns full current ISO-8601 date and time timestamp."""
    def __init__(self):
        super().__init__(
            name="get_current_datetime",
            description="Fetches full current system ISO-8601 timestamp.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["datetime", "timestamp", "current_datetime"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> str:
        return datetime.datetime.now().isoformat()


class TimezoneTool(BaseTool):
    """Returns local system timezone and UTC offset."""
    def __init__(self):
        super().__init__(
            name="get_system_timezone",
            description="Fetches local system timezone name and UTC offset.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["timezone", "tz", "utc_offset"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, str]:
        now = datetime.datetime.now().astimezone()
        return {
            "timezone": now.tzname() or "Local",
            "utc_offset": now.strftime("%z")
        }


class SystemInfoTool(BaseTool):
    """Returns system platform architecture and runtime diagnostic metadata."""
    def __init__(self):
        super().__init__(
            name="get_system_info",
            description="Returns OS platform, architecture, CPU, and Python runtime metadata.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["system_info", "os", "platform", "cpu", "python_version"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor() or "Unknown",
            "python_version": platform.python_version()
        }


class UptimeTool(BaseTool):
    """Returns application process uptime."""
    def __init__(self):
        super().__init__(
            name="get_app_uptime",
            description="Returns process uptime in seconds and formatted duration.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["uptime", "runtime", "process_time"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        elapsed = time.time() - _START_TIME
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return {
            "uptime_seconds": round(elapsed, 2),
            "formatted_uptime": f"{hours}h {minutes}m {seconds}s"
        }
