"""
app/tools/registry/registry_utils.py - Utility Helpers for Tool Registry
=======================================================================
Provides reflection helpers, module path resolvers, semantic versioning checkers,
and formatting utilities for tool discovery and registry reporting.
"""

import os
import re
import uuid
import inspect
from typing import Type, Any, Optional, Tuple
from app.tools.base_tool import BaseTool


def generate_tool_id(tool_name: str) -> str:
    """Generates a reproducible, clean tool identifier."""
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', tool_name.lower())
    return f"tool_{sanitized}"


def is_basetool_subclass(cls: Any) -> bool:
    """Checks if a given object is a concrete, instantiable subclass of BaseTool."""
    if not inspect.isclass(cls):
        return False
    if cls is BaseTool:
        return False
    return issubclass(cls, BaseTool) and not inspect.isabstract(cls)


def file_path_to_module_name(file_path: str, project_root: Optional[str] = None) -> str:
    """Converts a filesystem path (e.g., app/tools/system/time_tool.py) into a Python dot-module path."""
    abs_path = os.path.abspath(file_path)
    if project_root is None:
        project_root = os.getcwd()
    
    rel_path = os.path.relpath(abs_path, project_root)
    no_ext = os.path.splitext(rel_path)[0]
    module_path = no_ext.replace(os.sep, ".").replace("/", ".")
    return module_path


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Parses a semantic version string (e.g. '1.2.3') into a tuple (major, minor, patch)."""
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', version_str.strip())
    if not match:
        return (1, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_registry_report(report_data: Any) -> str:
    """Formats a RegistryReport dataclass into a clean CLI/Log text summary."""
    lines = [
        "============================================================",
        "              SANA AI TOOL REGISTRY REPORT                  ",
        "============================================================",
        f"Report ID: {getattr(report_data, 'report_id', 'N/A')}",
        f"Scanned Files/Classes: {getattr(report_data, 'total_scanned', 0)}",
        f"Registered Tools:      {getattr(report_data, 'total_registered', 0)}",
        f"Failed / Skipped:      {getattr(report_data, 'total_failed', 0)}",
        "------------------------------------------------------------"
    ]
    registrations = getattr(report_data, 'registrations', [])
    for reg in registrations:
        status_symbol = "✅" if reg.status.value == "success" else "❌"
        lines.append(f"{status_symbol} Tool: {reg.tool_name:<25} | Status: {reg.status.value.upper()}")
        if reg.error_message:
            lines.append(f"   └── Reason: {reg.error_message}")
    lines.append("============================================================")
    return "\n".join(lines)
