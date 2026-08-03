"""
app/tools/developer/developer_tools.py - Developer & Codebase Inspection Tools
=============================================================================
Provides code inspection tools: Read Logs, Search Logs, Project Tree,
Read Source File, Count Lines, Find TODOs, Search Code, and Project Statistics.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.tool_exceptions import ToolExecutionException
from app.tools.library_config import get_library_config
from app.tools.filesystem.filesystem_tools import _get_validated_workspace_path

logger = logging.getLogger("sana_ai.tools.developer")


class ReadLogsTool(BaseTool):
    """Reads latest N lines from application log files."""
    def __init__(self):
        super().__init__(
            name="read_logs_tool",
            description="Reads the last N lines from recent application log files.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of recent lines to read (default 50)"}
                }
            },
            tags=["logs", "read_logs", "diagnostics"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        num_lines = min(max(parameters.get("lines", 50), 1), 500)
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        # Look for log files in logs directory or root
        log_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".log"):
                    log_files.append(os.path.join(root, file))

        if not log_files:
            return {"lines_requested": num_lines, "logs": "No log files found in workspace."}

        latest_log = max(log_files, key=os.path.getmtime)
        with open(latest_log, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        recent = all_lines[-num_lines:]
        return {
            "log_file": os.path.relpath(latest_log, base_dir),
            "total_lines": len(all_lines),
            "retrieved_lines": len(recent),
            "content": "".join(recent)
        }


class SearchLogsTool(BaseTool):
    """Searches application logs for matching keywords."""
    def __init__(self):
        super().__init__(
            name="search_logs_tool",
            description="Searches application logs for occurrences of a keyword or error code.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            tags=["search_logs", "grep_logs", "error_search"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        query = parameters["query"].lower()
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        matches = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".log"):
                    full_p = os.path.join(root, file)
                    with open(full_p, "r", encoding="utf-8", errors="replace") as f:
                        for idx, line in enumerate(f, 1):
                            if query in line.lower():
                                matches.append({
                                    "file": os.path.relpath(full_p, base_dir),
                                    "line_number": idx,
                                    "text": line.strip()
                                })
                                if len(matches) >= 100:
                                    break

        return {"query": query, "count": len(matches), "matches": matches}


class ProjectTreeTool(BaseTool):
    """Generates ASCII directory tree of project workspace."""
    def __init__(self):
        super().__init__(
            name="project_tree_tool",
            description="Generates a visual directory tree structure of the codebase.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "max_depth": {"type": "integer"}
                }
            },
            tags=["tree", "project_structure", "codebase_map"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        max_depth = min(max(parameters.get("max_depth", 3), 1), 6)
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        lines = [os.path.basename(base_dir) or "workspace"]
        self._build_tree(base_dir, "", lines, current_depth=1, max_depth=max_depth)
        return {"tree": "\n".join(lines)}

    def _build_tree(self, path: str, prefix: str, lines: List[str], current_depth: int, max_depth: int):
        if current_depth > max_depth:
            return

        ignored = {".git", "__pycache__", ".venv", "node_modules", ".idea", ".vscode"}
        try:
            entries = sorted([e for e in os.listdir(path) if e not in ignored])
        except Exception:
            return

        for idx, entry in enumerate(entries):
            is_last = (idx == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            full = os.path.join(path, entry)
            lines.append(f"{prefix}{connector}{entry}")

            if os.path.isdir(full):
                extension = "    " if is_last else "│   "
                self._build_tree(full, prefix + extension, lines, current_depth + 1, max_depth)


class ReadSourceFileTool(BaseTool):
    """Reads source code file with line numbers."""
    def __init__(self):
        super().__init__(
            name="read_source_file_tool",
            description="Reads Python source code with line number annotations.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"}
                },
                "required": ["filepath"]
            },
            tags=["read_code", "source_code", "python_file"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fpath = parameters["filepath"]
        target = _get_validated_workspace_path(fpath, check_exists=True)

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        numbered = [f"{idx + 1:4d} | {line}" for idx, line in enumerate(lines)]
        return {"filepath": fpath, "total_lines": len(lines), "content": "".join(numbered)}


class CountLinesTool(BaseTool):
    """Counts lines of code across project source files."""
    def __init__(self):
        super().__init__(
            name="count_lines_tool",
            description="Counts total lines of code, blank lines, and comment lines in workspace.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["cloc", "count_lines", "loc"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        total_code = 0
        total_blank = 0
        total_comments = 0
        file_count = 0

        for root, dirs, files in os.walk(base_dir):
            if any(part in root for part in [".git", "__pycache__", ".venv"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    file_count += 1
                    full = os.path.join(root, file)
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            sline = line.strip()
                            if not sline:
                                total_blank += 1
                            elif sline.startswith("#"):
                                total_comments += 1
                            else:
                                total_code += 1

        return {
            "python_files": file_count,
            "code_lines": total_code,
            "comment_lines": total_comments,
            "blank_lines": total_blank,
            "total_lines": total_code + total_comments + total_blank
        }


class FindTodoTool(BaseTool):
    """Finds TODO, FIXME, and XXX comments in codebase."""
    def __init__(self):
        super().__init__(
            name="find_todo_tool",
            description="Scans codebase for TODO, FIXME, and XXX developer annotations.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["todo", "fixme", "notes", "code_annotations"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)
        todos = []

        for root, dirs, files in os.walk(base_dir):
            if any(part in root for part in [".git", "__pycache__", ".venv"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, base_dir)
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for idx, line in enumerate(f, 1):
                            if any(k in line for k in ["TODO", "FIXME", "XXX"]):
                                todos.append({
                                    "file": rel,
                                    "line": idx,
                                    "text": line.strip()
                                })

        return {"count": len(todos), "todos": todos}


class SearchCodeTool(BaseTool):
    """Searches workspace Python files for code symbols or keywords."""
    def __init__(self):
        super().__init__(
            name="search_code_tool",
            description="Searches source code files for function, class, or symbol usages.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            tags=["search_code", "grep", "symbol_search"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        query = parameters["query"]
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)
        matches = []

        for root, dirs, files in os.walk(base_dir):
            if any(part in root for part in [".git", "__pycache__", ".venv"]):
                continue
            for file in files:
                if file.endswith((".py", ".json", ".md")):
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, base_dir)
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        for idx, line in enumerate(f, 1):
                            if query in line:
                                matches.append({
                                    "file": rel,
                                    "line": idx,
                                    "text": line.strip()
                                })
                                if len(matches) >= 100:
                                    break

        return {"query": query, "count": len(matches), "matches": matches}


class ProjectStatisticsTool(BaseTool):
    """Generates overall codebase statistics and metrics."""
    def __init__(self):
        super().__init__(
            name="get_project_statistics",
            description="Returns high-level project statistics (total files, directories, size).",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {}},
            tags=["project_stats", "metrics", "codebase_summary"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        total_files = 0
        total_dirs = 0
        total_size = 0
        ext_counts: Dict[str, int] = {}

        for root, dirs, files in os.walk(base_dir):
            if any(part in root for part in [".git", "__pycache__", ".venv"]):
                continue
            total_dirs += len(dirs)
            for file in files:
                total_files += 1
                full = os.path.join(root, file)
                total_size += os.path.getsize(full)
                ext = os.path.splitext(file)[1].lower() or "no_ext"
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

        return {
            "total_directories": total_dirs,
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_extensions": ext_counts
        }
