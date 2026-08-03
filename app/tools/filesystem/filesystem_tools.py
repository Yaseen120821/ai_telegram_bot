"""
app/tools/filesystem/filesystem_tools.py - Workspace Filesystem Operations Tools
================================================================================
Provides secure workspace filesystem tools: Read, Write, Append, Delete, Rename,
Move, Copy, Create Folder, Delete Folder, List Directory, Search Files, File Metadata.
"""

import os
import shutil
import logging
from typing import Dict, Any, List, Optional
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.tool_exceptions import ToolExecutionException, PermissionDeniedException
from app.tools.library_config import get_library_config
from app.tools.execution.execution_utils import is_safe_filepath

logger = logging.getLogger("sana_ai.tools.filesystem")


def _get_validated_workspace_path(path_str: str, check_exists: bool = True) -> str:
    """Helper verifying that path is within workspace boundaries and optionally exists."""
    config = get_library_config()
    base_dir = os.path.abspath(config.workspace_root)
    full_path = os.path.abspath(os.path.join(base_dir, path_str))

    if not is_safe_filepath(full_path, base_dir=base_dir):
        raise PermissionDeniedException(f"Security block: Path '{path_str}' escapes workspace boundary.", tool_name="filesystem")

    if check_exists and not os.path.exists(full_path):
        raise ToolExecutionException(f"Path '{path_str}' does not exist.", tool_name="filesystem")

    ext = os.path.splitext(full_path)[1].lower()
    if ext in config.blocked_extensions:
        raise PermissionDeniedException(f"Security block: Extension '{ext}' is blocked.", tool_name="filesystem")

    return full_path


class ReadFileTool(BaseTool):
    """Reads content from a text file within workspace."""
    def __init__(self):
        super().__init__(
            name="read_file_tool",
            description="Reads text content from a specified workspace file.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Relative filename or path inside workspace."}
                },
                "required": ["filename"]
            },
            tags=["read_file", "file", "open", "read"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fname = parameters["filename"]
        target = _get_validated_workspace_path(fname, check_exists=True)
        config = get_library_config()

        if os.path.getsize(target) > config.max_read_size_bytes:
            raise ToolExecutionException(f"File '{fname}' exceeds maximum read size limit.", tool_name=self.name)

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return {"filename": fname, "size_bytes": len(content), "content": content}


class WriteFileTool(BaseTool):
    """Writes text content to a workspace file (overwrites existing)."""
    def __init__(self):
        super().__init__(
            name="write_file_tool",
            description="Writes or overwrites text content to a file within workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            },
            tags=["write_file", "write", "save", "create_file"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fname = parameters["filename"]
        content = parameters["content"]
        target = _get_validated_workspace_path(fname, check_exists=False)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        return {"filename": fname, "written_bytes": len(content), "status": "saved"}


class AppendFileTool(BaseTool):
    """Appends text content to an existing workspace file."""
    def __init__(self):
        super().__init__(
            name="append_file_tool",
            description="Appends text content to the end of a file within workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            },
            tags=["append_file", "append"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fname = parameters["filename"]
        content = parameters["content"]
        target = _get_validated_workspace_path(fname, check_exists=False)

        with open(target, "a", encoding="utf-8") as f:
            f.write(content)

        return {"filename": fname, "appended_bytes": len(content), "status": "appended"}


class DeleteFileTool(BaseTool):
    """Deletes a file inside the workspace."""
    def __init__(self):
        super().__init__(
            name="delete_file_tool",
            description="Deletes a file inside workspace directory.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
            },
            tags=["delete_file", "remove_file", "unlink"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fname = parameters["filename"]
        target = _get_validated_workspace_path(fname, check_exists=True)
        if not os.path.isfile(target):
            raise ToolExecutionException(f"Target '{fname}' is not a regular file.", tool_name=self.name)

        os.remove(target)
        return {"filename": fname, "status": "deleted"}


class RenameFileTool(BaseTool):
    """Renames a file or folder inside the workspace."""
    def __init__(self):
        super().__init__(
            name="rename_file_tool",
            description="Renames a file or folder within workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "old_filename": {"type": "string"},
                    "new_filename": {"type": "string"}
                },
                "required": ["old_filename", "new_filename"]
            },
            tags=["rename", "mv"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        old_p = _get_validated_workspace_path(parameters["old_filename"], check_exists=True)
        new_p = _get_validated_workspace_path(parameters["new_filename"], check_exists=False)

        os.rename(old_p, new_p)
        return {"old_filename": parameters["old_filename"], "new_filename": parameters["new_filename"], "status": "renamed"}


class MoveFileTool(BaseTool):
    """Moves a file to a new workspace destination."""
    def __init__(self):
        super().__init__(
            name="move_file_tool",
            description="Moves a file to a new directory within workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"]
            },
            tags=["move", "move_file"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        src = _get_validated_workspace_path(parameters["source"], check_exists=True)
        dst = _get_validated_workspace_path(parameters["destination"], check_exists=False)

        shutil.move(src, dst)
        return {"source": parameters["source"], "destination": parameters["destination"], "status": "moved"}


class CopyFileTool(BaseTool):
    """Copies a file inside workspace."""
    def __init__(self):
        super().__init__(
            name="copy_file_tool",
            description="Copies a file inside workspace directory.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"]
            },
            tags=["copy", "cp", "copy_file"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        src = _get_validated_workspace_path(parameters["source"], check_exists=True)
        dst = _get_validated_workspace_path(parameters["destination"], check_exists=False)

        shutil.copy2(src, dst)
        return {"source": parameters["source"], "destination": parameters["destination"], "status": "copied"}


class CreateFolderTool(BaseTool):
    """Creates a new folder inside workspace."""
    def __init__(self):
        super().__init__(
            name="create_folder_tool",
            description="Creates a new directory inside workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string"}
                },
                "required": ["folder_name"]
            },
            tags=["mkdir", "create_folder"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        target = _get_validated_workspace_path(parameters["folder_name"], check_exists=False)
        os.makedirs(target, exist_ok=True)
        return {"folder_name": parameters["folder_name"], "status": "created"}


class DeleteFolderTool(BaseTool):
    """Deletes a directory inside workspace."""
    def __init__(self):
        super().__init__(
            name="delete_folder_tool",
            description="Deletes a directory tree within workspace.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "folder_name": {"type": "string"}
                },
                "required": ["folder_name"]
            },
            tags=["rmdir", "delete_folder"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        target = _get_validated_workspace_path(parameters["folder_name"], check_exists=True)
        if not os.path.isdir(target):
            raise ToolExecutionException(f"Target '{parameters['folder_name']}' is not a directory.", tool_name=self.name)

        shutil.rmtree(target)
        return {"folder_name": parameters["folder_name"], "status": "deleted"}


class ListDirectoryTool(BaseTool):
    """Lists files and folders inside a workspace directory."""
    def __init__(self):
        super().__init__(
            name="list_directory_tool",
            description="Lists files and subdirectories within a workspace path.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "directory": {"type": "string"}
                }
            },
            tags=["ls", "dir", "list_directory"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        dir_p = parameters.get("directory", ".")
        target = _get_validated_workspace_path(dir_p, check_exists=True)

        items = []
        for name in os.listdir(target):
            full = os.path.join(target, name)
            items.append({
                "name": name,
                "is_dir": os.path.isdir(full),
                "size_bytes": os.path.getsize(full) if os.path.isfile(full) else 0
            })

        return {"directory": dir_p, "total_items": len(items), "items": items}


class SearchFilesTool(BaseTool):
    """Searches workspace files matching pattern or keyword."""
    def __init__(self):
        super().__init__(
            name="search_files_tool",
            description="Searches for files matching a filename keyword or extension.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"}
                },
                "required": ["keyword"]
            },
            tags=["search_files", "find"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        keyword = parameters["keyword"].lower()
        config = get_library_config()
        base_dir = os.path.abspath(config.workspace_root)

        matches = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if keyword in file.lower():
                    rel = os.path.relpath(os.path.join(root, file), base_dir)
                    matches.append(rel)
                if len(matches) >= 50:
                    break

        return {"keyword": keyword, "count": len(matches), "matches": matches}


class FileMetadataTool(BaseTool):
    """Returns size, timestamps, and metadata for a workspace file."""
    def __init__(self):
        super().__init__(
            name="get_file_metadata",
            description="Fetches file size, modification date, and permissions metadata.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
            },
            tags=["file_metadata", "stat", "file_info"]
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        fname = parameters["filename"]
        target = _get_validated_workspace_path(fname, check_exists=True)
        st = os.stat(target)

        return {
            "filename": fname,
            "size_bytes": st.st_size,
            "modified_timestamp": st.st_mtime,
            "created_timestamp": st.st_ctime,
            "is_file": os.path.isfile(target),
            "is_dir": os.path.isdir(target)
        }
