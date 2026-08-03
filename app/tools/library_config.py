"""
app/tools/library_config.py - Centralized Tool Library Configuration
====================================================================
Manages filesystem workspace roots, security boundaries, maximum file sizes,
extension whitelists/blacklists, and tool operational constraints.
"""

import os
from typing import Set
from dataclasses import dataclass, field


@dataclass
class ToolLibraryConfig:
    """Centralized configuration for SANA AI Tool Library."""

    workspace_root: str = "."
    max_file_size_bytes: int = 10 * 1024 * 1024     # 10 MB limit
    max_read_size_bytes: int = 1 * 1024 * 1024       # 1 MB limit
    max_write_size_bytes: int = 5 * 1024 * 1024      # 5 MB limit
    enable_file_safety_checks: bool = True
    allowed_extensions: Set[str] = field(default_factory=lambda: {
        ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log",
        ".ini", ".cfg", ".html", ".css", ".js", ".sql", ".xml"
    })
    blocked_extensions: Set[str] = field(default_factory=lambda: {
        ".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".vbs", ".sh", ".bin"
    })

    @classmethod
    def from_env(cls) -> "ToolLibraryConfig":
        return cls(
            workspace_root=os.path.abspath(os.getenv("SANA_WORKSPACE_ROOT", ".")),
            max_file_size_bytes=int(os.getenv("SANA_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            max_read_size_bytes=int(os.getenv("SANA_MAX_READ_SIZE", str(1 * 1024 * 1024))),
            max_write_size_bytes=int(os.getenv("SANA_MAX_WRITE_SIZE", str(5 * 1024 * 1024))),
            enable_file_safety_checks=os.getenv("SANA_FILE_SAFETY", "true").lower() == "true",
        )


_library_config_instance: ToolLibraryConfig = ToolLibraryConfig.from_env()


def get_library_config() -> ToolLibraryConfig:
    """Returns global tool library configuration singleton."""
    return _library_config_instance


def set_library_config(config: ToolLibraryConfig) -> None:
    """Overrides global tool library configuration."""
    global _library_config_instance
    _library_config_instance = config
