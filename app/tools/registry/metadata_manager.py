"""
app/tools/registry/metadata_manager.py - Tool Metadata Indexing & Cache
=======================================================================
Maintains an in-memory cached index of tool metadata records, tag associations,
schema mappings, and permission indexes for instant lookup without reflection overhead.
"""

import logging
from typing import Dict, List, Optional, Set
from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel, ToolState
from app.tools.registry.registry_models import EnhancedToolMetadata
from app.tools.registry.registry_types import ToolVisibility

logger = logging.getLogger("sana_ai.tools.registry.metadata")


class MetadataManager:
    """Manages metadata caching, tagging indexes, and schema transformations."""

    def __init__(self):
        self._metadata_cache: Dict[str, EnhancedToolMetadata] = {}
        self._category_index: Dict[ToolCategory, Set[str]] = {}
        self._permission_index: Dict[PermissionLevel, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

    def index_tool(self, tool: BaseTool) -> EnhancedToolMetadata:
        """Extracts, enhances, and caches metadata for a tool instance."""
        meta = tool.metadata
        enhanced = EnhancedToolMetadata(
            name=meta.name,
            description=meta.description,
            category=meta.category,
            version=meta.version,
            permission_level=meta.permission_level,
            input_schema=meta.parameters_schema,
            timeout_seconds=meta.timeout_seconds,
            tags=list(meta.tags),
            state=meta.state,
            visibility=ToolVisibility.PUBLIC
        )

        tool_name = meta.name
        self._metadata_cache[tool_name] = enhanced

        # Index by Category
        if meta.category not in self._category_index:
            self._category_index[meta.category] = set()
        self._category_index[meta.category].add(tool_name)

        # Index by Permission Level
        if meta.permission_level not in self._permission_index:
            self._permission_index[meta.permission_level] = set()
        self._permission_index[meta.permission_level].add(tool_name)

        # Index by Tags
        for tag in meta.tags:
            tag_clean = tag.lower().strip()
            if tag_clean not in self._tag_index:
                self._tag_index[tag_clean] = set()
            self._tag_index[tag_clean].add(tool_name)

        logger.debug(f"Indexed metadata for tool '{tool_name}'.")
        return enhanced

    def remove_tool(self, tool_name: str) -> None:
        """Removes tool from all metadata indexes."""
        if tool_name in self._metadata_cache:
            meta = self._metadata_cache.pop(tool_name)
            if meta.category in self._category_index:
                self._category_index[meta.category].discard(tool_name)
            if meta.permission_level in self._permission_index:
                self._permission_index[meta.permission_level].discard(tool_name)
            for tag in meta.tags:
                tag_clean = tag.lower().strip()
                if tag_clean in self._tag_index:
                    self._tag_index[tag_clean].discard(tool_name)

    def get_metadata(self, tool_name: str) -> Optional[EnhancedToolMetadata]:
        """Retrieves cached metadata for a tool."""
        return self._metadata_cache.get(tool_name)

    def search_by_tag(self, tag: str) -> Set[str]:
        """Returns set of tool names associated with a tag."""
        return self._tag_index.get(tag.lower().strip(), set())

    def search_by_category(self, category: ToolCategory) -> Set[str]:
        """Returns set of tool names belonging to a category."""
        return self._category_index.get(category, set())

    def search_by_permission(self, permission_level: PermissionLevel) -> Set[str]:
        """Returns set of tool names matching a permission level."""
        return self._permission_index.get(permission_level, set())

    def clear(self) -> None:
        """Clears all metadata caches."""
        self._metadata_cache.clear()
        self._category_index.clear()
        self._permission_index.clear()
        self._tag_index.clear()
