"""
app/tools/registry/tool_discovery.py - Dynamic Tool Discovery Scanner
======================================================================
Scans target filesystem directories, inspects Python modules dynamically,
locates valid BaseTool subclasses, and returns candidate classes for registration.
"""

import os
import sys
import glob
import importlib
import logging
from typing import List, Type, Set, Optional
from app.tools.base_tool import BaseTool
from app.tools.registry.registry_config import get_registry_config, ToolRegistryConfig
from app.tools.registry.registry_utils import is_basetool_subclass, file_path_to_module_name

logger = logging.getLogger("sana_ai.tools.registry.discovery")


class ToolDiscovery:
    """
    Automated dynamic discovery engine.
    
    Scans configured filesystem paths for BaseTool implementation classes,
    handling module import exceptions safely without crashing application startup.
    """

    def __init__(self, config: Optional[ToolRegistryConfig] = None):
        self.config = config or get_registry_config()

    def discover_tools(self, project_root: Optional[str] = None) -> List[Type[BaseTool]]:
        """
        Scans discovery paths and returns list of unique BaseTool implementation classes.
        """
        if project_root is None:
            project_root = os.getcwd()

        discovered_classes: Set[Type[BaseTool]] = set()
        logger.info(f"Starting Tool Discovery scan [Mode: {self.config.discovery_mode.value}]...")

        for relative_path in self.config.discovery_paths:
            full_path = os.path.join(project_root, relative_path)
            if not os.path.exists(full_path):
                logger.debug(f"Discovery path does not exist, skipping: {full_path}")
                continue

            classes_in_path = self._scan_directory(full_path, project_root)
            discovered_classes.update(classes_in_path)

        logger.info(f"Discovery completed: Found {len(discovered_classes)} candidate tool class(es).")
        return list(discovered_classes)

    def _scan_directory(self, dir_path: str, project_root: str) -> List[Type[BaseTool]]:
        """Recursively scans a directory for .py files and extracts BaseTool subclasses."""
        found: List[Type[BaseTool]] = []

        # Find all python files in directory recursively
        search_pattern = os.path.join(dir_path, "**", "*.py")
        py_files = glob.glob(search_pattern, recursive=True)

        for file_path in py_files:
            filename = os.path.basename(file_path)
            if filename in self.config.excluded_filenames:
                continue

            classes = self._inspect_file(file_path, project_root)
            found.extend(classes)

        return found

    def _inspect_file(self, file_path: str, project_root: str) -> List[Type[BaseTool]]:
        """Dynamically imports a python file and inspects it for BaseTool subclasses."""
        discovered: List[Type[BaseTool]] = []
        module_name = file_path_to_module_name(file_path, project_root)

        try:
            # Import module dynamically
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)

            # Inspect all attributes in module
            for attr_name in dir(module):
                attr_value = getattr(module, attr_name)
                if is_basetool_subclass(attr_value):
                    # Check if the class was defined in this module (avoid re-registering imported bases)
                    if getattr(attr_value, '__module__', '') == module_name:
                        discovered.append(attr_value)
                        logger.debug(f"Discovered tool class '{attr_value.__name__}' in module '{module_name}'.")

        except Exception as err:
            logger.warning(
                f"Failed to import/inspect module file '{file_path}' (Module: '{module_name}'): {err}. "
                "Skipping file safely without halting startup."
            )

        return discovered
