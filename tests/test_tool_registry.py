"""
tests/test_tool_registry.py - Tool Registry & Discovery Diagnostic Test Suite
=============================================================================

Verifies:
1. Dynamic Tool Discovery across directory paths.
2. Safe instantiation & metadata validation in ToolLoader.
3. Metadata caching & multi-criteria search in MetadataManager.
4. Registration lifecycle, duplicate detection, enable/disable in RegistryManager.
5. Error recovery (broken imports/classes skipped safely).
6. Startup health audit report generation.
"""

import sys
import os
import unittest
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel, ToolState
from app.tools.registry import (
    RegistryManager,
    ToolDiscovery,
    ToolLoader,
    MetadataManager,
    RegistryValidator,
    RegistryState,
    RegistrationStatus,
    ToolSearchResult,
    RegistryStatistics,
    RegistryHealth,
    RegistryReport,
    ToolRegistryConfig
)
from app.tools.tool_exceptions import DuplicateToolException, ToolNotFoundException, ToolValidationException


# -----------------------------------------------------------------------------
# Mock Tools for Registry Testing
# -----------------------------------------------------------------------------

class MockSystemDiagnosticsTool(BaseTool):
    """Diagnostic tool for testing system category."""
    def __init__(self):
        super().__init__(
            name="mock_diagnostics",
            description="Inspects memory usage and CPU diagnostic statistics.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={"type": "object", "properties": {"verbose": {"type": "boolean"}}}
        )
        self.metadata.tags = ["system", "metrics", "diagnostics"]

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return {"status": "ok", "cpu": "12%"}


class MockFileWriteTool(BaseTool):
    """Filesystem tool requiring user confirmation."""
    def __init__(self):
        super().__init__(
            name="mock_write_file",
            description="Writes text data to target filesystem path.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={"type": "object", "properties": {"filename": {"type": "string"}}}
        )
        self.metadata.tags = ["filesystem", "storage", "io"]

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return "File written successfully."


class MockInvalidShortDescTool(BaseTool):
    """Tool with invalid (too short) description."""
    def __init__(self):
        super().__init__(
            name="mock_invalid",
            description="Short", # Under 10 chars
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return None


class TestToolRegistryFramework(unittest.TestCase):
    def setUp(self):
        """Reset registry singleton state before each test."""
        self.registry = RegistryManager.get_instance()
        self.registry.clear()

    # 1. Test Manual Registration & Lookup
    def test_manual_registration_and_lookup(self):
        diag_tool = MockSystemDiagnosticsTool()
        record = self.registry.register_tool(diag_tool)

        self.assertEqual(record.status, RegistrationStatus.SUCCESS)
        self.assertTrue(self.registry.has_tool("mock_diagnostics"))

        retrieved = self.registry.get_tool("mock_diagnostics")
        self.assertEqual(retrieved.name, "mock_diagnostics")

    # 2. Test Duplicate Registration Protection
    def test_duplicate_registration_protection(self):
        diag_tool = MockSystemDiagnosticsTool()
        self.registry.register_tool(diag_tool)

        with self.assertRaises(DuplicateToolException):
            self.registry.register_tool(MockSystemDiagnosticsTool())

    # 3. Test Invalid Metadata Validation
    def test_invalid_metadata_validation(self):
        invalid_tool = MockInvalidShortDescTool()
        with self.assertRaises(ToolValidationException):
            self.registry.register_tool(invalid_tool)

    # 4. Test Enable / Disable Tool State
    def test_enable_disable_tool_state(self):
        diag_tool = MockSystemDiagnosticsTool()
        self.registry.register_tool(diag_tool)

        self.assertTrue(diag_tool.is_enabled)
        self.registry.disable_tool("mock_diagnostics")
        self.assertFalse(diag_tool.is_enabled)

        # Filtering by only_enabled should omit disabled tool
        enabled_tools = self.registry.list_tools(only_enabled=True)
        self.assertEqual(len(enabled_tools), 0)

        self.registry.enable_tool("mock_diagnostics")
        self.assertTrue(diag_tool.is_enabled)
        self.assertEqual(len(self.registry.list_tools(only_enabled=True)), 1)

    # 5. Test Multi-Criteria Tool Search
    def test_multi_criteria_tool_search(self):
        self.registry.register_tool(MockSystemDiagnosticsTool())
        self.registry.register_tool(MockFileWriteTool())

        # Search by tag
        res_tag = self.registry.search_tools(tag="metrics")
        self.assertEqual(res_tag.total_found, 1)
        self.assertEqual(res_tag.matched_tools[0].name, "mock_diagnostics")

        # Search by category
        res_cat = self.registry.search_tools(category=ToolCategory.FILESYSTEM)
        self.assertEqual(res_cat.total_found, 1)
        self.assertEqual(res_cat.matched_tools[0].name, "mock_write_file")

        # Search by permission level
        res_perm = self.registry.search_tools(permission_level=PermissionLevel.CONFIRMATION_REQUIRED)
        self.assertEqual(res_perm.total_found, 1)
        self.assertEqual(res_perm.matched_tools[0].name, "mock_write_file")

        # Query text match
        res_text = self.registry.search_tools(query="CPU")
        self.assertEqual(res_text.total_found, 1)
        self.assertEqual(res_text.matched_tools[0].name, "mock_diagnostics")

    # 6. Test Registry Telemetry & Health Audit Report
    def test_registry_telemetry_and_health(self):
        self.registry.register_tool(MockSystemDiagnosticsTool())
        self.registry.register_tool(MockFileWriteTool())

        stats: RegistryStatistics = self.registry.get_registry_statistics()
        self.assertEqual(stats.total_registered, 2)
        self.assertEqual(stats.enabled_count, 2)
        self.assertEqual(stats.category_breakdown.get("system"), 1)
        self.assertEqual(stats.category_breakdown.get("filesystem"), 1)

        health: RegistryHealth = self.registry.get_registry_health()
        self.assertEqual(health.loaded_tools_count, 2)
        self.assertEqual(health.failed_imports_count, 0)

    # 7. Test Loader Graceful Error Recovery
    def test_loader_graceful_error_recovery(self):
        loader = ToolLoader()
        existing = {}
        
        # Passing an invalid class (not a BaseTool subclass)
        class NotATool:
            pass

        record = loader.load_tool_class(NotATool, existing)
        self.assertEqual(record.status, RegistrationStatus.INVALID_METADATA)
        self.assertIsNotNone(record.error_message)

    # 8. Test Dynamic Discovery Initialization
    def test_dynamic_discovery_initialization(self):
        report = self.registry.initialize()
        self.assertIsNotNone(report)
        self.assertEqual(self.registry.state, RegistryState.READY)


if __name__ == "__main__":
    unittest.main()
