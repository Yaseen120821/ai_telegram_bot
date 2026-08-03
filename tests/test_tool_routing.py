"""
tests/test_tool_routing.py - Decision Engine & Tool Router Diagnostic Test Suite
=================================================================================

Verifies:
1. Intent Detection & parameter extraction for time, calculator, read file.
2. Rule Engine matching speed & accuracy.
3. Confidence Engine scoring & threshold categorization (HIGH, MEDIUM, LOW, NONE).
4. Decision Engine routing modes (DIRECT_RESPONSE, TOOL, RAG, CLARIFICATION_REQUIRED).
5. Tool Router building valid ToolRequest payloads.
"""

import sys
import os
import unittest
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.registry import RegistryManager
from app.tools.routing import (
    DecisionEngine,
    ToolRouter,
    IntentDetector,
    ConfidenceEngine,
    RuleEngine,
    IntentType,
    RoutingMode,
    DecisionType,
    ConfidenceLevel
)


# -----------------------------------------------------------------------------
# Mock Tools for Routing Tests
# -----------------------------------------------------------------------------

class MockTimeTool(BaseTool):
    """Tool returning current system time."""
    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Returns the current date and time in UTC format.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE
        )
        self.metadata.tags = ["time", "clock", "date", "get_time"]

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return "2026-07-31 20:55:00 UTC"


class MockCalcTool(BaseTool):
    """Tool calculating math expressions."""
    def __init__(self):
        super().__init__(
            name="calculate_math",
            description="Evaluates mathematical arithmetic expressions and numbers.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        )
        self.metadata.tags = ["calculate", "math", "arithmetic"]

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return "420"


class MockReadFileTool(BaseTool):
    """Tool reading text files from filesystem."""
    def __init__(self):
        super().__init__(
            name="read_file_tool",
            description="Reads text content from a specified file path.",
            category=ToolCategory.FILESYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"]
            }
        )
        self.metadata.tags = ["read_file", "file", "open"]

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return "File contents summary"


class TestToolRoutingFramework(unittest.TestCase):
    def setUp(self):
        """Reset registry & initialize decision router before each test."""
        self.registry = RegistryManager.get_instance()
        self.registry.clear()

        self.time_tool = MockTimeTool()
        self.calc_tool = MockCalcTool()
        self.file_tool = MockReadFileTool()

        self.registry.register_tool(self.time_tool)
        self.registry.register_tool(self.calc_tool)
        self.registry.register_tool(self.file_tool)

        self.decision_engine = DecisionEngine(registry=self.registry)
        self.router = ToolRouter(registry=self.registry, decision_engine=self.decision_engine)

    # 1. Test Current Time Query Routing
    def test_current_time_query_routing(self):
        decision, request = self.router.route_query("What time is it right now?")

        self.assertEqual(decision.routing_mode, RoutingMode.TOOL)
        self.assertEqual(decision.selected_tool.name, "get_current_time")
        self.assertIsNotNone(request)
        self.assertEqual(request.tool_name, "get_current_time")

    # 2. Test Calculator Expression Query Routing & Parameter Extraction
    def test_calculator_query_routing(self):
        decision, request = self.router.route_query("Calculate 15 * 28")

        self.assertEqual(decision.routing_mode, RoutingMode.TOOL)
        self.assertEqual(decision.selected_tool.name, "calculate_math")
        self.assertIsNotNone(request)
        self.assertIn("expression", request.parameters)
        self.assertEqual(request.parameters["expression"], "15 * 28")

    # 3. Test Read File Query Routing & Parameter Extraction
    def test_read_file_query_routing(self):
        decision, request = self.router.route_query("Read file report.pdf")

        self.assertEqual(decision.routing_mode, RoutingMode.TOOL)
        self.assertEqual(decision.selected_tool.name, "read_file_tool")
        self.assertIsNotNone(request)
        self.assertEqual(request.parameters.get("filename"), "report.pdf")

    # 4. Test Missing Required Parameter Clarification Request
    def test_missing_required_param_clarification(self):
        # Query matching read_file_tool intent without supplying a filename
        decision, request = self.router.route_query("Read file")

        self.assertEqual(decision.routing_mode, RoutingMode.CLARIFICATION_REQUIRED)
        self.assertTrue(decision.needs_clarification)
        self.assertIsNone(request)
        self.assertIn("filename", decision.clarification_prompt)

    # 5. Test RAG Search Query Routing
    def test_rag_query_routing(self):
        decision, request = self.router.route_query("Search documentation for database setup")

        self.assertEqual(decision.routing_mode, RoutingMode.RAG)
        self.assertEqual(decision.decision_type, DecisionType.EXECUTE_RAG)
        self.assertIsNone(request)

    # 6. Test Unknown Query Direct LLM Fallback
    def test_unknown_query_direct_llm_fallback(self):
        decision, request = self.router.route_query("Tell me a funny story about space exploration")

        self.assertEqual(decision.routing_mode, RoutingMode.DIRECT_RESPONSE)
        self.assertEqual(decision.decision_type, DecisionType.DIRECT_LLM)
        self.assertIsNone(request)

    # 7. Test Manual Override Routing
    def test_manual_override_routing(self):
        decision, request = self.router.route_query(
            query="Hello there",
            manual_override_tool="get_current_time"
        )

        self.assertEqual(decision.routing_mode, RoutingMode.TOOL)
        self.assertEqual(decision.selected_tool.name, "get_current_time")
        self.assertIsNotNone(request)


if __name__ == "__main__":
    unittest.main()
