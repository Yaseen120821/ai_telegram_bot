"""
tests/test_tool_framework.py - Tool Calling Framework Unit & Integration Test Suite
===================================================================================

Verifies:
1. BaseTool interface and subclassing contract.
2. Dynamic Registration, duplicate detection, list/lookup, enable/disable in ToolRegistry.
3. Security level verification and permission enforcement in PermissionManager.
4. Intent Detection, candidate scoring, manual override, and routing thresholds in ToolRouter.
5. Isolated Tool Execution, parameter validation, timeout handling, retries in ToolExecutor.
6. Unified end-to-end pipeline in ToolManager.
"""

import sys
import os
import time
import unittest
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools import (
    BaseTool,
    ToolCategory,
    PermissionLevel,
    ExecutionStatus,
    ToolMetadata,
    ExecutionContext,
    ToolRequest,
    ToolRegistry,
    PermissionManager,
    ToolRouter,
    ToolExecutor,
    ToolManager,
    DuplicateToolException,
    ToolNotFoundException,
    ToolValidationException,
    ToolDisabledException,
    get_tool_config,
    set_tool_config,
    ToolFrameworkConfig
)


# -----------------------------------------------------------------------------
# Mock Tools for Testing
# -----------------------------------------------------------------------------

class MockCalculatorTool(BaseTool):
    """Safe tool that calculates math expressions."""
    def __init__(self):
        super().__init__(
            name="mock_calculator",
            description="Calculates arithmetic expressions and returns numbers.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        )
        self.metadata.tags = ["math", "calculator", "arithmetic", "calculate"]

    def _run(self, parameters: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        expr = parameters.get("expression", "0")
        return eval(expr, {"__builtins__": None}, {})


class MockDangerousTool(BaseTool):
    """Dangerous tool requiring user confirmation."""
    def __init__(self):
        super().__init__(
            name="mock_reboot",
            description="Reboots the server system.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={
                "type": "object",
                "properties": {
                    "force": {"type": "boolean"}
                }
            }
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        return "System reboot initiated successfully."


class MockSlowTool(BaseTool):
    """Tool that sleeps to trigger timeout errors."""
    def __init__(self):
        super().__init__(
            name="mock_slow",
            description="Simulates a slow long-running operation.",
            category=ToolCategory.AUTOMATION,
            permission_level=PermissionLevel.SAFE,
            timeout_seconds=0.2
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[ExecutionContext] = None) -> Any:
        time.sleep(0.5)
        return "Completed after delay."


class TestToolCallingFramework(unittest.TestCase):
    def setUp(self):
        """Reset registry singleton before each test."""
        self.registry = ToolRegistry.get_instance()
        self.registry.clear()
        self.perm_manager = PermissionManager()
        self.executor = ToolExecutor(registry=self.registry, permission_manager=self.perm_manager)
        self.router = ToolRouter(registry=self.registry)
        self.manager = ToolManager(
            registry=self.registry,
            router=self.router,
            permission_manager=self.perm_manager,
            executor=self.executor
        )

    # 1. Test Registry Operations & Duplicate Registration
    def test_tool_registration_and_lookup(self):
        calc_tool = MockCalculatorTool()
        self.registry.register(calc_tool)

        self.assertTrue(self.registry.has_tool("mock_calculator"))
        retrieved = self.registry.get_tool("mock_calculator")
        self.assertEqual(retrieved.name, "mock_calculator")

        # Test duplicate registration rejection
        with self.assertRaises(DuplicateToolException):
            self.registry.register(MockCalculatorTool())

    # 2. Test Enable / Disable Tool
    def test_enable_disable_tool(self):
        calc_tool = MockCalculatorTool()
        self.registry.register(calc_tool)
        self.assertTrue(calc_tool.is_enabled)

        self.registry.disable_tool("mock_calculator")
        self.assertFalse(calc_tool.is_enabled)

        # Execution should be rejected when tool is disabled
        request = ToolRequest(tool_name="mock_calculator", parameters={"expression": "5 + 5"})
        response = self.executor.execute(request)
        self.assertEqual(response.status, ExecutionStatus.FAILURE)

        self.registry.enable_tool("mock_calculator")
        self.assertTrue(calc_tool.is_enabled)

    # 3. Test Parameter Validation Failure
    def test_parameter_validation_failure(self):
        calc_tool = MockCalculatorTool()
        self.registry.register(calc_tool)

        # Missing required parameter "expression"
        request = ToolRequest(tool_name="mock_calculator", parameters={})
        response = self.executor.execute(request)
        self.assertEqual(response.status, ExecutionStatus.VALIDATION_ERROR)

    # 4. Test Permission Failure handling
    def test_permission_failure_and_confirmation(self):
        danger_tool = MockDangerousTool()
        self.registry.register(danger_tool)

        # Case A: User NOT confirmed -> DENIED
        req_unconfirmed = ToolRequest(tool_name="mock_reboot", parameters={"force": True}, user_confirmed=False)
        resp_unconfirmed = self.executor.execute(req_unconfirmed)
        self.assertEqual(resp_unconfirmed.status, ExecutionStatus.PERMISSION_DENIED)
        self.assertFalse(resp_unconfirmed.permission_result.is_allowed)

        # Case B: User confirmed -> GRANTED
        req_confirmed = ToolRequest(tool_name="mock_reboot", parameters={"force": True}, user_confirmed=True)
        resp_confirmed = self.executor.execute(req_confirmed)
        self.assertEqual(resp_confirmed.status, ExecutionStatus.SUCCESS)
        self.assertTrue(resp_confirmed.permission_result.is_allowed)

    # 5. Test Timeout Handling
    def test_tool_timeout_handling(self):
        slow_tool = MockSlowTool()
        self.registry.register(slow_tool)

        request = ToolRequest(tool_name="mock_slow", parameters={})
        response = self.executor.execute(request)
        self.assertEqual(response.status, ExecutionStatus.TIMEOUT)

    # 6. Test Single Tool Execution & Output Formatting
    def test_single_tool_execution(self):
        calc_tool = MockCalculatorTool()
        self.registry.register(calc_tool)

        request = ToolRequest(tool_name="mock_calculator", parameters={"expression": "10 * 4"})
        response = self.executor.execute(request)
        self.assertEqual(response.status, ExecutionStatus.SUCCESS)
        self.assertIn("Output:\n40", response.formatted_output)

    # 7. Test Intent Routing & Manual Override
    def test_intent_routing_and_override(self):
        calc_tool = MockCalculatorTool()
        self.registry.register(calc_tool)

        # Case A: Query matching tool description/keywords
        decision, _ = self.router.route_query("Please calculate 10 * 4")
        self.assertTrue(decision.should_call_tool)
        self.assertEqual(decision.selected_tool.name, "mock_calculator")

        # Case B: Unrelated query -> Fallback to No Tool
        decision_fallback, _ = self.router.route_query("What is the capital of France?")
        self.assertFalse(decision_fallback.should_call_tool)

        # Case C: Manual Override
        decision_override, _ = self.router.route_query("Tell me a story", manual_override_tool="mock_calculator")
        self.assertTrue(decision_override.should_call_tool)
        self.assertEqual(decision_override.selected_tool.name, "mock_calculator")

    # 8. Test End-to-End Facade (ToolManager)
    def test_tool_manager_end_to_end(self):
        calc_tool = MockCalculatorTool()
        self.manager.register_tool(calc_tool)

        decision, response = self.manager.route_and_execute(
            query="Please calculate 100 / 5",
            parameters={"expression": "100 / 5"}
        )
        self.assertTrue(decision.should_call_tool)
        self.assertIsNotNone(response)
        self.assertEqual(response.status, ExecutionStatus.SUCCESS)
        self.assertIn("20.0", response.formatted_output)


if __name__ == "__main__":
    unittest.main()
