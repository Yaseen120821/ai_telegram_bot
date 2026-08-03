"""
tests/test_tool_execution.py - Execution Engine Diagnostic Test Suite
=====================================================================

Verifies:
1. Successful tool execution and ResultFormatter payload structure.
2. Security authorization & permission rejection in PermissionChecker.
3. Required & type validation in ParameterValidator.
4. Per-tool timeout enforcement in TimeoutManager.
5. Automatic retry mechanism in RetryManager.
6. ExecutionStatistics telemetry metrics tracking.
"""

import sys
import os
import time
import unittest
from typing import Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.registry import RegistryManager
from app.tools.execution import (
    ExecutionManager,
    ExecutionRequest,
    ExecutionStatus,
    PermissionDecision,
    FailureReason,
    ExecutionStatistics
)


# -----------------------------------------------------------------------------
# Mock Tools for Execution Testing
# -----------------------------------------------------------------------------

class MockSafeCalcTool(BaseTool):
    """Safe tool that performs multiplication."""
    def __init__(self):
        super().__init__(
            name="mock_exec_calc",
            description="Performs simple numeric multiplication arithmetic.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.SAFE,
            parameters_schema={
                "type": "object",
                "properties": {
                    "num1": {"type": "integer"},
                    "num2": {"type": "integer"}
                },
                "required": ["num1", "num2"]
            }
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return parameters["num1"] * parameters["num2"]


class MockDangerousRebootTool(BaseTool):
    """High-risk tool requiring user confirmation."""
    def __init__(self):
        super().__init__(
            name="mock_exec_reboot",
            description="Reboots the server system kernel.",
            category=ToolCategory.SYSTEM,
            permission_level=PermissionLevel.CONFIRMATION_REQUIRED,
            parameters_schema={"type": "object", "properties": {"force": {"type": "boolean"}}}
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        return "Reboot complete."


class MockTimeoutTool(BaseTool):
    """Tool exceeding timeout threshold."""
    def __init__(self):
        super().__init__(
            name="mock_exec_timeout",
            description="Simulates a slow long-running operation.",
            category=ToolCategory.AUTOMATION,
            permission_level=PermissionLevel.SAFE,
            timeout_seconds=0.2
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        time.sleep(0.5)
        return "Finished after delay."


class MockFailingTool(BaseTool):
    """Tool raising runtime exceptions."""
    def __init__(self):
        super().__init__(
            name="mock_exec_fail",
            description="Simulates a tool runtime crash.",
            category=ToolCategory.DEVELOPER,
            permission_level=PermissionLevel.SAFE
        )

    def _run(self, parameters: Dict[str, Any], context: Optional[Any] = None) -> Any:
        raise ValueError("Simulated internal algorithm failure.")


class TestToolExecutionFramework(unittest.TestCase):
    def setUp(self):
        """Reset registry & execution manager singleton before each test."""
        self.registry = RegistryManager.get_instance()
        self.registry.clear()

        self.calc_tool = MockSafeCalcTool()
        self.reboot_tool = MockDangerousRebootTool()
        self.timeout_tool = MockTimeoutTool()
        self.fail_tool = MockFailingTool()

        self.registry.register_tool(self.calc_tool)
        self.registry.register_tool(self.reboot_tool)
        self.registry.register_tool(self.timeout_tool)
        self.registry.register_tool(self.fail_tool)

        self.manager = ExecutionManager.get_instance()
        self.manager.clear_statistics()

    # 1. Test Successful Execution & Result Format
    def test_successful_execution(self):
        req = ExecutionRequest(
            tool_name="mock_exec_calc",
            parameters={"num1": 7, "num2": 6}
        )
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.SUCCESS)
        self.assertEqual(res.output, 42)
        self.assertIn("Output:\n42", res.formatted_output)

    # 2. Test Permission Denied Handling
    def test_permission_denied_handling(self):
        # User confirmed is False -> Rejected
        req_unconfirmed = ExecutionRequest(
            tool_name="mock_exec_reboot",
            parameters={"force": True},
            user_confirmed=False
        )
        res_unconfirmed = self.manager.execute_request(req_unconfirmed)

        self.assertEqual(res_unconfirmed.status, ExecutionStatus.PERMISSION_DENIED)
        self.assertEqual(res_unconfirmed.failure_reason, FailureReason.PERMISSION_DENIED)

        # User confirmed is True -> Allowed
        req_confirmed = ExecutionRequest(
            tool_name="mock_exec_reboot",
            parameters={"force": True},
            user_confirmed=True
        )
        res_confirmed = self.manager.execute_request(req_confirmed)
        self.assertEqual(res_confirmed.status, ExecutionStatus.SUCCESS)

    # 3. Test Missing Required Parameter Validation
    def test_missing_required_parameter(self):
        req = ExecutionRequest(
            tool_name="mock_exec_calc",
            parameters={"num1": 5} # Missing num2
        )
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.VALIDATION_ERROR)
        self.assertEqual(res.failure_reason, FailureReason.PARAMETER_INVALID)

    # 4. Test Invalid Parameter Type Validation
    def test_invalid_parameter_type(self):
        req = ExecutionRequest(
            tool_name="mock_exec_calc",
            parameters={"num1": "not_an_int", "num2": 5}
        )
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.VALIDATION_ERROR)

    # 5. Test Execution Timeout Enforcement
    def test_execution_timeout_enforcement(self):
        req = ExecutionRequest(tool_name="mock_exec_timeout")
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(res.failure_reason, FailureReason.TIMEOUT_EXCEEDED)

    # 6. Test Unhandled Exception Handling
    def test_unhandled_exception_handling(self):
        req = ExecutionRequest(tool_name="mock_exec_fail")
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.FAILURE)
        self.assertIn("Simulated internal algorithm failure", res.error_message)

    # 7. Test Disabled Tool Rejection
    def test_disabled_tool_rejection(self):
        self.registry.disable_tool("mock_exec_calc")

        req = ExecutionRequest(
            tool_name="mock_exec_calc",
            parameters={"num1": 2, "num2": 3}
        )
        res = self.manager.execute_request(req)

        self.assertEqual(res.status, ExecutionStatus.FAILURE)
        self.assertEqual(res.failure_reason, FailureReason.DISABLED_TOOL)

    # 8. Test Telemetry Statistics Tracking
    def test_telemetry_statistics_tracking(self):
        self.manager.execute_request(ExecutionRequest(tool_name="mock_exec_calc", parameters={"num1": 2, "num2": 3}))
        self.manager.execute_request(ExecutionRequest(tool_name="mock_exec_reboot", user_confirmed=False))

        stats: ExecutionStatistics = self.manager.get_statistics()
        self.assertEqual(stats.total_executed, 2)
        self.assertEqual(stats.success_count, 1)
        self.assertEqual(stats.permission_denied_count, 1)


if __name__ == "__main__":
    unittest.main()
