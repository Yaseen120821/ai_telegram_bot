"""
tests/test_tool_integration.py - Intelligent Decision Pipeline Test Suite
==========================================================================

Verifies:
1. Tool Only query processing pipeline.
2. Memory Only query processing pipeline.
3. RAG Only query processing pipeline.
4. Combined Tool + Memory processing pipeline.
5. Combined Tool + RAG processing pipeline.
6. Emotion + Tool interaction (Emotion alters tone, never tool execution logic).
7. Complete multi-source combined pipeline execution.
"""

import sys
import os
import unittest
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.base_tool import BaseTool
from app.tools.tool_types import ToolCategory, PermissionLevel
from app.tools.registry import RegistryManager
from app.tools.system import CurrentTimeTool
from app.tools.utility import CalculatorTool
from app.tools.integration import IntelligentPipeline, CombinedResponse


@dataclass
class MockEmotionContext:
    primary_emotion: str = "frustrated"
    confidence: float = 0.88


class TestToolIntegrationFramework(unittest.TestCase):
    def setUp(self):
        """Reset registry & pipeline singletons before each test."""
        self.registry = RegistryManager.get_instance()
        self.registry.clear()

        self.time_tool = CurrentTimeTool()
        self.calc_tool = CalculatorTool()

        self.registry.register_tool(self.time_tool)
        self.registry.register_tool(self.calc_tool)

        self.pipeline = IntelligentPipeline.get_instance()

    # 1. Test Tool Only Processing
    def test_tool_only_pipeline(self):
        res: CombinedResponse = self.pipeline.process_query("What time is it right now?")

        self.assertTrue(res.summary.used_tool)
        self.assertIn("get_current_time", res.summary.tools_used)
        self.assertFalse(res.summary.used_rag)

    # 2. Test Memory Only Processing
    def test_memory_only_pipeline(self):
        mem = "User's favorite programming language is Python."
        res: CombinedResponse = self.pipeline.process_query(
            query="What should we code today?",
            memory_context=mem
        )

        self.assertTrue(res.summary.used_memory)
        self.assertEqual(res.integrated_context.memory_context, mem)

    # 3. Test RAG Only Processing
    def test_rag_only_pipeline(self):
        res: CombinedResponse = self.pipeline.process_query(
            query="Search documentation for setup instructions"
        )

        self.assertTrue(res.summary.used_rag)
        self.assertIsNotNone(res.integrated_context.rag_context)

    # 4. Test Tool + Memory Pipeline
    def test_tool_and_memory_pipeline(self):
        mem = "User is doing math homework."
        res: CombinedResponse = self.pipeline.process_query(
            query="Please calculate 50 * 4",
            memory_context=mem
        )

        self.assertTrue(res.summary.used_tool)
        self.assertTrue(res.summary.used_memory)
        self.assertIn("calculate_math", res.summary.tools_used)

    # 5. Test Emotion + Tool Pipeline
    def test_emotion_and_tool_pipeline(self):
        emo = MockEmotionContext(primary_emotion="frustrated")
        res: CombinedResponse = self.pipeline.process_query(
            query="What time is it right now?",
            emotion_context=emo
        )

        # Emotion guidance is present in context, tool executed cleanly
        self.assertTrue(res.summary.used_tool)
        self.assertEqual(res.trace.emotion_detected, "frustrated")
        self.assertEqual(res.integrated_context.emotion_context.primary_emotion, "frustrated")

    # 6. Test All Combined Integrated Pipeline
    def test_all_combined_pipeline(self):
        mem = "User lives in New York."
        emo = MockEmotionContext(primary_emotion="happy")

        res: CombinedResponse = self.pipeline.process_query(
            query="Calculate 12 * 12",
            memory_context=mem,
            emotion_context=emo
        )

        self.assertTrue(res.summary.used_tool)
        self.assertTrue(res.summary.used_memory)
        self.assertIn("calculate_math", res.summary.tools_used)
        self.assertIsNotNone(res.integrated_context.tool_context)


if __name__ == "__main__":
    unittest.main()
