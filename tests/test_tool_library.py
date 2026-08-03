"""
tests/test_tool_library.py - Professional Local Tool Library Test Suite
=====================================================================

Verifies:
1. System Tools execution (Time, Date, DateTime, Timezone, Info, Uptime).
2. Utility Tools execution (Calculator AST, UUID, Password, Hashes, Base64).
3. Filesystem Tools execution & Path Traversal Security blocks.
4. Developer Tools execution (Line count, TODO search, Project Stats).
5. Productivity Tools execution (Notes, Todo, Reminders, Stopwatch).
6. AI Text Analysis Tools execution (Summarizer, Stats, Sentiment).
7. Security blocks on restricted extensions and directory escape attempts.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.system import CurrentTimeTool, CurrentDateTool, CurrentDateTimeTool, SystemInfoTool, UptimeTool
from app.tools.utility import CalculatorTool, UUIDGeneratorTool, PasswordGeneratorTool, HashGeneratorTool, Base64EncodeTool, Base64DecodeTool
from app.tools.filesystem import ReadFileTool, WriteFileTool, AppendFileTool, DeleteFileTool, ListDirectoryTool, FileMetadataTool
from app.tools.developer import CountLinesTool, FindTodoTool, ProjectStatisticsTool
from app.tools.productivity import NotesTool, TodoTool, StopwatchTool
from app.tools.ai import TextSummarizerTool, TextStatisticsTool, SentimentAnalysisTool
from app.tools.tool_exceptions import ToolExecutionException, PermissionDeniedException


class TestToolLibraryFramework(unittest.TestCase):
    def setUp(self):
        """Prepare temporary test directory inside workspace."""
        self.test_filename = "test_workspace_file.txt"
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def tearDown(self):
        """Clean up temporary test file."""
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    # 1. System Tools
    def test_system_tools(self):
        time_res = CurrentTimeTool()._run({})
        self.assertIsInstance(time_res, str)

        date_res = CurrentDateTool()._run({})
        self.assertIn("-", date_res)

        datetime_res = CurrentDateTimeTool()._run({})
        self.assertIn("T", datetime_res)

        info_res = SystemInfoTool()._run({})
        self.assertIn("os", info_res)

        uptime_res = UptimeTool()._run({})
        self.assertIn("formatted_uptime", uptime_res)

    # 2. Utility Tools
    def test_utility_tools(self):
        calc_res = CalculatorTool()._run({"expression": "(10 + 5) * 4"})
        self.assertEqual(calc_res["result"], 60)

        uuid_res = UUIDGeneratorTool()._run({"count": 3})
        self.assertEqual(len(uuid_res["uuids"]), 3)

        pwd_res = PasswordGeneratorTool()._run({"length": 16})
        self.assertEqual(len(pwd_res["password"]), 16)

        hash_res = HashGeneratorTool()._run({"text": "sana_ai", "algorithm": "sha256"})
        self.assertEqual(len(hash_res["hash"]), 64)

        b64_enc = Base64EncodeTool()._run({"text": "Hello World"})
        self.assertEqual(b64_enc["encoded"], "SGVsbG8gV29ybGQ=")

        b64_dec = Base64DecodeTool()._run({"encoded_text": "SGVsbG8gV29ybGQ="})
        self.assertEqual(b64_dec["decoded"], "Hello World")

    # 3. Filesystem Tools & Path Traversal Security
    def test_filesystem_tools_and_security(self):
        # Write
        w_res = WriteFileTool()._run({"filename": self.test_filename, "content": "SANA AI Test Content"})
        self.assertEqual(w_res["status"], "saved")

        # Read
        r_res = ReadFileTool()._run({"filename": self.test_filename})
        self.assertEqual(r_res["content"], "SANA AI Test Content")

        # Append
        a_res = AppendFileTool()._run({"filename": self.test_filename, "content": "\nAppended Line"})
        self.assertEqual(a_res["status"], "appended")

        # Stat
        meta_res = FileMetadataTool()._run({"filename": self.test_filename})
        self.assertTrue(meta_res["is_file"])

        # Delete
        d_res = DeleteFileTool()._run({"filename": self.test_filename})
        self.assertEqual(d_res["status"], "deleted")

        # Security Block on Path Traversal
        with self.assertRaises(PermissionDeniedException):
            ReadFileTool()._run({"filename": "../../etc/passwd"})

        # Security Block on Blocked Extension
        with self.assertRaises(PermissionDeniedException):
            WriteFileTool()._run({"filename": "malware.exe", "content": "test"})

    # 4. Developer Tools
    def test_developer_tools(self):
        loc_res = CountLinesTool()._run({})
        self.assertIn("python_files", loc_res)

        todo_res = FindTodoTool()._run({})
        self.assertIn("count", todo_res)

        stats_res = ProjectStatisticsTool()._run({})
        self.assertIn("total_files", stats_res)

    # 5. Productivity Tools
    def test_productivity_tools(self):
        notes = NotesTool()
        notes._run({"action": "add", "title": "meeting", "content": "Discuss architecture."})
        n_res = notes._run({"action": "get", "title": "meeting"})
        self.assertEqual(n_res["content"], "Discuss architecture.")

        todos = TodoTool()
        todos._run({"action": "add", "item": "Write unit tests"})
        t_res = todos._run({"action": "list"})
        self.assertEqual(t_res["total"], 1)

        sw = StopwatchTool()
        sw._run({"action": "start"})
        sw_res = sw._run({"action": "read"})
        self.assertEqual(sw_res["status"], "running")

    # 6. AI Text Analysis Tools
    def test_ai_tools(self):
        sum_res = TextSummarizerTool()._run({"text": "SANA AI is an intelligent assistant. It supports tool calling. It runs locally.", "max_sentences": 2})
        self.assertEqual(sum_res["summary_sentences"], 2)

        stats_res = TextStatisticsTool()._run({"text": "Quick brown fox jumps over the lazy dog."})
        self.assertEqual(stats_res["word_count"], 8)

        sent_res = SentimentAnalysisTool()._run({"text": "SANA AI provides great excellent awesome results!"})
        self.assertEqual(sent_res["sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
