import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


class AgentCoreTests(unittest.TestCase):
    def test_compact_conv_includes_all_eligible_history(self):
        messages = [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"unique-history-{index}"}
            for index in range(50)
        ]
        prompts = []

        async def fake_complete(model, history, **kwargs):
            prompts.append(history[0]["content"])
            return "summary"

        with patch.object(main, "_llm_complete", side_effect=fake_complete):
            replacement, summary = asyncio.run(
                main._compact_conv(messages, keep_last=6, model="test-model")
            )

        self.assertEqual(summary, "summary")
        self.assertEqual(replacement[0]["role"], "user")
        self.assertIn("unique-history-0", prompts[0])
        self.assertIn("unique-history-43", prompts[0])

    def test_json_list_cache_returns_independent_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text("[1]")
            main._JSON_LIST_CACHE.clear()
            first = main._load_json_list(path)
            first.append(2)
            self.assertEqual(main._load_json_list(path), [1])

    def test_json_list_cache_does_not_cache_parse_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text("{broken")
            main._JSON_LIST_CACHE.clear()
            self.assertEqual(main._load_json_list(path), [])
            path.write_text("[3]")
            os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 1))
            self.assertEqual(main._load_json_list(path), [3])

    def test_task_schema_and_instructions_match(self):
        task = next(tool["function"] for tool in main.TOOLS if tool["function"]["name"] == "task")
        self.assertIn("agent", task["parameters"]["properties"])
        instructions = main._agent_tool_instructions()
        self.assertNotIn("run tools", instructions)
        self.assertIn("run_command", instructions)

    def test_search_accepts_mounted_drive_absolute_paths(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as mounted:
            target = Path(mounted) / "notes.txt"
            target.write_text("mounted content")
            request = main.FileSearchRequest(pattern="*", path=mounted)
            with patch.object(main, "BASE_PROJECTS", home), patch.object(
                main, "_wide_scan_roots", return_value=[Path(home), Path(mounted)]
            ):
                result = asyncio.run(main.search_files(request))
            self.assertTrue(result["results"])
            self.assertEqual(result["results"][0]["path"], str(target))

    def test_native_tool_call_converts_to_agent_fence(self):
        call = {"name": "read_file", "input": json.dumps({"path": "notes.txt"})}
        self.assertEqual(
            main._native_tool_call_to_fence(call),
            '```tool\n{"name": "read_file", "arguments": {"path": "notes.txt"}}\n```',
        )

    def test_provider_usage_is_normalized_for_compaction(self):
        usage = main._normalize_provider_usage({"prompt_tokens": 12, "completion_tokens": 4})
        self.assertEqual(usage["prompt_eval_count"], 12)
        self.assertEqual(usage["eval_count"], 4)

    def test_auto_compaction_allows_tool_result_turns(self):
        conv = [{"role": "tool", "content": "result"}] * 10
        self.assertTrue(
            main._should_auto_compact(
                conv,
                usage={"prompt_eval_count": 80, "eval_count": 1},
                window=100,
                turns_since_compact=3,
                turn=1,
            )
        )

    def test_repeat_signature_uses_full_canonical_arguments(self):
        left = {"content": "a" * 3000}
        right = {"content": "b" * 3000}
        self.assertNotEqual(main._tool_call_signature("write_file", left), main._tool_call_signature("write_file", right))


if __name__ == "__main__":
    unittest.main()
