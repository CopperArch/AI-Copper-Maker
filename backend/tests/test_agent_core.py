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

    def test_slugify_provider_name_matches_existing_key_convention(self):
        self.assertEqual(main._slugify_provider_name("NVIDIA NIM"), "nvidianim")
        self.assertEqual(main._slugify_provider_name("Ollama Cloud"), "ollamacloud")
        self.assertEqual(main._slugify_provider_name("Together AI (Free)"), "togetheraifree")

    def test_register_free_llm_api_providers_wires_new_gateway(self):
        fixture = {
            "providers": [
                {
                    "name": "Testonly Cloud",
                    "category": "inference_provider",
                    "baseUrl": "https://api.testonly.example/v1",
                    "models": [{"id": "testonly-model-1"}, {"id": "testonly-model-2"}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.json"
            data_path.write_text(json.dumps(fixture))
            with (
                patch.object(main, "_FREE_LLM_APIS_DATA", data_path),
                patch.object(main, "CLOUD_PROVIDERS", dict(main.CLOUD_PROVIDERS)),
                patch.object(main, "OPENAI_COMPATIBLE_BASE_URLS", dict(main.OPENAI_COMPATIBLE_BASE_URLS)),
                patch.object(main, "GATEWAY_MODEL_CHOICES", dict(main.GATEWAY_MODEL_CHOICES)),
                patch.object(main, "GATEWAY_LIVE_PRICING_FETCHERS", dict(main.GATEWAY_LIVE_PRICING_FETCHERS)),
                patch.object(main, "_API_KEY_ENV", dict(main._API_KEY_ENV)),
            ):
                main._register_free_llm_api_providers()

                self.assertEqual(
                    main.CLOUD_PROVIDERS["testonlycloud"],
                    {"label": "Testonly Cloud (free)", "default_model": "testonly-model-1"},
                )
                self.assertEqual(main.OPENAI_COMPATIBLE_BASE_URLS["testonlycloud"], "https://api.testonly.example/v1")
                self.assertEqual(
                    main.GATEWAY_MODEL_CHOICES["testonlycloud"],
                    [{"model": "testonly-model-1", "label": "testonly-model-1"}, {"model": "testonly-model-2", "label": "testonly-model-2"}],
                )
                self.assertTrue(callable(main.GATEWAY_LIVE_PRICING_FETCHERS["testonlycloud"]))
                self.assertEqual(main._API_KEY_ENV["testonlycloud"], "TESTONLY_CLOUD_API_KEY")

    def test_register_free_llm_api_providers_skips_unsuitable_entries(self):
        fixture = {
            "providers": [
                # Already hand-curated — must not clobber the curated entry.
                {
                    "name": "Groq",
                    "category": "inference_provider",
                    "baseUrl": "https://should-not-win.example/v1",
                    "models": [{"id": "should-not-appear"}],
                },
                # Templated base URL — no single fixed endpoint to store.
                {
                    "name": "Templated Thing",
                    "category": "inference_provider",
                    "baseUrl": "https://api.example.com/{region}/v1",
                    "models": [{"id": "m1"}],
                },
                # No models listed.
                {
                    "name": "No Models Provider",
                    "category": "inference_provider",
                    "baseUrl": "https://api.nomodels.example/v1",
                    "models": [],
                },
                # Non-OpenAI-compatible category — out of scope for this loader.
                {
                    "name": "Cohere",
                    "category": "provider_api",
                    "baseUrl": "https://api.cohere.ai/v1",
                    "models": [{"id": "command-r"}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.json"
            data_path.write_text(json.dumps(fixture))
            original_groq = dict(main.CLOUD_PROVIDERS["groq"])
            with (
                patch.object(main, "_FREE_LLM_APIS_DATA", data_path),
                patch.object(main, "CLOUD_PROVIDERS", dict(main.CLOUD_PROVIDERS)),
                patch.object(main, "OPENAI_COMPATIBLE_BASE_URLS", dict(main.OPENAI_COMPATIBLE_BASE_URLS)),
                patch.object(main, "GATEWAY_MODEL_CHOICES", dict(main.GATEWAY_MODEL_CHOICES)),
                patch.object(main, "GATEWAY_LIVE_PRICING_FETCHERS", dict(main.GATEWAY_LIVE_PRICING_FETCHERS)),
                patch.object(main, "_API_KEY_ENV", dict(main._API_KEY_ENV)),
            ):
                main._register_free_llm_api_providers()

                self.assertEqual(main.CLOUD_PROVIDERS["groq"], original_groq)
                self.assertNotIn("templatedthing", main.CLOUD_PROVIDERS)
                self.assertNotIn("nomodelsprovider", main.CLOUD_PROVIDERS)
                self.assertNotIn("cohere", main.CLOUD_PROVIDERS)

    def test_register_free_llm_api_providers_is_noop_without_vendored_clone(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "data.json"
            with (
                patch.object(main, "_FREE_LLM_APIS_DATA", missing_path),
                patch.object(main, "CLOUD_PROVIDERS", dict(main.CLOUD_PROVIDERS)) as providers,
            ):
                before = dict(providers)
                main._register_free_llm_api_providers()
                self.assertEqual(providers, before)

    def test_register_free_llm_api_providers_is_noop_on_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data.json"
            data_path.write_text("{not valid json")
            with (
                patch.object(main, "_FREE_LLM_APIS_DATA", data_path),
                patch.object(main, "CLOUD_PROVIDERS", dict(main.CLOUD_PROVIDERS)) as providers,
            ):
                before = dict(providers)
                main._register_free_llm_api_providers()
                self.assertEqual(providers, before)


if __name__ == "__main__":
    unittest.main()
