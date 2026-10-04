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


# ── AirLLM integration (local OpenAI-compatible backend, like LM Studio /
# ── llama.cpp) ───────────────────────────────────────────────────────────────

def _sse_bytes(events):
    out = b"".join(f"data: {json.dumps(ev)}\n\n".encode() for ev in events)
    return out + b"data: [DONE]\n\n"


class _FakeSSEStream:
    def __init__(self, sse: bytes):
        self._sse = sse

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def aiter_bytes(self):
        async def gen():
            yield self._sse
        return gen()


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient in the local-server tests: records
    every stream()/post() call and returns canned SSE / JSON bodies from
    the per-test sse_by_url / json_by_url tables."""

    sse_by_url = {}
    json_by_url = {}
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None, **kwargs):
        type(self).calls.append((method, url, json))
        return _FakeSSEStream(self.sse_by_url[url])

    async def get(self, url, **kwargs):
        type(self).calls.append(("get", url, None))

        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return self.json_body

        resp = _Resp()
        resp.json_body = self.json_by_url[url]
        return resp

    async def post(self, url, json=None, **kwargs):
        type(self).calls.append(("post", url, json))

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return self.json_body

        resp = _Resp()
        resp.json_body = self.json_by_url[url]
        return resp


class AirllmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.sse = {}
        self.jsonb = {}
        # The fake client's response tables are class-level so the
        # main.app-created instances pick up this test's fixtures.
        _FakeAsyncClient.sse_by_url = self.sse
        _FakeAsyncClient.json_by_url = self.jsonb

    def _patch_client(self, fake_cls):
        return patch.object(main.httpx, "AsyncClient", fake_cls)

    def test_airllm_namespace_detection(self):
        self.assertTrue(main._is_airllm_model("airllm/meta-llama/Llama-3.1-8B-Instruct"))
        self.assertFalse(main._is_airllm_model("llama-cpp/model.gguf"))
        self.assertFalse(main._is_airllm_model("ollama-qwen"))

    def test_local_backend_resolution_covers_all_three_servers(self):
        self.assertEqual(
            main._local_openai_compatible_backend("lmstudio/some-model"),
            (main.LMSTUDIO, "some-model"))
        self.assertEqual(
            main._local_openai_compatible_backend("llama-cpp/m.gguf"),
            (main.LLAMA_CPP, "m.gguf"))
        self.assertEqual(
            main._local_openai_compatible_backend("airllm/Qwen/Qwen3-30B-A3B"),
            (main.AIRLLM, "Qwen/Qwen3-30B-A3B"))
        self.assertIsNone(main._local_openai_compatible_backend("qwen2.5-coder:32b"))

    def test_llm_complete_sends_airllm_to_airllm_server_with_stripped_id(self):
        url = f"{main.AIRLLM}/v1/chat/completions"
        self.jsonb[url] = {"choices": [{"message": {"content": "air answer"}}]}
        # _llm_complete verifies the sidecar has the model loaded first.
        self.jsonb[f"{main.AIRLLM}/health"] = {
            "ok": True, "model": "Qwen/Qwen3-30B-A3B", "loading": None, "error": None}
        with self._patch_client(_FakeAsyncClient):
            answer = asyncio.run(main._llm_complete("airllm/Qwen/Qwen3-30B-A3B",
                                                    [{"role": "user", "content": "hi"}]))
        self.assertEqual(answer, "air answer")

    def test_stream_openai_compatible_chat_yields_token_tool_and_usage_events(self):
        url = f"{main.AIRLLM}/v1/chat/completions"
        self.sse[url] = _sse_bytes([
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo!"}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"name": "read_file", "id": "call-1", "function": {"arguments": '{"path":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"name": "", "id": "call-1", "function": {"arguments": '"a.txt"}'}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        ])
        async def collect():
            return [
                ev async for ev in main._stream_openai_compatible_chat(
                    main.AIRLLM, "some-model", [{"role": "user", "content": "x"}])
            ]

        with self._patch_client(_FakeAsyncClient):
            events = asyncio.run(collect())
        self.assertEqual([e["type"] for e in events],
                         ["token", "token", "tool_use_start", "tool_use_delta", "tool_use_delta",
                          "usage", "tool_use_stop"])
        self.assertEqual(events[0]["content"] + events[1]["content"], "Hello!")
        self.assertEqual(events[5]["usage"]["prompt_eval_count"], 10)
        self.assertEqual(events[5]["usage"]["eval_count"], 3)

    def test_agent_turns_routes_airllm_models_through_openai_compatible_stream(self):
        url = f"{main.AIRLLM}/v1/chat/completions"
        self.sse[url] = _sse_bytes([
            {"choices": [{"delta": {"content": "Done: 2+2=4."}}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 6}},
        ])
        # The agent loop verifies the sidecar has the model loaded first.
        self.jsonb[f"{main.AIRLLM}/health"] = {
            "ok": True, "model": "some-model", "loading": None, "error": None}

        conv = [{"role": "user", "content": "what is 2+2?"}]

        async def drive():
            seen = []
            async for ev in main._agent_turns("airllm/some-model", conv, max_turns=3, tier="free",
                                              continuous=False):
                seen.append(ev)
            return seen

        with self._patch_client(_FakeAsyncClient):
            seen = asyncio.run(drive())

        types = [e["type"] for e in seen]
        self.assertIn("token", types)
        self.assertIn("done", types)
        done = next(e for e in seen if e["type"] == "done")
        self.assertIn("2+2=4", done["content"])
        self.assertEqual(done["usage"]["prompt_eval_count"], 20)
        self.assertEqual(done["usage"]["eval_count"], 6)
        # the assistant's reply was folded back into the conversation for the
        # next turn / persistence
        self.assertEqual(conv[-1], {"role": "assistant", "content": "Done: 2+2=4."})

    def test_agent_turns_empty_response_is_error_not_silent_done(self):
        # A zero-token stream used to end the turn with `done` carrying
        # empty content: the frontend rendered "nothing happened", and an
        # empty assistant row was persisted to the session. It must count
        # as a failed attempt instead — reported, retried within the same
        # 3-attempt budget as stream errors, with no empty reply folded in.
        url = f"{main.LLAMA_CPP}/v1/chat/completions"
        self.sse[url] = _sse_bytes([
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}},
        ])
        self.jsonb[f"{main.OLLAMA}/api/show"] = {"details": {"context_length": 16384}}

        conv = [{"role": "user", "content": "what is 2+2?"}]

        async def drive():
            seen = []
            async for ev in main._agent_turns("llama-cpp/model.gguf", conv, max_turns=3,
                                              tier="free", continuous=False):
                seen.append(ev)
            return seen

        with self._patch_client(_FakeAsyncClient):
            seen = asyncio.run(drive())

        types = [e["type"] for e in seen]
        self.assertNotIn("done", types)
        errors = [e for e in seen if e["type"] == "error"]
        self.assertEqual(len(errors), 3)
        self.assertIn("empty response", errors[0]["content"])
        self.assertEqual(conv, [{"role": "user", "content": "what is 2+2?"}])


class WebSearchFallbackTests(unittest.TestCase):
    def setUp(self):
        self.jsonb = {}
        _FakeAsyncClient.sse_by_url = {}
        _FakeAsyncClient.json_by_url = self.jsonb
        _FakeAsyncClient.calls = []

    def _patch_client(self):
        return patch.object(main.httpx, "AsyncClient", _FakeAsyncClient)

    def test_wikipedia_fallback_parses_full_text_search_shape(self):
        # The fallback must use `list=search` (full-text relevance): opensearch
        # is a title-prefix matcher that returned zero rows for
        # sentence-shaped queries, silently disabling the fallback exactly
        # when a factual question needed it (confirmed live).
        url = "https://en.wikipedia.org/w/api.php"
        self.jsonb[url] = {"query": {"search": [
            {"title": "Moons of Saturn",
             "snippet": "<span class=\"searchmatch\">Saturn</span> has 293 moons "
                        "with confirmed orbits&#039;"}]}}

        async def run():
            return await main._wikipedia_search(
                "current count of moons for each planet in the solar system", 5)

        with self._patch_client():
            results = asyncio.run(run())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Moons of Saturn")
        self.assertIn("293 moons", results[0]["snippet"])
        self.assertEqual(results[0]["url"], "https://en.wikipedia.org/wiki/Moons_of_Saturn")

    def test_wikipedia_fallback_returns_empty_on_legacy_opensearch_shape(self):
        # The old opensearch wire shape is a bare 4-list — the parser must
        # degrade to [] instead of crashing.
        url = "https://en.wikipedia.org/w/api.php"
        self.jsonb[url] = ["query", ["Some page"], ["desc"], ["https://x"]]

        async def run():
            return await main._wikipedia_search("anything", 5)

        with self._patch_client():
            self.assertEqual(asyncio.run(run()), [])


if __name__ == "__main__":
    unittest.main()
