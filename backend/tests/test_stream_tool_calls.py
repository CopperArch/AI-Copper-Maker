"""Regression tests for _stream_openai_compatible_chat tool-call handling.

Native OpenAI-compatible tool calls (LM Studio / llama.cpp / AirLLM) must be
accumulated per tool-call index and surfaced as tool_use_start /
tool_use_delta / tool_use_stop events, or the agent loop silently drops real
tool calls (they never reach response_text / _extract_tool_call).
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import main


def _sse(events):
    out = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)
    return out + b"data: [DONE]\n\n"


class _FakeStream:
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


class _FakeClient:
    sse = b""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, json=None, **kwargs):
        return _FakeStream(type(self).sse)


def _run_stream(events):
    _FakeClient.sse = _sse(events)

    async def collect():
        return [
            e async for e in main._stream_openai_compatible_chat(
                "http://x", "m", [{"role": "user", "content": "hi"}]
            )
        ]

    with patch.object(main.httpx, "AsyncClient", _FakeClient):
        return asyncio.run(collect())


class StreamToolCallTests(unittest.TestCase):
    def test_plain_text_and_usage(self):
        ev = _run_stream([
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ])
        self.assertEqual([e["type"] for e in ev], ["token", "token", "usage"])
        self.assertEqual(ev[0]["content"] + ev[1]["content"], "Hello")
        self.assertEqual(ev[2]["usage"]["prompt_eval_count"], 5)
        self.assertEqual(ev[2]["usage"]["eval_count"], 2)

    def test_single_tool_call_with_function_name(self):
        ev = _run_stream([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": '{"path":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"a.txt"}'}}]}}]},
        ])
        self.assertEqual([e["type"] for e in ev],
                         ["tool_use_start", "tool_use_delta", "tool_use_delta", "tool_use_stop"])
        self.assertEqual(ev[0]["name"], "read_file")
        self.assertEqual(ev[0]["id"], "c1")
        self.assertEqual(ev[3]["name"], "read_file")
        self.assertEqual(ev[3]["input"], '{"path":"a.txt"}')

    def test_multiple_tool_call_indexes(self):
        ev = _run_stream([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "f1", "arguments": '{"x":1}'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "id": "b", "function": {"name": "f2", "arguments": '{"y":2}'}}]}}]},
        ])
        starts = [e for e in ev if e["type"] == "tool_use_start"]
        stops = [e for e in ev if e["type"] == "tool_use_stop"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(stops), 2)
        self.assertEqual(sorted(s["name"] for s in starts), ["f1", "f2"])
        self.assertEqual(stops[0]["input"], '{"x":1}')
        self.assertEqual(stops[1]["input"], '{"y":2}')

    def test_tool_call_level_name(self):
        ev = _run_stream([
            {"choices": [{"delta": {"tool_calls": [
                {"name": "greet", "id": "g1", "function": {"arguments": '{"who":"a"}'}}]}}]},
        ])
        start = next(e for e in ev if e["type"] == "tool_use_start")
        stop = next(e for e in ev if e["type"] == "tool_use_stop")
        self.assertEqual(start["name"], "greet")
        self.assertEqual(stop["name"], "greet")
        self.assertEqual(stop["input"], '{"who":"a"}')

    def test_text_tool_usage_event_ordering(self):
        ev = _run_stream([
            {"choices": [{"delta": {"content": "Let me check."}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "z", "function": {"name": "ls", "arguments": "{}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
        ])
        self.assertEqual([e["type"] for e in ev],
                         ["token", "tool_use_start", "tool_use_delta", "usage", "tool_use_stop"])

    def test_malformed_chunk_is_skipped(self):
        raw = b'data: {broken json\n\ndata: ' + json.dumps(
            {"choices": [{"delta": {"content": "ok"}}]}).encode() + b"\n\ndata: [DONE]\n\n"
        _FakeClient.sse = raw
        fake = _FakeClient

        async def collect():
            return [
                e async for e in main._stream_openai_compatible_chat("http://x", "m", [])
            ]

        with patch.object(main.httpx, "AsyncClient", fake):
            ev = asyncio.run(collect())
        self.assertEqual([e["type"] for e in ev], ["token"])

    def test_incomplete_arguments_still_emit_stop(self):
        ev = _run_stream([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "p", "function": {"name": "partial", "arguments": '{"unfin'}}]}}]},
        ])
        stop = next(e for e in ev if e["type"] == "tool_use_stop")
        self.assertEqual(stop["name"], "partial")
        self.assertEqual(stop["input"], '{"unfin')

    def test_no_tool_calls_emits_no_stop(self):
        ev = _run_stream([
            {"choices": [{"delta": {"content": "x"}}]},
            {"choices": [{"delta": {}}]},
        ])
        self.assertNotIn("tool_use_stop", [e["type"] for e in ev])

    def test_error_chunk_raises_instead_of_silent_empty(self):
        # LM Studio's llama.cpp engine answers HTTP 200 but carries the
        # engine error in the terminal chunk when the prompt exceeds the
        # server's context size — the stream must raise, not yield nothing
        # (which the agent loop would turn into a silently empty reply).
        _FakeClient.sse = _sse([
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"error": {"code": 400,
                       "message": "request (8733 tokens) exceeds the available context size (8192 tokens)"}},
        ])

        async def collect():
            return [e async for e in main._stream_openai_compatible_chat("http://x", "m", [])]

        with patch.object(main.httpx, "AsyncClient", _FakeClient):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(collect())
        self.assertIn("exceeds the available context size", str(ctx.exception))


class StreamChatNdjsonLocalTests(unittest.TestCase):
    """_stream_chat_ndjson must stream token-by-token for local
    OpenAI-compatible backends (llama-cpp/…) instead of waiting for the
    whole response — the old non-streaming branch froze the UI for the
    entire prefill+generation on long system prompts."""

    def _collect(self, events):
        _FakeClient.sse = _sse(events)

        async def run():
            return [line async for line in
                    main._stream_chat_ndjson("llama-cpp/model",
                                             [{"role": "user", "content": "hi"}],
                                             timeout=60)]

        with patch.object(main.httpx, "AsyncClient", _FakeClient):
            return [json.loads(l) for l in asyncio.run(run())]

    def test_streams_token_chunks_then_empty_done(self):
        lines = self._collect([
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        ])
        # Per-token lines first, terminal done line with EMPTY content (the
        # frontend accumulates message.content, so the done line must not
        # repeat the whole reply — matches Ollama's native stream shape).
        self.assertEqual([l.get("done") for l in lines], [None, None, True])
        self.assertEqual([l["message"]["content"] for l in lines[:2]], ["He", "llo"])
        self.assertEqual(lines[2]["message"]["content"], "")

    def test_tool_call_surfaces_as_fence_not_executed(self):
        lines = self._collect([
            {"choices": [{"delta": {"content": "Let me check."}}]},
            {"choices": [
                {"delta": {
                    "tool_calls": [
                        {"index": 0, "id": "c1",
                         "function": {"name": "run_command",
                                      "arguments": "{\"cmd\": \"ls\"}"}}
                    ]
                }}
            ]},
        ])
        contents = [l["message"]["content"] for l in lines]
        self.assertEqual(len(lines), 3)  # text, fence, done
        self.assertEqual(contents[0], "Let me check.")
        self.assertIn("```tool", contents[1])
        self.assertIn("run_command", contents[1])
        self.assertEqual(lines[2].get("done"), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
