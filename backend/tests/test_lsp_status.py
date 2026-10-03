"""LSP status semantics: a chip may say "available" only when every
executable in the server's chain exists in the container RIGHT NOW.

Regression guarded here: the typescript/bash servers used to run through
`npx -y <pkg>` and the status logic treated any command starting with
npx/node as "available" even when node itself was absent from the image —
the app reported Available for a server it could never start. Since the
image now bakes every LSP tool at pinned versions (see Dockerfile) and each
registry entry declares its full `requires` chain, status is computed from
real which() results, and /api/lsp/check proves startability with an actual
initialize→initialized→shutdown→exit handshake (fake processes, no forks).
"""
import asyncio
import json
import unittest
from unittest.mock import patch

import main


def _frame(obj):
    b = json.dumps(obj).encode()
    return b"Content-Length: %d\r\n\r\n" % len(b) + b


def _all_required():
    return {b for s in main.LSP_BUILTINS.values() for b in main._lsp_requires(s)}


class _FakeStdin:
    def __init__(self):
        self.frames = []

    def write(self, data):
        self.frames.append(bytes(data))

    async def drain(self):
        pass


class _FakeProc:
    """Stdout is the proc itself: readline()/read() serve from `stream` via a
    byte pointer (no BytesIO — it has no .find())."""

    def __init__(self, stream, rc=0, hang=False):
        self.stdin = _FakeStdin()
        self._data = stream
        self._pos = 0
        self._hang = hang
        self.returncode = None
        self.killed = False

    @property
    def stdout(self):
        return self

    async def readline(self):
        if self._hang:
            await asyncio.sleep(30)
        i = self._data.find(b"\n", self._pos)
        if i < 0:
            return b""
        chunk = self._data[self._pos:i + 1]
        self._pos = i + 1
        return chunk

    async def read(self, n):
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True


class LspStatusSemanticsTests(unittest.TestCase):
    def setUp(self):
        main._lsp_clients.clear()
        main._lsp_last_errors.clear()
        main._lsp_verified.clear()

    def _status(self, present):
        with patch("shutil.which", side_effect=lambda b: b in present):
            return asyncio.run(main.lsp_status())

    def test_everything_missing_reports_missing_with_reasons(self):
        d = self._status(set())
        self.assertTrue(d["enabled"])
        for s in d["servers"]:
            self.assertEqual(s["state"], "missing", s["name"])
            self.assertIn("not installed", s["reason"], s["name"])

    def test_everything_present_reports_available(self):
        d = self._status(_all_required())
        for s in d["servers"]:
            self.assertEqual(s["state"], "available", s["name"])
            self.assertIn("requires", s)

    def test_typescript_never_available_without_node(self):
        """The original bug: the npx-era logic said 'available' for
        typescript whenever its command STARTED with npx — regardless of
        whether node even existed. Now the whole chain must exist."""
        present = _all_required() - {"node"}
        d = self._status(present)
        ts = next(s for s in d["servers"] if s["name"] == "typescript")
        self.assertEqual(ts["state"], "missing")
        self.assertIn("node", ts["reason"])
        bash = next(s for s in d["servers"] if s["name"] == "bash")
        self.assertEqual(bash["state"], "missing")
        self.assertIn("node", bash["reason"])
        # the single-binary servers are unaffected by a missing node
        py = next(s for s in d["servers"] if s["name"] == "python")
        self.assertEqual(py["state"], "available")

    def test_single_binary_missing_marks_just_that_server(self):
        present = _all_required() - {"clangd"}
        d = self._status(present)
        states = {s["name"]: s["state"] for s in d["servers"]}
        self.assertEqual(states["cpp"], "missing")
        self.assertIn("clangd", next(
            s for s in d["servers"] if s["name"] == "cpp")["reason"])
        self.assertEqual(states["python"], "available")
        self.assertEqual(states["typescript"], "available")

    def test_present_but_failing_start_is_error_not_available(self):
        present = _all_required()

        async def boom(*_a, **_k):
            raise OSError("Exec format error")

        with patch("shutil.which", side_effect=lambda b: b in present), \
             patch("asyncio.create_subprocess_exec", side_effect=boom):
            client = asyncio.run(main._lsp_client_for(".ts"))
        self.assertIsNone(client)
        self.assertIn("OSError", main._lsp_last_errors["typescript"])
        d = self._status(present)
        ts = next(s for s in d["servers"] if s["name"] == "typescript")
        self.assertEqual(ts["state"], "error")
        self.assertIn("OSError", ts["reason"])
        self.assertIn("OSError", ts["error"])

    def test_verified_client_makes_reason_say_so(self):
        present = _all_required()
        main._lsp_verified["python"] = 123.0
        d = self._status(present)
        py = next(s for s in d["servers"] if s["name"] == "python")
        self.assertEqual(py["state"], "available")
        self.assertEqual(py["verified_at"], 123.0)
        self.assertIn("handshake", py["reason"])


class LspHandshakeTests(unittest.TestCase):
    """_lsp_handshake against fake processes: the probe must speak the real
    LSP framing and report exactly what happened (ok / why not / how long)."""

    def _run(self, stream, present=None, timeout=2.0, hang=False):
        if present is None:
            present = _all_required()
        spec = main.LSP_BUILTINS["python"]
        proc = _FakeProc(stream, hang=hang)

        async def spawn(*cmd, **_kw):
            return proc

        with patch("shutil.which", side_effect=lambda b: b in present):
            return asyncio.run(main._lsp_handshake(spec, timeout=timeout, spawn=spawn)), proc

    def test_full_lifecycle_passes(self):
        stream = _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + \
            _frame({"jsonrpc": "2.0", "id": 2, "result": None})
        (ok, detail, ms), proc = self._run(stream)
        self.assertTrue(ok, detail)
        self.assertGreaterEqual(ms, 0)
        self.assertEqual(proc.returncode, 0)
        sent = [json.loads(f.split(b"\r\n\r\n", 1)[1]) for f in proc.stdin.frames]
        methods = [m.get("method") for m in sent if m.get("method")]
        self.assertEqual(methods, ["initialize", "initialized", "shutdown", "exit"])
        self.assertEqual(sent[0]["id"], 1)
        self.assertEqual(sent[2]["id"], 2)

    def test_initialize_error_payload_fails(self):
        stream = _frame({"jsonrpc": "2.0", "id": 1,
                         "error": {"code": -32600, "message": "bad request"}})
        (ok, detail, _), _ = self._run(stream)
        self.assertFalse(ok)
        self.assertIn("initialize error", detail)

    def test_server_closing_stdout_fails(self):
        (ok, detail, _), _ = self._run(b"")
        self.assertFalse(ok)
        self.assertIn("closed the connection", detail)

    def test_silence_times_out(self):
        (ok, detail, _), _ = self._run(b"", timeout=0.3, hang=True)
        self.assertFalse(ok)
        self.assertIn("no LSP response", detail)


class LspCheckEndpointTests(unittest.TestCase):
    def setUp(self):
        main._lsp_clients.clear()
        main._lsp_last_errors.clear()

    def test_unknown_server_rejected(self):
        d = asyncio.run(main.lsp_check(main.LspCheckRequest(servers=["nope"])))
        self.assertFalse(d["ok"])
        self.assertIn("unknown server", d["servers"][0]["error"])

    def test_missing_binary_reported_without_spawn(self):
        with patch("shutil.which", return_value=None):
            d = asyncio.run(main.lsp_check(main.LspCheckRequest(servers=["python"])))
        self.assertFalse(d["ok"])
        self.assertIn("not installed", d["servers"][0]["error"])
        self.assertNotIn("ms", d["servers"][0])  # never even tried to start

    def test_handshake_success_flips_ok(self):
        stream = _frame({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}) + \
            _frame({"jsonrpc": "2.0", "id": 2, "result": None})

        async def spawn(*_cmd, **_kw):
            return _FakeProc(stream)

        with patch("shutil.which", side_effect=lambda b: b in _all_required()), \
             patch("asyncio.create_subprocess_exec", side_effect=spawn):
            d = asyncio.run(main.lsp_check(main.LspCheckRequest(servers=["python"])))
        self.assertTrue(d["ok"])
        self.assertTrue(d["servers"][0]["ok"])
        self.assertGreaterEqual(d["servers"][0]["ms"], 0)


if __name__ == "__main__":
    unittest.main()
