"""Tests for the AirLLM model discovery/download layer.

Covers, with all HTTP faked (no real 8082 port, no network):
  * main.airllm_catalog() — curated catalog merged with the sidecar's
    installed/loaded state, server-down tolerance, custom-model extras,
    live Hugging Face size vs fallback estimate;
  * main.download_airllm_model() — disk pre-check (reject + proxy),
    unknown-size skip, sidecar error passthrough, sidecar-down 502,
    already-installed no-op;
  * airllm_server._installed_models() — the HF cache scan (the
    `splitted_model` marker) against a fake cache tree;
  * airllm_server.health() — cache_dir + disk_free reporting;
  * routing guards — "airllm/<id>" refs hit the AirLLM server with the
    prefix stripped, and a namespaced plain Ollama model stays on Ollama.

Nothing here touches the real config.json, credentials, or model files.
"""
import asyncio
import atexit
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main

GB = 1024 ** 3

# Hermeticity: importing airllm_server mkdirs CACHE_DIR at import time — as
# root inside the container that would recreate a root-owned dir under the
# real (user-owned) config mount. Point the cache at a temp dir BEFORE any
# airllm_server import.
_AIRLLM_TEST_CACHE = Path(tempfile.mkdtemp(prefix="airllm-test-cache-"))
os.environ["AIRLLM_CACHE_DIR"] = str(_AIRLLM_TEST_CACHE)


def _cleanup_test_cache():
    shutil.rmtree(_AIRLLM_TEST_CACHE, ignore_errors=True)


atexit.register(_cleanup_test_cache)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}: {self.text or self._payload}")


class _FakeClient:
    """Stands in for httpx.AsyncClient in the airllm endpoints: routes
    GET/POST to canned responses via `router`, records every call."""

    def __init__(self, router):
        self._router = router
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        self.calls.append(("GET", url))
        return self._router("GET", url)

    async def post(self, url, **kw):
        self.calls.append(("POST", url))
        return self._router("POST", url)


def _sidecar_factory(ok=True, installed=(), loaded_model=None, disk_free=None,
                     download=(200, None)):
    """Factory for patch(main.httpx.AsyncClient) that answers the AirLLM
    sidecar (health / installed / download) and fails everything else —
    including the Hugging Face size lookups, which forces the fallback
    estimate path."""
    clients = []

    def factory(**_kw):
        def router(method, url):
            if url.startswith(main.AIRLLM):
                if url.endswith("/health"):
                    if not ok:
                        raise ConnectionError("sidecar down (test)")
                    return _Resp(200, {"ok": True, "device": "cpu",
                                       "model": loaded_model, "loading": None,
                                       "error": None, "disk_free": disk_free,
                                       "cache_dir": "/cache"})
                if url.endswith("/v1/airllm/installed"):
                    return _Resp(200, {"installed": list(installed)})
                if url.endswith("/v1/airllm/download"):
                    code, payload = download
                    return _Resp(code, payload or {"downloading": True})
            raise ConnectionError("no fake route for " + url)

        c = _FakeClient(router)
        clients.append(c)
        return c

    factory.clients = clients
    return factory


class AirLLMCatalogTests(unittest.TestCase):
    def setUp(self):
        main._AIRLLM_SIZE_CACHE.clear()

    def test_catalog_merges_installed_and_loaded(self):
        qwen = "Qwen/Qwen3-30B-A3B"
        factory = _sidecar_factory(installed=(qwen,), loaded_model=qwen,
                                   disk_free=200 * GB)
        with patch("main.httpx.AsyncClient", factory):
            data = asyncio.run(main.airllm_catalog())
        self.assertTrue(data["server"]["ok"])
        self.assertEqual(data["server"]["disk_free_gb"], 200.0)
        models = {m["id"]: m for m in data["models"]}
        self.assertEqual(set(models),
                         {e["id"] for e in main.AIRLLM_RECOMMENDED_MODELS})
        self.assertTrue(models[qwen]["installed"])
        self.assertTrue(models[qwen]["loaded"])
        llama = "meta-llama/Llama-3.1-8B-Instruct"
        self.assertFalse(models[llama]["installed"])
        self.assertFalse(models[llama]["loaded"])
        # Hugging Face unreachable in the fake → catalog fallback estimate
        self.assertEqual(models[llama]["size_gb"], 17.0)

    def test_catalog_survives_sidecar_down(self):
        with patch("main.httpx.AsyncClient", _sidecar_factory(ok=False)):
            data = asyncio.run(main.airllm_catalog())
        self.assertFalse(data["server"]["ok"])
        self.assertIsNone(data["server"]["disk_free_gb"])
        self.assertEqual(len(data["models"]), len(main.AIRLLM_RECOMMENDED_MODELS))
        self.assertTrue(all(not m["installed"] for m in data["models"]))
        self.assertTrue(all(not m["loaded"] for m in data["models"]))

    def test_catalog_lists_custom_installed_models(self):
        with patch("main.httpx.AsyncClient",
                   _sidecar_factory(installed=("myorg/my-custom-1B",))):
            data = asyncio.run(main.airllm_catalog())
        models = {m["id"]: m for m in data["models"]}
        self.assertEqual(len(data["models"]), len(main.AIRLLM_RECOMMENDED_MODELS) + 1)
        self.assertTrue(models["myorg/my-custom-1B"]["installed"])
        self.assertEqual(models["myorg/my-custom-1B"]["name"], "myorg/my-custom-1B")

    def test_catalog_live_size_wins_over_estimate(self):
        llama = "meta-llama/Llama-3.1-8B-Instruct"
        live = int(16.5 * GB)

        def factory(**_kw):
            def router(method, url):
                if url.startswith(main.AIRLLM):
                    if url.endswith("/health"):
                        return _Resp(200, {"ok": True, "model": None,
                                           "loading": None, "error": None,
                                           "disk_free": 200 * GB,
                                           "cache_dir": "/c"})
                    if url.endswith("/v1/airllm/installed"):
                        return _Resp(200, {"installed": []})
                if url.startswith("https://huggingface.co/api/models/"):
                    return _Resp(200, {"usedStorage": live})
                raise ConnectionError("no fake route for " + url)
            return _FakeClient(router)

        with patch("main.httpx.AsyncClient", factory):
            data = asyncio.run(main.airllm_catalog())
        entry = next(m for m in data["models"] if m["id"] == llama)
        self.assertEqual(entry["size_gb"], 16.5)

    def test_size_cache_reuses_live_lookup(self):
        llama = "meta-llama/Llama-3.1-8B-Instruct"
        lookups = []

        def factory(**_kw):
            def router(method, url):
                if url.startswith("https://huggingface.co/api/models/"):
                    lookups.append(url)
                    return _Resp(200, {"usedStorage": int(16.5 * GB)})
                raise ConnectionError("no fake route for " + url)
            return _FakeClient(router)

        with patch("main.httpx.AsyncClient", factory):
            first = asyncio.run(main._airllm_repo_size_bytes(llama))
            second = asyncio.run(main._airllm_repo_size_bytes(llama))
        self.assertEqual(first, second)
        self.assertEqual(len(lookups), 1)


class AirLLMDownloadTests(unittest.TestCase):
    def setUp(self):
        main._AIRLLM_SIZE_CACHE.clear()

    def _run(self, model, **kw):
        factory = _sidecar_factory(**kw)
        with patch("main.httpx.AsyncClient", factory):
            result = asyncio.run(main.download_airllm_model(
                main.AirLLMDownloadRequest(model=model)))
        return result, factory

    def test_rejects_when_disk_full_raises_400(self):
        factory = _sidecar_factory(disk_free=5 * GB)
        with patch("main.httpx.AsyncClient", factory):
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main.download_airllm_model(
                    main.AirLLMDownloadRequest(
                        model="meta-llama/Llama-3.1-8B-Instruct")))
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("Not enough disk space", str(cm.exception.detail))
        self.assertFalse(any(c[0] == "POST" for client in factory.clients
                             for c in client.calls))

    def test_proxies_when_space_ok(self):
        result, factory = self._run("meta-llama/Llama-3.1-8B-Instruct",
                                    disk_free=100 * GB)
        self.assertTrue(result.get("downloading"))
        self.assertTrue(any(
            c == ("POST", main.AIRLLM + "/v1/airllm/download")
            for client in factory.clients for c in client.calls))

    def test_unknown_model_skips_precheck(self):
        # Not in the catalog, Hugging Face unreachable in the fake → size
        # unknown → no disk pre-check, the sidecar's own check_space guards
        # the split step instead.
        result, _ = self._run("myorg/my-custom-1B", disk_free=1 * GB)
        self.assertTrue(result.get("downloading"))

    def test_surfaces_sidecar_error(self):
        detail = "Repository 'nope/missing' not found"
        result_factory = _sidecar_factory(
            disk_free=100 * GB, download=(404, {"detail": detail}))
        with patch("main.httpx.AsyncClient", result_factory):
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main.download_airllm_model(
                    main.AirLLMDownloadRequest(model="nope/missing")))
        self.assertEqual(cm.exception.status_code, 404)
        self.assertEqual(str(cm.exception.detail), detail)

    def test_sidecar_down_raises_502(self):
        with patch("main.httpx.AsyncClient", _sidecar_factory(ok=False)):
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main.download_airllm_model(
                    main.AirLLMDownloadRequest(
                        model="meta-llama/Llama-3.1-8B-Instruct")))
        self.assertEqual(cm.exception.status_code, 502)
        self.assertIn("setup/airllm-linux.sh", str(cm.exception.detail))

    def test_already_installed_is_passthrough_noop(self):
        payload = {"model": "Qwen/Qwen3-30B-A3B", "already_installed": True}
        result, factory = self._run("Qwen/Qwen3-30B-A3B",
                                    download=(200, payload))
        self.assertEqual(result, payload)
        self.assertTrue(any(
            c == ("POST", main.AIRLLM + "/v1/airllm/download")
            for client in factory.clients for c in client.calls))


class AirLLMSidecarCacheTests(unittest.TestCase):
    """The sidecar's cache helpers, against a fake HF cache tree (tmp dir
    only — no real model files, no real cache)."""

    def test_installed_scan_finds_split_shards(self):
        import airllm_server
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            good = hub / "models--myorg--good-model" / "snapshots" / "abc123"
            (good / "splitted_model").mkdir(parents=True)
            (good / "model.safetensors").write_bytes(b"x")
            partial = hub / "models--myorg--partial" / "snapshots" / "def456"
            partial.mkdir(parents=True)
            (partial / "model.safetensors").write_bytes(b"x")
            (hub / "not-a-model").mkdir()
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                self.assertEqual(airllm_server._installed_models(),
                                 ["myorg/good-model"])

    def test_installed_scan_empty_without_cache(self):
        import airllm_server
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,
                            {"HF_HUB_CACHE": str(Path(tmp) / "hub")}):
                self.assertEqual(airllm_server._installed_models(), [])

    def test_hub_cache_dir_defaults_under_hf_home(self):
        import airllm_server
        env = dict(os.environ)
        os.environ.pop("HF_HUB_CACHE", None)
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
        try:
            self.assertEqual(airllm_server._hub_cache_dir(),
                             Path(os.environ["HF_HOME"]) / "hub")
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_health_reports_cache_dir_and_disk(self):
        import airllm_server
        health = asyncio.run(airllm_server.health())
        self.assertTrue(health["ok"])
        self.assertEqual(health["cache_dir"], str(airllm_server.CACHE_DIR))
        self.assertIsInstance(health["disk_free"], int)
        self.assertGreater(health["disk_free"], 0)

    def test_installed_scan_skips_in_flight_split(self):
        """A repo mid-download/split already has a partial splitted_model/
        dir — it must NOT count as installed until the worker is done."""
        import airllm_server
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            inflight = hub / "models--myorg--in-flight" / "snapshots" / "abc"
            (inflight / "splitted_model").mkdir(parents=True)
            done = hub / "models--myorg--done" / "snapshots" / "def"
            (done / "splitted_model").mkdir(parents=True)
            with patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
                airllm_server.STATE.loading = "splitting myorg/in-flight into per-layer shards"
                try:
                    self.assertEqual(airllm_server._installed_models(), ["myorg/done"])
                finally:
                    airllm_server.STATE.loading = None
                self.assertEqual(airllm_server._installed_models(),
                                 ["myorg/done", "myorg/in-flight"])

    def test_run_generation_decodes_sequence_dim_not_batch_dim(self):
        """Regression (found live): the non-streaming decode sliced
        seq[input_len:] — the BATCH dim (always 1 row, so an empty slice and
        an empty reply) instead of seq[0, input_len:] (the sequence dim).
        Faked tensors keep this test runnable without torch."""
        import airllm_server

        class _Ids:
            shape = (1, 4)

        class _Tok:
            pad_token_id = 5
            eos_token_id = 6
            received = None

            def __len__(self):
                return 100

            def decode(self, ids, skip_special_tokens=True):
                type(self).received = list(ids)
                return "text"

        class _Seq:
            def __init__(self, rows):
                self._rows = rows
                self.shape = (len(rows), len(rows[0]))

            def __getitem__(self, key):
                if isinstance(key, tuple):
                    row, s = key  # the correct (0, slice(...)) form
                    return list(self._rows[row][s])
                if isinstance(key, slice):  # batch-dim slice (the bug)
                    return [list(r) for r in self._rows[key]]
                return list(self._rows[key])

        model = type("_M", (), {"tokenizer": _Tok(), "device": "cpu"})()
        model.generate = lambda **kw: _Seq([[1, 2, 3, 4, 7, 8, 6]])

        with patch.object(airllm_server, "_apply_chat_template",
                          return_value=(_Ids(), object())):
            pt, ct, text = airllm_server._run_generation(model, [], 0.0, 16, 1.0)
        self.assertEqual(text, "text")
        self.assertEqual((pt, ct), (4, 3))
        self.assertEqual(_Tok.received, [7, 8, 6])


class _Dev:
    """Mimics torch.device: an object exposing .type."""

    def __init__(self, t):
        self.type = t

    def __eq__(self, other):
        return isinstance(other, _Dev) and self.type == other.type

    def __repr__(self):
        return f"_Dev({self.type})"


class _FakeStorage:
    """The storage a materialised fused parameter wraps; records every
    slice assignment."""

    def __init__(self, shape, device_type):
        self.shape = tuple(shape)
        self.device = _Dev(device_type)
        self.writes = []


class _FakeValue:
    def __init__(self, dtype="bf16"):
        self.dtype = dtype
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


class _FusedParam:
    """Fused expert parameter. In its starting (meta) state it carries
    shape/dtype like the real meta placeholder; when the shim materialises
    it — `type(param)(storage, requires_grad=...)`, exactly what the real
    code does — it wraps the storage instead. (torch 2.14 forbids filling a
    meta parameter via `param.data = ...`, so the shim must replace the
    object in the module's _parameters dict; the fake mirrors that.)"""

    def __init__(self, first, dtype="bf16", requires_grad=False):
        self.requires_grad = requires_grad
        if isinstance(first, _FakeStorage):
            self.storage = first
            self.device = first.device
            self.shape = first.shape
        else:
            self.shape = tuple(first)
            self.dtype = dtype
            self.device = _Dev("meta")
            self.storage = None

    def __setitem__(self, key, value):
        self.storage.writes.append((key, value))


class _FakeTorch:
    def __init__(self):
        self.empty_calls = []

    def device(self, spec):
        return _Dev(spec)

    def empty(self, shape, dtype=None, device=None):
        self.empty_calls.append((tuple(shape), dtype, device))
        return _FakeStorage(shape, device.type)


class _FusedExperts:
    def __init__(self, n_exp=4, inter=3, hidden=2):
        self._parameters = {
            "gate_up_proj": _FusedParam((n_exp, 2 * inter, hidden)),
            "down_proj": _FusedParam((n_exp, hidden, inter)),
        }


class _FusedModel:
    """Stands in for the streamed model: only the `experts` submodule the
    shim resolves matters."""

    def __init__(self):
        self.experts = _FusedExperts()

    def get_submodule(self, name):
        return getattr(self, name.rsplit(".", 1)[-1])


class FusedMoeExpertShimTests(unittest.TestCase):
    """transformers >=5 fused-expert compat in airllm_server._make_fused_expert_shim:
    legacy per-expert checkpoint keys (`...experts.<e>.gate_proj.weight`) must
    land in the fused gate_up_proj / down_proj tensors (expert e at dim 0,
    gate/up split across dim 1), not die on the missing `experts.<e>`
    attribute (verified live on Qwen3-30B-A3B: every MoE layer forward
    raised AttributeError without it)."""

    def _shim(self, torch):
        import airllm_server

        def orig_boom(model, tensor_name, device, **kw):
            raise AttributeError(f"fused module has no attribute "
                                 f"'{tensor_name.rsplit('.', 2)[-2]}'")

        return airllm_server._make_fused_expert_shim(orig_boom, torch)

    def test_gate_slice_written_into_fused_param(self):
        torch = _FakeTorch()
        model = _FusedModel()
        value = _FakeValue()
        self._shim(torch)(model, "mlp.experts.1.gate_proj.weight", "cpu", value=value)
        big = model.experts._parameters["gate_up_proj"]
        self.assertEqual(big.device.type, "cpu",
                         "materialisation must happen on first write")
        self.assertEqual(len(torch.empty_calls), 1)
        # no dtype kwarg -> materialise with the meta param's own dtype
        self.assertEqual(torch.empty_calls[0][1], "bf16")
        (key, val), = big.storage.writes
        # expert 1 of 4, inter=3 -> gate = rows [0, 3) of expert 1's row
        self.assertEqual(key, (1, slice(0, 3, None), slice(None, None, None)))
        self.assertIs(val, value)
        ((dev,), _kw), = value.to_calls
        self.assertEqual(dev.type, "cpu")

    def test_up_slice_written_at_second_half(self):
        torch = _FakeTorch()
        model = _FusedModel()
        value = _FakeValue()
        self._shim(torch)(model, "mlp.experts.1.up_proj.weight", "cpu", value=value)
        (key, _val), = model.experts._parameters["gate_up_proj"].storage.writes
        self.assertEqual(key, (1, slice(3, 6, None), slice(None, None, None)))

    def test_down_slice_written_by_expert_index(self):
        torch = _FakeTorch()
        model = _FusedModel()
        value = _FakeValue()
        self._shim(torch)(model, "mlp.experts.3.down_proj.weight", "cpu", value=value)
        big = model.experts._parameters["down_proj"]
        self.assertEqual(len(torch.empty_calls), 1)
        (key, _val), = big.storage.writes
        self.assertEqual(key, 3)

    def test_repeated_writes_reuse_one_materialisation(self):
        torch = _FakeTorch()
        model = _FusedModel()
        shim = self._shim(torch)
        shim(model, "mlp.experts.0.gate_proj.weight", "cpu", value=_FakeValue())
        shim(model, "mlp.experts.0.up_proj.weight", "cpu", value=_FakeValue())
        shim(model, "mlp.experts.0.down_proj.weight", "cpu", value=_FakeValue())
        self.assertEqual(len(torch.empty_calls), 2)  # one per fused param
        # gate and up both land in the same materialised gate_up_proj
        self.assertEqual(len(model.experts._parameters["gate_up_proj"].storage.writes), 2)

    def test_dtype_kwarg_honored(self):
        torch = _FakeTorch()
        model = _FusedModel()
        value = _FakeValue()
        self._shim(torch)(model, "mlp.experts.0.gate_proj.weight", "cpu",
                          value=value, dtype="fp32")
        # value is cast to the requested dtype, then moved to the device
        self.assertEqual(value.to_calls[0][0], ("fp32",))
        self.assertEqual(torch.empty_calls[0][1], "fp32")

    def test_unmatched_name_reraises(self):
        torch = _FakeTorch()
        model = _FusedModel()
        with self.assertRaises(AttributeError):
            self._shim(torch)(model, "mlp.experts.0.gate_proj.bias", "cpu",
                              value=_FakeValue())


class _HealthSeqRouter:
    """Stateful fake for _ensure_airllm_model_loaded: /health pops canned
    responses (a None entry means 'sidecar down'), /v1/airllm/load is
    counted and answered with a fixed status."""

    def __init__(self, health_seq, load_status=200):
        self.health_seq = list(health_seq)
        self.load_status = load_status
        self.posts = 0

    def __call__(self, method, url):
        if url.startswith(main.AIRLLM):
            if url.endswith("/health"):
                # Pop through the sequence; the last entry repeats once the
                # sequence is exhausted (a still-busy sidecar).
                h = self.health_seq.pop(0) if len(self.health_seq) > 1 else self.health_seq[0]
                if h is None:
                    raise ConnectionError("sidecar down (test)")
                return _Resp(200, h)
            if url.endswith("/v1/airllm/load"):
                self.posts += 1
                return _Resp(self.load_status,
                             {"detail": "boom"} if self.load_status >= 400
                             else {"loading": True})
        raise ConnectionError("no fake route for " + url)


def _h(model=None, loading=None, error=None):
    return {"ok": True, "model": model, "loading": loading, "error": error}


def _fast_sleep():
    """Collapse the helper's real 3s/5s poll sleeps to no-ops so these
    tests run in milliseconds."""
    async def _fast(*_args, **_kwargs):
        return None
    return patch("asyncio.sleep", _fast)


class AirLLMAutoLoadTests(unittest.TestCase):
    """_ensure_airllm_model_loaded — what makes a downloaded-but-not-loaded
    dropdown selection usable instead of 409-ing on the first send."""

    def _run(self, health_seq, load_status=200, **patch_kwargs):
        router = _HealthSeqRouter(health_seq, load_status)
        with patch("main.httpx.AsyncClient",
                   lambda **_kw: _FakeClient(router)), _fast_sleep(), \
                patch("main._AIRLLM_LOAD_WAIT_SECONDS", patch_kwargs.pop("wait", 900)):
            result = asyncio.run(main._ensure_airllm_model_loaded("Qwen/Qwen3-30B-A3B"))
        return router, result

    def test_already_loaded_no_load_call(self):
        router, _ = self._run([_h(model="Qwen/Qwen3-30B-A3B")])
        self.assertEqual(router.posts, 0)

    def test_not_loaded_triggers_load_then_waits(self):
        router, _ = self._run([
            _h(),
            _h(loading="Qwen/Qwen3-30B-A3B (download + per-layer split on first use)"),
            _h(model="Qwen/Qwen3-30B-A3B"),
        ])
        self.assertEqual(router.posts, 1)

    def test_existing_error_surfaces(self):
        router = _HealthSeqRouter([_h(error="OSError: disk full")])
        with patch("main.httpx.AsyncClient",
                   lambda **_kw: _FakeClient(router)), _fast_sleep():
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main._ensure_airllm_model_loaded("Qwen/Qwen3-30B-A3B"))
        self.assertEqual(cm.exception.status_code, 500)
        self.assertIn("disk full", str(cm.exception.detail))
        self.assertEqual(router.posts, 0)

    def test_sidecar_down_raises_502_detail(self):
        router = _HealthSeqRouter([None])
        with patch("main.httpx.AsyncClient",
                   lambda **_kw: _FakeClient(router)), _fast_sleep():
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main._ensure_airllm_model_loaded("Qwen/Qwen3-30B-A3B"))
        self.assertEqual(cm.exception.status_code, 502)
        self.assertIn("setup/airllm-linux.sh", str(cm.exception.detail))

    def test_forever_busy_times_out(self):
        router = _HealthSeqRouter([_h(loading="splitting Qwen/Qwen3-30B-A3B into per-layer shards")])
        with patch("main.httpx.AsyncClient",
                   lambda **_kw: _FakeClient(router)), _fast_sleep(), \
                patch("main._AIRLLM_LOAD_WAIT_SECONDS", 0):
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main._ensure_airllm_model_loaded("Qwen/Qwen3-30B-A3B"))
        self.assertEqual(cm.exception.status_code, 504)
        self.assertEqual(router.posts, 0)

    def test_load_endpoint_error_propagates(self):
        router = _HealthSeqRouter([_h()], load_status=503)
        with patch("main.httpx.AsyncClient",
                   lambda **_kw: _FakeClient(router)), _fast_sleep():
            with self.assertRaises(main.HTTPException) as cm:
                asyncio.run(main._ensure_airllm_model_loaded("Qwen/Qwen3-30B-A3B"))
        self.assertEqual(cm.exception.status_code, 503)
        self.assertEqual(router.posts, 1)


class AirLLMRoutingGuards(unittest.TestCase):
    """The hot-path dispatch in _llm_complete: the new AirLLM refs must not
    be able to hijack Ollama models, and plain (even namespaced) Ollama
    model names must not leak to the AirLLM server."""

    def test_airllm_ref_routes_to_airllm_server_stripped(self):
        calls = []

        def factory(**_kw):
            def router(method, url):
                calls.append((method, url))
                if url == main.AIRLLM + "/health":
                    return _Resp(200, {"ok": True,
                                       "model": "Qwen/Qwen3-30B-A3B",
                                       "loading": None, "error": None})
                if url == main.AIRLLM + "/v1/chat/completions":
                    return _Resp(200, {"choices": [{"message": {"content": "pong"}}]})
                raise ConnectionError("unexpected " + url)
            return _FakeClient(router)

        with patch("main.httpx.AsyncClient", factory):
            out = asyncio.run(main._llm_complete(
                "airllm/Qwen/Qwen3-30B-A3B",
                [{"role": "user", "content": "ping"}]))
        self.assertEqual(out, "pong")
        self.assertEqual(calls, [
            ("GET", main.AIRLLM + "/health"),
            ("POST", main.AIRLLM + "/v1/chat/completions"),
        ])

    def test_llm_complete_ensures_airllm_model_loaded(self):
        """Selecting a downloaded-but-not-loaded AirLLM model must trigger
        the load in _llm_complete instead of 409-ing on the chat call."""
        router = _HealthSeqRouter([
            _h(),
            _h(model="Qwen/Qwen3-30B-A3B"),
        ])
        chat_posted = []

        def factory(**_kw):
            def route(method, url):
                if url == main.AIRLLM + "/v1/chat/completions":
                    chat_posted.append(url)
                    return _Resp(200, {"choices": [{"message": {"content": "pong"}}]})
                return router(method, url)
            return _FakeClient(route)

        with patch("main.httpx.AsyncClient", factory), _fast_sleep():
            out = asyncio.run(main._llm_complete(
                "airllm/Qwen/Qwen3-30B-A3B",
                [{"role": "user", "content": "ping"}]))
        self.assertEqual(out, "pong")
        self.assertEqual(router.posts, 1)
        self.assertEqual(chat_posted, [main.AIRLLM + "/v1/chat/completions"])

    def test_plain_namespaced_ollama_model_stays_on_ollama(self):
        calls = []

        def factory(**_kw):
            def router(method, url):
                calls.append((method, url))
                if url == main.OLLAMA + "/api/chat":
                    return _Resp(200, {"message": {"content": "ollama-pong"}})
                raise ConnectionError("unexpected " + url)
            return _FakeClient(router)

        with patch("main.httpx.AsyncClient", factory):
            out = asyncio.run(main._llm_complete(
                "ttempvnn/0bserverx-qwen3.8:latest",
                [{"role": "user", "content": "ping"}]))
        self.assertEqual(out, "ollama-pong")
        self.assertEqual(calls, [("POST", main.OLLAMA + "/api/chat")])


class TestPumpStreamer(unittest.TestCase):
    """Regression test for the TextIteratorStreamer idle-timeout fix.

    Verifies that _pump_streamer correctly handles the streamer's idle
    watchdog (queue.Empty): it must NOT treat an idle timeout as fatal
    while the generation thread is still running (gen_done not set), but
    it IS terminal once generation has finished.
    """

    def setUp(self):
        self._test_cache = tempfile.mkdtemp(prefix="pump-test-cache-")
        os.environ["AIRLLM_CACHE_DIR"] = self._test_cache
        import airllm_server  # noqa: E402 — needed for _pump_streamer

    def tearDown(self):
        shutil.rmtree(self._test_cache, ignore_errors=True)
        if "AIRLLM_CACHE_DIR" in os.environ:
            del os.environ["AIRLLM_CACHE_DIR"]

    def test_pump_streamer_continues_on_idle_timeout_while_generation_running(self):
        """Empty from the streamer's idle watchdog must NOT break the pump
        while the generation thread is still alive. The pump keeps waiting
        for the next token, which eventually arrives."""
        gen_done = threading.Event()

        class _FakeStreamer:
            def __init__(self):
                self._first = True
                self._tokens = ["a", "b"]

            def __iter__(self):
                return self

            def __next__(self):
                if self._first:
                    self._first = False
                    raise Empty()
                if self._tokens:
                    return self._tokens.pop(0)
                gen_done.set()
                raise StopIteration

        streamer = _FakeStreamer()
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        t = threading.Thread(
            target=airllm_server._pump_streamer,
            args=(streamer, loop, queue, gen_done),
            daemon=True)
        t.start()

        items = []
        while True:
            item = await queue.get()
            items.append(item)
            if item is None:
                break

        t.join(timeout=5)
        self.assertEqual(items, ["a", "b", None])

    def test_pump_streamer_stops_on_idle_timeout_when_generation_ended(self):
        """Empty when gen_done IS set must be treated as terminal: the pump
        stops and only the None sentinel is delivered (no tokens, no error)."""
        gen_done = threading.Event()

        class _FakeStreamer2:
            def __init__(self):
                self._first = True

            def __iter__(self):
                return self

            def __next__(self):
                if self._first:
                    self._first = False
                    gen_done.set()
                    raise Empty()
                raise StopIteration

        streamer2 = _FakeStreamer2()
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        t = threading.Thread(
            target=airllm_server._pump_streamer,
            args=(streamer2, loop, queue, gen_done),
            daemon=True)
        t.start()

        items = []
        while True:
            item = await queue.get()
            items.append(item)
            if item is None:
                break

        t.join(timeout=5)
        self.assertEqual(items, [None])


if __name__ == "__main__":
    unittest.main()
