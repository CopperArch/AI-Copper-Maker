#!/usr/bin/env python3
"""AirLLM inference server — OpenAI-compatible API over AirLLM's
streamed-layer runtime (github.com/lyogavin/airllm, vendored at
vendor/airllm at v4.0.0).

Runs in its OWN venv (backend/venv-airllm) — the same pattern as
venv-imagegen: torch/transformers are heavy, opt-in dependencies that stay
out of the main app's venv/requirements.txt. The main FastAPI app never
imports this module; it reaches this server over HTTP the same way it
reaches llama.cpp / LM Studio (AIRLLM_HOST env var, model refs prefixed
"airllm/<id>").

Start it with:
  <repo>/backend/venv-airllm/bin/python -m uvicorn airllm_server:app \
      --host 127.0.0.1 --port 8082

Why a separate process at all: AirLLM's streaming hooks stream a layer's
weights disk -> GPU -> back to the meta device around every forward pass.
That is stateful and NOT safe to share across concurrent generate() calls
(upstream v4.0.0 ships no concurrency guard — two overlapping generations
would evict each other's live layers and produce garbage). This server
therefore:
  * loads exactly one model at a time (load/unload are explicit endpoints),
  * serializes every chat completion behind one lock,
  * runs generation in worker threads (torch is blocking) and pumps text
    into the async event loop for SSE streaming.

Endpoints (OpenAI-compatible subset + AirLLM extensions):
  GET  /health                  {ok, device, model, loading, error, cache_dir, disk_free}
  GET  /v1/models               OpenAI list of loaded models
  POST /v1/chat/completions     OpenAI chat completion, stream + non-stream
  GET  /v1/airllm/installed     {"installed": [repo ids fully downloaded + split]}
  POST /v1/airllm/download      {"model": "<hf repo id>"} — download + split into the
                                 cache without loading (a later /load then skips the
                                 download entirely); no-op when already installed
  POST /v1/airllm/load          {"model": "<hf repo id or local path>",
                                  "max_seq_len"?: int, "compression"?: "4bit"|"8bit"}
  POST /v1/airllm/unload        free the loaded model
"""
import asyncio
import gc
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from queue import Empty

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

REPO_DIR = Path(__file__).parent.parent
AIRLLM_VENDOR = REPO_DIR / "vendor" / "airllm" / "air_llm"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Model cache lives under the app's config dir on purpose: that dir is the
# one state dir bind-mounted into the Docker container, so downloaded
# weights (original shards + per-layer split) survive container rebuilds
# and host/container see the same cache. AIRLLM_CACHE_DIR overrides.
CACHE_DIR = Path(os.environ.get("AIRLLM_CACHE_DIR")
                 or Path.home() / ".config" / "ai-copper-maker" / "airllm-cache")
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # /health still reports the intended path; loads will fail loudly

app = FastAPI(title="AirLLM Server")


def _airllm_import_hint() -> str:
    return ("AirLLM is not available — run setup/airllm-linux.sh "
            "(clones vendor/airllm and builds backend/venv-airllm). "
            f"Checked for the vendored clone at {AIRLLM_VENDOR}.")


def _import_airllm():
    """Lazily import the vendored airllm package (vendor/airllm/air_llm puts
    the `airllm` package on sys.path). Returns None when the vendored clone
    or its torch/transformers stack is missing, instead of crashing at
    import time — the server must stay up (and /health usable) so the main
    app can report 'disconnected' rather than 'crashed'."""
    if not AIRLLM_VENDOR.is_dir():
        return None
    if str(AIRLLM_VENDOR) not in sys.path:
        sys.path.insert(0, str(AIRLLM_VENDOR))
    try:
        import airllm  # noqa: F401
        import torch  # noqa: F401
        _patch_cpu_cleanup(airllm, torch)
        _patch_fused_moe_experts(airllm, torch)
        return airllm
    except Exception as e:  # missing torch, transformers, etc.
        print(f"airllm import failed: {type(e).__name__}: {e}")
        return None


def _patch_cpu_cleanup(airllm, torch):
    """CPU hosts: AirLLM's streamed-layer post-hook calls clean_memory()
    (gc.collect() + malloc_trim + cuda.empty_cache) after EVERY decoder-layer
    forward — on GPU that is cheap VRAM reclamation, on CPU it is a full-heap
    scan per layer per token, which turns generation into minutes of
    garbage collection per token (verified live: 0.5B, 8 tokens, ~8 min and
    counting). The layer evict (module.to('meta')) is what makes streaming
    work and stays; only the global cleanup is dropped, plus one gc.collect()
    after each full generation (in _run_generation)."""
    if not torch.cuda.is_available():
        from airllm import airllm_base
        airllm_base.clean_memory = lambda: None


# Checkpoint keys AirLLM streams for MoE expert weights, in the pre-fusion
# layout transformers <=4 used (one submodule per expert index).
_FUSED_EXPERT_TAIL = re.compile(
    r"^(?P<prefix>.+)\.experts\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")


def _make_fused_expert_shim(orig, torch):
    """Fallback wrapper for accelerate.set_module_tensor_to_device.

    transformers >=5 stores Qwen3-MoE (and friends) experts as ONE fused
    module: a gate_up_proj [E, 2*inter, hidden] and a down_proj
    [E, hidden, inter] parameter instead of E expert submodules. The
    checkpoint (and therefore AirLLM's split shards) still ships the legacy
    per-expert keys `...experts.<e>.gate_proj.weight` etc., and the original
    writer walks that dotted path through attributes — dying at the fused
    module (`'Qwen3MoeExperts' object has no attribute '0'`), so every MoE
    layer forward failed. When the original walk raises AttributeError and
    the name matches the legacy expert layout, write the expert's slice
    into the fused parameter instead: expert e sits at dim 0 index e, and
    gate/up split the second dim in halves. A meta parameter cannot be
    filled in place (torch 2.14 rejects `param.data = <real tensor>` with
    "incompatible tensor type"), so the fused parameter is materialised by
    replacing the object in the module's _parameters dict — the same
    mechanism accelerate itself uses."""
    def set_compat(model, tensor_name, device, **kwargs):
        try:
            return orig(model, tensor_name, device, **kwargs)
        except AttributeError:
            m = _FUSED_EXPERT_TAIL.match(tensor_name)
            if m is None:
                raise
            experts_mod = model.get_submodule(m["prefix"] + ".experts")
            e, proj = int(m["idx"]), m["proj"]
            value = kwargs["value"]
            target = torch.device(device)
            param_name = "down_proj" if proj == "down_proj" else "gate_up_proj"
            big = experts_mod._parameters[param_name]
            if big.device.type == "meta":
                dtype = kwargs.get("dtype") or big.dtype
                experts_mod._parameters[param_name] = big = type(big)(
                    torch.empty(big.shape, dtype=dtype, device=target),
                    requires_grad=big.requires_grad)
            if kwargs.get("dtype") is not None and value.dtype != kwargs["dtype"]:
                value = value.to(kwargs["dtype"])
            if proj == "down_proj":
                big[e] = value.to(target, non_blocking=False)
            else:
                inter = big.shape[1] // 2
                lo = 0 if proj == "gate_proj" else inter
                big[e, lo:lo + inter, :] = value.to(target, non_blocking=False)

    return set_compat


def _patch_fused_moe_experts(airllm, torch):
    from airllm import airllm_base
    airllm_base.set_module_tensor_to_device = \
        _make_fused_expert_shim(airllm_base.set_module_tensor_to_device, torch)


class _State:
    """Single-model-at-a-time runtime state. Every mutation happens under
    STATE.lock; readers of `airllm_model` only ever see a fully-loaded
    model or None (a half-constructed AirLLMBaseModel is useless and its
    streaming hooks would run on top of nothing)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model_id = None
        self.airllm_model = None
        self.device = None
        self.max_seq_len = 8192
        self.loading = None      # human-readable progress while a load is in flight
        self.error = None        # last load error, for /health


STATE = _State()

# One generation at a time, full stop (see the module docstring). The
# asyncio lock is held for the WHOLE completion (prompt + full stream),
# not per chunk.
GEN_LOCK = asyncio.Lock()


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        # ROCm torch reports as cuda too — either way the driver is CUDA-style.
        return "cuda:0"
    return "cpu"


def _hub_cache_dir() -> Path:
    """Where huggingface_hub snapshots land under HF_HOME (the env default
    is HF_HOME/hub, but HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE win if set)."""
    return Path(os.environ.get("HF_HUB_CACHE")
                or os.environ.get("HUGGINGFACE_HUB_CACHE")
                or Path(os.environ.get("HF_HOME", str(CACHE_DIR))) / "hub")


def _installed_models() -> list:
    """Repo ids whose cache is complete: downloaded AND split into
    per-layer shards (the marker is the `splitted_model/` subdir AirLLM
    writes next to the snapshot, which is what makes a later load skip
    the download). A repo whose download/split is IN FLIGHT is excluded:
    the splitter creates `splitted_model/` at the start of the job, so
    counting it mid-split would report a half-done model as installed."""
    with STATE.lock:
        busy = STATE.loading or ""
    hub = _hub_cache_dir()
    ready = []
    if not hub.is_dir():
        return ready
    for entry in hub.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        repo_id = entry.name[len("models--"):].replace("--", "/", 1)
        if repo_id in busy:
            continue
        for snap in (entry / "snapshots").glob("*"):
            if snap.is_dir() and (snap / "splitted_model").is_dir():
                ready.append(repo_id)
                break
    return sorted(ready)


def _check_loaded(state: _State):
    if state.loading:
        raise HTTPException(409, f"AirLLM is still loading ({state.loading}) — poll /health.")
    if state.error:
        raise HTTPException(500, f"Last AirLLM load failed: {state.error}")
    raise HTTPException(
        409, "No AirLLM model loaded. POST /v1/airllm/load with "
             '{"model": "<huggingface repo id or local path>"}.')


@app.get("/health")
async def health():
    # torch is optional HERE on purpose: this endpoint's job is to tell the
    # main app "the server is up", which must hold even before (or without)
    # the torch stack being installed — the load endpoint reports that state
    # separately.
    device, cuda_available = None, False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if STATE.device is None:
            device = _pick_device()
    except Exception:
        pass
    disk_free = None
    try:
        disk_free = shutil.disk_usage(CACHE_DIR).free
    except OSError:
        pass
    with STATE.lock:
        return {
            "ok": True,
            "device": STATE.device or device,
            "model": STATE.model_id,
            "max_seq_len": STATE.max_seq_len,
            "loading": STATE.loading,
            "error": STATE.error,
            "cuda_available": cuda_available,
            "cache_dir": str(CACHE_DIR),
            "disk_free": disk_free,
        }


@app.get("/v1/models")
async def list_models():
    with STATE.lock:
        models = [STATE.model_id] if STATE.model_id else []
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": int(time.time()), "owned_by": "airllm"}
            for m in models
        ],
    }


@app.get("/v1/airllm/installed")
async def installed_models():
    """Repo ids fully prepared in the cache (downloaded + split). Cheap:
    a directory scan, no torch involved."""
    return {"installed": _installed_models()}


class LoadRequest(BaseModel):
    model: str
    max_seq_len: int = 8192
    compression: str | None = None  # "4bit" / "8bit" (needs bitsandbytes)


class DownloadRequest(BaseModel):
    model: str


@app.post("/v1/airllm/download")
async def download_model(req: DownloadRequest):
    """Download + split a Hugging Face repo into the cache WITHOUT loading
    it — the expensive first-use work, done ahead of time so /v1/airllm/load
    is fast. Idempotent: an already-installed repo is a no-op, and the split
    step itself re-checks free disk (AirLLM's own check_space)."""
    airllm = _import_airllm()
    if airllm is None:
        raise HTTPException(503, _airllm_import_hint())
    with STATE.lock:
        if STATE.loading:
            raise HTTPException(409, f"AirLLM is busy ({STATE.loading}).")
        if req.model in _installed_models():
            return {"model": req.model, "already_installed": True}
        STATE.loading = f"downloading {req.model} from Hugging Face"
        STATE.error = None

    def _worker():
        try:
            from airllm.utils import find_or_create_local_splitted_path
            with STATE.lock:
                STATE.loading = f"splitting {req.model} into per-layer shards"
            find_or_create_local_splitted_path(req.model, hf_token=HF_TOKEN)
            with STATE.lock:
                STATE.loading = None
        except Exception as e:
            with STATE.lock:
                STATE.loading = None
                STATE.error = f"{type(e).__name__}: {e}"
            print(f"airllm download failed: {STATE.error}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"downloading": True, "model": req.model}


@app.post("/v1/airllm/load")
async def load_model(req: LoadRequest):
    airllm = _import_airllm()
    if airllm is None:
        raise HTTPException(503, _airllm_import_hint())
    with STATE.lock:
        if STATE.loading:
            raise HTTPException(409, f"AirLLM is already loading ({STATE.loading}).")
        STATE.loading = f"{req.model} (download + per-layer split on first use)"
        STATE.error = None

    def _worker():
        with STATE.lock:
            # Drop any previous model first so its streamed layers (and its
            # split-shard memory maps) don't pile up behind the new one.
            STATE.airllm_model = None
            STATE.model_id = None
            gc.collect()
        try:
            device = _pick_device()
            kwargs = {
                "device": device,
                "max_seq_len": req.max_seq_len,
                # delete_original stays False on purpose: upstream v4.0.0's
                # remove_real_and_linked_file() has a NameError bug when the
                # path being deleted is not a symlink, and re-splitting after
                # a delete costs a full re-download anyway.
            }
            if req.compression:
                kwargs["compression"] = req.compression
            if HF_TOKEN:
                kwargs["hf_token"] = HF_TOKEN
            model = airllm.AutoModel.from_pretrained(req.model, **kwargs)
            with STATE.lock:
                STATE.airllm_model = model
                STATE.model_id = req.model
                STATE.device = device
                STATE.max_seq_len = req.max_seq_len
                STATE.loading = None
        except Exception as e:
            with STATE.lock:
                STATE.loading = None
                STATE.error = f"{type(e).__name__}: {e}"
            print(f"airllm load failed: {STATE.error}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"loading": True, "model": req.model}


@app.post("/v1/airllm/unload")
async def unload_model():
    with STATE.lock:
        had_model = STATE.airllm_model is not None
        STATE.airllm_model = None
        STATE.model_id = None
        STATE.loading = None
        STATE.error = None
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"unloaded": had_model}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    # Tolerate and ignore provider-specific extras the main app sends
    # (stream_options, etc.) — OpenAI-style servers accept unknown keys.
    model_config = {"extra": "allow"}


def _usage_response(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _apply_chat_template(airllm_model, messages: list) -> tuple:
    """Messages -> input_ids on the model's device, truncated to
    STATE.max_seq_len. Left-truncation (the tokenizer's default) keeps the
    most recent turn intact — the right edge of the prompt is where the
    model needs continuity."""
    try:
        text = airllm_model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some chat templates choke on role names outside system/user/
        # assistant (the agent loop's tool results ride on role="tool");
        # fold those into plain user text rather than fail.
        fallback = [
            {"role": m.get("role") if m.get("role") in ("system", "user", "assistant") else "user",
             "content": m.get("content") or ""}
            for m in messages
        ]
        text = airllm_model.tokenizer.apply_chat_template(
            fallback, tokenize=False, add_generation_prompt=True)
    enc = airllm_model.tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=STATE.max_seq_len, padding=False)
    device = airllm_model.device
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _run_generation(airllm_model, messages: list, temperature: float,
                    max_new_tokens: int, top_p: float, streamer=None):
    """Blocking generate() — must run in a worker thread. With streamer=None
    returns (prompt_tokens, completion_tokens, text); with a streamer the
    text has already flowed through it, so the text is "". """
    input_ids, attention_mask = _apply_chat_template(airllm_model, messages)
    tok = airllm_model.tokenizer
    # A tokenizer without an explicit pad token reports pad_token_id as the
    # sentinel -1, NOT None — passing -1 straight into generate() makes
    # transformers mask logits at index -1 (which wraps to the last vocab
    # entry) and silently corrupts every sampled token. Validate the range,
    # and fall back to eos (always a valid in-vocab token) when it's unset.
    pad_token = tok.pad_token_id
    if pad_token is None or not (0 <= pad_token < len(tok)):
        pad_token = tok.eos_token_id
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=temperature > 0,
        pad_token_id=pad_token,
    )
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = min(top_p, 1.0)
    if streamer is not None:
        kwargs["streamer"] = streamer
    out = airllm_model.generate(**kwargs)
    gc.collect()  # once per generation, not per layer (see _patch_cpu_cleanup)
    seq = out
    prompt_tokens = input_ids.shape[1]
    completion_tokens = int(seq.shape[1] - input_ids.shape[1])
    text = ""
    if streamer is None:
        # seq is (batch, seq); batch is always 1 (one completion at a time,
        # see GEN_LOCK) — slice the SEQUENCE dim after row 0, not the batch
        # dim, or decode sees zero rows and yields "".
        text = airllm_model.tokenizer.decode(seq[0, input_ids.shape[1]:],
                                             skip_special_tokens=True)
    return prompt_tokens, max(completion_tokens, 0), text


def _pump_streamer(streamer, loop, queue, gen_done):
    """Run in a worker thread: forward every streamed text chunk to `queue`
    (thread-safe via call_soon_threadsafe), then push the None sentinel.

    A TextIteratorStreamer idle timeout (`queue.Empty` from its internal
    `get(timeout=...)`) means ONLY "no token for `timeout` seconds" — a slow
    model (CPU layer-offload) legitimately exceeds that between tokens, so it
    must NOT be treated as a stream error on its own. It is terminal only once
    the generation thread has actually finished (normally or via error); while
    generation is still running the pump keeps waiting for the next token.

    The None sentinel is pushed only AFTER gen_done is set, so the final SSE
    chunk is built from a populated result (usage/error), never an empty dict.
    """
    try:
        while True:
            try:
                text = next(streamer)
            except StopIteration:
                break
            except Empty:
                if gen_done.is_set():
                    break
                continue
            loop.call_soon_threadsafe(queue.put_nowait, text)
    except Exception as e:
        loop.call_soon_threadsafe(queue.put_nowait, f"\n[stream error: {e}]")
    finally:
        gen_done.wait()
        loop.call_soon_threadsafe(queue.put_nowait, None)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    with STATE.lock:
        if STATE.airllm_model is None:
            _check_loaded(STATE)
        airllm_model = STATE.airllm_model
        model_id = STATE.model_id
    # The main app strips its "airllm/" prefix before calling; accept the
    # bare id, or the full ref, or any placeholder name for the one model
    # that is loaded.
    asked = req.model[len("airllm/"):] if req.model.startswith("airllm/") else req.model
    if asked and asked not in (model_id, "local"):
        raise HTTPException(404, f"Model '{req.model}' is not loaded. "
                                 f"Loaded: '{model_id}'.")

    max_new_tokens = req.max_tokens or 2048
    async with GEN_LOCK:
        if not req.stream:
            try:
                prompt_tokens, completion_tokens, text = await asyncio.to_thread(
                    _run_generation, airllm_model, req.messages,
                    req.temperature, max_new_tokens, req.top_p, None)
            except Exception as e:
                raise HTTPException(500, f"AirLLM generation failed: {type(e).__name__}: {e}")
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop"}],
                "usage": _usage_response(prompt_tokens, completion_tokens),
            }

        from transformers import TextIteratorStreamer
        # skip_special_tokens goes through **decode_kwargs in both the 4.x
        # and 5.x TextIteratorStreamer signatures — a decode_kwargs=dict kwarg
        # is swallowed by that same ** and silently ignored in both.
        streamer = TextIteratorStreamer(
            airllm_model.tokenizer, skip_prompt=True, skip_special_tokens=True,
            timeout=60)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        result: dict = {}

        gen_done = threading.Event()

        def _gen():
            try:
                result["usage"] = _run_generation(
                    airllm_model, req.messages, req.temperature,
                    max_new_tokens, req.top_p, streamer)[:2]
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}"
                loop.call_soon_threadsafe(queue.put_nowait, f"\n[generation error: {e}]")
            finally:
                gen_done.set()
                # No-op on transformers >=5 (the streamer auto-stops),
                # required on 4.x to release the pump.
                try:
                    streamer.end()
                except AttributeError:
                    pass

        pump_thread = threading.Thread(
            target=_pump_streamer, args=(streamer, loop, queue, gen_done),
            daemon=True)
        gen_thread = threading.Thread(target=_gen, daemon=True)
        pump_thread.start()
        gen_thread.start()

        async def _sse():
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    payload = {"id": f"chatcmpl-{int(time.time())}",
                               "object": "chat.completion.chunk",
                               "created": int(time.time()), "model": model_id,
                               "choices": [{"index": 0, "delta": {"content": chunk},
                                            "finish_reason": None}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                # Final chunk: empty choices + usage (the OpenAI shape for
                # stream_options.include_usage). An error field rides along
                # when generation failed — the main app surfaces it in the
                # reply text instead of a silent empty answer.
                usage = _usage_response(*result["usage"]) if "usage" in result else None
                final = {"id": f"chatcmpl-{int(time.time())}",
                         "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": model_id,
                         "choices": [], "usage": usage}
                if "error" in result:
                    final["error"] = result["error"]
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # If the client disconnected mid-stream, drain the queue so
                # the pump thread can exit instead of blocking forever on
                # queue.put_nowait.
                for _ in range(max(0, queue.qsize())):
                    try:
                        queue.get_nowait()
                    except Exception:
                        break

        return StreamingResponse(_sse(), media_type="text/event-stream")
