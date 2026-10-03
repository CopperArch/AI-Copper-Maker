from __future__ import annotations

import json
import logging
import os
from typing import Any

from inference_backend import (
    InferenceBackend,
    ModelCapabilities,
    get_registry,
)

logger = logging.getLogger(__name__)

AIRLLM_HOST = os.environ.get("AIRLLM_HOST", "http://localhost:8082")


class AirLLMBackend(InferenceBackend):
    """Adapter wrapping the existing AirLLM OpenAI-compatible server.

    This backend does NOT modify or re-export any AirLLM internal modules.
    It communicates with the AirLLM server via its HTTP OpenAI-compatible
    API ( /v1/chat/completions ), exactly as the existing main.py
    _llm_complete and _stream_openai_compatible_chat functions do.

    The adapter implements the InferenceBackend interface so the
    BackendRouter can treat AirLLM on the same footing as KTransformers
    and any future backends.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or AIRLLM_HOST
        self._loaded_model: str | None = None

    # ------------------------------------------------------------------
    # Interface: health_check
    # ------------------------------------------------------------------
    def health_check(self) -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(f"{self._base_url}/health")
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "installed": True,
                        "compatible": True,
                        "max_seq_len": data.get("max_seq_len"),
                        "server_version": data.get("version"),
                    }
        except Exception as e:
            logger.warning("AirLLM health check failed: %s", e)

        return {"installed": False, "compatible": False, "error": "server unreachable"}

    # ------------------------------------------------------------------
# Interface: check_dependencies
# ------------------------------------------------------------------
    def check_dependencies(self) -> dict[str, Any]:
        """AirLLM depends on the server being reachable; no Python packages
        need be installed in the process since the server runs externally."""
        health = self.health_check()
        installed = health.get("installed", False)
        compatible = health.get("compatible", False)
        return {
            "met": installed and compatible,
            "missing": [] if installed else ["AirLLM server not reachable at " + self._base_url],
            "version_mismatch": [],
        }

    # ------------------------------------------------------------------
    # Interface: check_hardware
    # ------------------------------------------------------------------
    def check_hardware(self) -> dict[str, Any]:
        """Report what the AirLLM server reports about its capabilities."""
        health = self.health_check()
        if not health.get("installed", False):
            return {"met": False, "missing": ["AirLLM server not reachable"], "reason": "server down"}
        return {"met": True, "details": health}

    # ------------------------------------------------------------------
    # Interface: check_model
    # ------------------------------------------------------------------
    def check_model(self, capabilities: ModelCapabilities) -> dict[str, Any]:
        """AirLLM can run any model the server supports.

        The existing AirLLM server (v4.0.0) is broadly compatible with
        dense and MoE models that fit within its configured context window.
        We conservatively mark everything as compatible here; the router
        will later verify VRAM/context-length fitness.
        """
        return {
            "compatible": True,
            "reason": None,
            "note": "AirLLM server compatibility — verify VRAM and context length",
        }

    # ------------------------------------------------------------------
    # Interface: load_model
    # ------------------------------------------------------------------
    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """AirLLM models are loaded on the server side.

        The caller provides the model identifier (e.g.
        "airllm/Qwen/Qwen3-30B-A3B"). The adapter stores this reference
        but does not perform any local loading — the server manages the
        model lifecycle.
        """
        model_id = model_path
        if not model_path.startswith("airllm/"):
            model_id = "airllm/" + model_path
        self._loaded_model = model_id
        return {
            "success": True,
            "model_id": model_id,
            "server_url": self._base_url,
            "message": f"Model {model_id} loading requested on AirLLM server",
        }

    # ------------------------------------------------------------------
    # Interface: generate (non-streaming)
    # ------------------------------------------------------------------
    async def generate(self, model_id: str, messages: list, **kwargs: Any) -> str:
        """Generate text via the AirLLM OpenAI-compatible server (non-streaming).

        Mirrors the logic in main.py:_llm_complete for the local AirLLM
        branch — POST to {base_url}/v1/chat/completions.
        """
        import httpx

        # Strip "airllm/" prefix if present — the server expects the
        # bare model id.
        bare_model = model_id
        if bare_model.startswith("airllm/"):
            bare_model = bare_model.split("/", 1)[1]

        payload = {
            "model": bare_model,
            "messages": messages,
            "stream": False,
            "temperature": kwargs.get("temperature", 0.2),
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            choices = r.json().get("choices") or [{}]
            return (choices[0].get("message") or {}).get("content", "")

    # ------------------------------------------------------------------
    # Interface: generate (streaming)
    # ------------------------------------------------------------------
    async def generate_stream(self, model_id: str, messages: list, **kwargs: Any):
        """Generate text via the AirLLM OpenAI-compatible server (streaming).

        Yields events in the agent-loop's shape:
          {"type": "token", "content": ...}
          {"type": "usage", "usage": {...}}
        and finally {"type": "usage", "usage": {...}}}.
        """
        import httpx

        bare_model = model_id
        if bare_model.startswith("airllm/"):
            bare_model = bare_model.split("/", 1)[1]

        payload = {
            "model": bare_model,
            "messages": messages,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.2),
            "stream_options": {"include_usage": True},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            ) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    for line in chunk.decode(errors="replace").split("\n"):
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]" or not payload:
                            continue
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = data.get("choices") or [{}]
                        if isinstance(choices, dict):
                            choices = [choices]
                        if choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield {"type": "token", "content": content}
                            usage_field = data.get("usage")
                            if usage_field:
                                from main import _normalize_provider_usage

                                yield {"type": "usage", "usage": _normalize_provider_usage(usage_field)}
                # yield tool_use_stop if there was a pending tool call
                yield {"type": "usage", "usage": {}}

    # ------------------------------------------------------------------
    # Interface: unload_model
    # ------------------------------------------------------------------
    def unload_model(self) -> dict[str, Any]:
        """AirLLM unloads models on the server side; no-op from the client.

        Returns
        -------
        dict
            {"success": True, "error": None}
        """
        self._loaded_model = None
        return {"success": True, "error": None}

    # ------------------------------------------------------------------
    # Interface: get_model_info
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict[str, Any] | None:
        """Return info about the currently loaded model, if any."""
        if self._loaded_model:
            return {
                "model": self._loaded_model,
                "backend": "airllm",
                "server_url": self._base_url,
            }
        return None

    # ------------------------------------------------------------------
    # Interface: get_capabilities
    # ------------------------------------------------------------------
    def get_capabilities(self) -> ModelCapabilities:
        """Return capabilities of the AirLLM backend.

        AirLLM can run essentially any model that the server has loaded
        and that fits within its context window and VRAM.
        """
        return ModelCapabilities(
            architecture="auto-detected-by-server",
            parameter_count=None,
            is_moe=False,  # conservative — MoE supported but undetected without server query
            quantization=None,
            dtype=None,
            context_length=None,
            supported_backends=["airllm"],
            required_dependencies={"server": "http://localhost:8082 reachable"},
            approximate_memory_mb=None,
        )


# Auto-register this backend when the module is imported
_registry = get_registry()
_registry.register("airllm", AirLLMBackend())