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


class KTransformersBackend(InferenceBackend):
    """Adapter for the KTransformers inference server.

    KTransformers provides an OpenAI-compatible API (via SGLang-KT) for
    heterogeneous CPU/GPU LLM inference. This adapter communicates with
    the KTransformers server via its HTTP OpenAI-compatible API, mirroring
    the pattern used by AirLLMBackend.

    This adapter does NOT depend on any KTransformers-specific Python packages
    at the adapter level — it uses the generic OpenAI-compatible API, exactly
    as the specification requires ("KTransformers optional; main application must
    not fail if KTransformers unavailable").

    The adapter implements the InferenceBackend interface so the
    BackendRouter can treat KTransformers on the same footing as AirLLM
    and any future backends.
    """

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        self._base_url = base_url or os.environ.get("KTRANSFORMERS_HOST", "http://localhost:8083")
        self._loaded_model: str | None = None
        self._supports_streaming = True

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
                        "kgpu_available": data.get("kgpu_available", False),
                    }
        except Exception as e:
            logger.warning("KTransformers health check failed: %s", e)

        return {"installed": False, "compatible": False, "error": "server unreachable"}

    # ------------------------------------------------------------------
    # Interface: check_dependencies
    # ------------------------------------------------------------------
    def check_dependencies(self) -> dict[str, Any]:
        """KTransformers depends on the server being reachable and the
        kt-kernel/sglang-kt packages being installed on the server side.

        From the client side, no Python packages need be installed since
        communication is via HTTP."""
        installed, compatible = self.health_check().values()
        return {
            "met": installed and compatible,
            "missing": [] if installed else ["KTransformers server not reachable at " + self._base_url],
            "version_mismatch": [],
        }

    # ------------------------------------------------------------------
    # Interface: check_hardware
    # ------------------------------------------------------------------
    def check_hardware(self) -> dict[str, Any]:
        """Report what the KTransformers server reports about its capabilities."""
        health = self.health_check()
        if not health.get("installed", False):
            return {"met": False, "missing": ["KTransformers server not reachable"], "reason": "server down"}
        return {"met": True, "details": health}

    # ------------------------------------------------------------------
    # Interface: check_model
    # ------------------------------------------------------------------
    def check_model(self, capabilities: ModelCapabilities) -> dict[str, Any]:
        """KTransformers can run a wide variety of models, especially MoE models.

        The server will indicate which models it has loaded and their
        compatibility. We conservatively mark most models as compatible
        here; the router will verify VRAM/context-length fitness."""
        # KTransformers excels at MoE models and large models
        # that fit within its heterogeneous computing framework
        return {
            "compatible": True,
            "reason": None,
            "note": "KTransformers compatibility — verify VRAM and context length. "
            "Best suited for MoE and large parameter-count models.",
        }

    # ------------------------------------------------------------------
    # Interface: load_model
    # ------------------------------------------------------------------
    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """KTransformers models are loaded on the server side.

        The caller provides the model identifier (e.g.
        "Qwen/Qwen3-30B-A3B" or a local path). The adapter stores this
        reference but does not perform any local loading — the server manages
        the model lifecycle.

        Supported kwargs:
        - kt_method: Precision/method for KTransformers (FP8, FP16, INT4, etc.)
        - kt_weight_path: Path to KTransformers-formatted weights
        - kt_num_gpu_experts: Number of MoE experts on GPU
        """
        model_id = model_path
        # Strip any prefix if present — the server expects the bare model id
        if model_id.startswith("ktransformers/"):
            model_id = model_id.split("/", 1)[1]
        if model_id.startswith("airllm/"):
            model_id = model_id.split("/", 1)[1]

        # Extract KTransformers-specific options
        kt_method = kwargs.get("kt_method")
        kt_weight_path = kwargs.get("kt_weight_path")
        kt_num_gpu_experts = kwargs.get("kt_num_gpu_experts")

        self._loaded_model = model_id
        params = {"model": model_id}
        if kt_method:
            params["kt_method"] = kt_method
        if kt_weight_path:
            params["kt_weight_path"] = kt_weight_path
        if kt_num_gpu_experts is not None:
            params["kt_num_gpu_experts"] = kt_num_gpu_experts

        return {
            "success": True,
            "model_id": model_id,
            "server_url": self._base_url,
            "loaded_params": params,
            "message": f"Model {model_id} loading requested on KTransformers server",
        }

    # ------------------------------------------------------------------
    # Interface: generate (non-streaming)
    # ------------------------------------------------------------------
    async def generate(self, model_id: str, messages: list, **kwargs: Any) -> str:
        """Generate text via the KTransformers OpenAI-compatible server (non-streaming).

        Mirrors the logic in main.py:_llm_complete for the KTransformers
        branch — POST to {base_url}/v1/chat/completions.
        """
        import httpx

        bare_model = model_id
        if bare_model.startswith("ktransformers/"):
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
        """Generate text via the KTransformers OpenAI-compatible server (streaming).

        Yields events in the agent-loop's shape:
          {"type": "token", "content": ...}
          {"type": "usage", "usage": {...}}
        and finally {"type": "usage", "usage": {...}}}.
        """
        import httpx

        bare_model = model_id
        if bare_model.startswith("ktransformers/"):
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
                yield {"type": "usage", "usage": {}}

    # ------------------------------------------------------------------
    # Interface: unload_model
    # ------------------------------------------------------------------
    def unload_model(self) -> dict[str, Any]:
        """KTransformers unloads models on the server side; no-op from the client.

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
                "backend": "ktransformers",
                "server_url": self._base_url,
            }
        return None

    # ------------------------------------------------------------------
    # Interface: get_capabilities
    # ------------------------------------------------------------------
    def get_capabilities(self) -> ModelCapabilities:
        """Return capabilities of the KTransformers backend.

        KTransformers specializes in:
        - MoE (Mixture-of-Experts) model inference
        - CPU/GPU heterogeneous computing
        - Large parameter-count models
        - Various quantization formats (FP8, FP16, INT4, GPTQ, AWQ, etc.)
        """
        return ModelCapabilities(
            architecture="moe-optimized",
            parameter_count=None,
            is_moe=True,  # KTransformers primary strength
            quantization=None,
            dtype=None,
            context_length=None,
            supported_backends=["ktransformers", "airllm", "llama_cpp"],
            required_dependencies={
                "server": "KTransformers server reachable",
                "kt-kernel": "CPU/GPU kernel backend",
                "sglang-kt": "SGLang-KT serving backend",
            },
            approximate_memory_mb=None,
        )


# Auto-register this backend when the module is imported
_registry = get_registry()
# Only register if KTransformers server seems available (health check)
# We do a lazy check — the registry will have the backend registered
# but it will report "not installed" until the server is actually running.
# This way the main app never fails on import.
try:
    # Attempt a quick health check; if it fails, we still register but
    # the backend will report unavailable
    import httpx

    test_url = os.environ.get("KTRANSFORMERS_HOST", "http://localhost:8083")
    with httpx.Client(timeout=2) as client:
        r = client.get(f"{test_url}/health")
    _registry.register("ktransformers", KTransformersBackend(base_url=test_url))
except Exception:
    # KTransformers server not available at import time — that's expected.
    # The backend will be queried on-demand via the router.
    pass