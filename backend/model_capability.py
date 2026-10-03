from __future__ import annotations

import re

from inference_backend import ModelCapabilities


def _extract_parameter_count(model_path: str) -> int | None:
    """Try to extract parameter count from a model path/identifier.

    Looks for patterns like:
    - "7B", "7B/", "7B-instruct" → 7 billion
    - "30B", "30B-A3B" → 30 billion
    - "671B" → 671 billion
    - Numbers directly in the path
    """
    if not model_path:
        return None

    # Match patterns like N B where N is a number
    patterns = [
        r"(\d+(?:\.\d+)?)\s*[Bb]",  # "7B", "30B", "671B"
        r"/(\d+(?:\.\d+)?)\s*[Bb]",  # "owner/7B"
    ]
    for pat in patterns:
        m = re.search(pat, model_path)
        if m:
            try:
                val = float(m.group(1))
                # Convert to integer billions if it's a whole number
                if val == int(val):
                    return int(val * 1_000_000_000)
                return int(val * 1_000_000_000)
            except ValueError:
                continue
    return None


def _detect_architecture(model_path: str) -> str:
    """Detect model architecture from the model path/identifier.

    Looks for known architecture tokens in the model name or path.
    """
    if not model_path:
        return "unknown"

    mp = model_path.lower()
    # MoE patterns
    moe_tokens = ["mixtral", "moe", "moe-", "mixture-of-experts", "gpt4", "gpt-4"]
    for tok in moe_tokens:
        if tok in mp:
            return "moe"

    # Architecture patterns
    arch_tokens = {
        "llama": "llama",
        "qwen": "qwen",
        "baichuan": "baichuan",
        "internlm": "internlm",
        "glm": "glm",
        "mistral": "mistral",
        "mixtral": "mixtral",
        "phi": "phi",
        "chatglm": "chatglm",
        "yi": "yi",
        "phi-3": "phi-3",
        "deepseek": "deepseek",
        "llama-cpp": "llama_cpp",
    }
    for tok, arch in arch_tokens.items():
        if tok in mp:
            return arch

    return "unknown"


def _is_moe_model(model_path: str) -> bool:
    """Check if a model is a Mixture-of-Experts model."""
    mp = model_path.lower()
    moe_indicators = ["moe", "mixtral", "mixture-of-experts", "gpt4", "gpt-4"]
    return any(tok in mp for tok in moe_indicators)


def detect_model_capabilities(model_path: str) -> ModelCapabilities:
    """Detect and compute model capabilities from a model identifier/path.

    This is the central model inspection layer. It determines:
    - Architecture (dense vs MoE)
    - Parameter count (from naming conventions)
    - Quantization format (if indicated)
    - Context length hints (if available)
    - Supported backends based on architecture
    - Required dependencies

    Parameters
    ----------
    model_path : str
        Model identifier or path (e.g. "airllm/Qwen/Qwen3-30B-A3B",
        "meta-llama/Llama-3.1-8B-Instruct", "llama-cpp/model.gguf").

    Returns
    -------
    ModelCapabilities
        A ModelCapabilities instance populated with detected information.
    """
    architecture = _detect_architecture(model_path)
    is_moe = _is_moe_model(model_path)
    param_count = _extract_parameter_count(model_path)

    # Determine quantization from the model path or common patterns
    quantization = None
    mp = model_path.lower()
    q_patterns = [
        (r"[\-\_]?q[0-9]\\b", "GPTQ"),
        (r"[\-\_]?int4\\b", "INT4"),
        (r"[\-\_]?int8\\b", "INT8"),
        (r"[\-\_]?fp8\\b", "FP8"),
        (r"[\-\_]?bf16\\b", "BF16"),
        (r"[\-\_]?fp16\\b", "FP16"),
    ]
    for pat, qname in q_patterns:
        if re.search(pat, mp):
            quantization = qname
            break

    # Determine context length hints
    context_length = None
    # Common context lengths in model names
    ctx_patterns = [
        r"(\d+)[Kk]\s*context",
        r"context[-\_]?len[:\s]+(\d+)",
        r"(\d+)[Kk]",
    ]
    for pat in ctx_patterns:
        m = re.search(pat, mp)
        if m:
            val = int(m.group(1))
            if "K" in pat or "k" in pat.lower():
                val *= 1000
            if context_length is None or val > context_length:
                context_length = val

    # Determine supported backends
    supported_backends = ["airllm"]  # AirLLM is always a fallback
    if architecture == "moe" or is_moe:
        supported_backends.append("ktransformers")
    supported_backends.append("llama_cpp")  # GGUF fallback

    # Required dependencies based on architecture and quantization
    required_dependencies: dict[str, str | list[str]] = {"python": "3.10+"}
    if is_moe or architecture == "moe":
        required_dependencies["transformers"] = ">=5.0.0"
    if quantization and quantization in ("GPTQ", "AWQ"):
        required_dependencies["ggml"] = ">=2.0.0"
    if quantization == "FP8":
        required_dependencies["apex"] = ">=0.0.0"

    # Approximate memory estimate (in MB)
    approx_memory_mb = None
    if param_count and param_count > 0:
        # Very rough: ~4 bytes per parameter for FP16, ~2 for INT8
        if quantization in ("INT8", "FP16"):
            approx_memory_mb = int(param_count * 2 / 1_024 / 1_024)
        elif quantization in ("GPTQ", "AWQ"):
            approx_memory_mb = int(param_count * 1 / 1_024 / 1_024)
        else:  # FP16/BF16 default
            approx_memory_mb = int(param_count * 2 / 1_024 / 1_024)

    return ModelCapabilities(
        architecture=architecture,
        parameter_count=param_count,
        is_moe=is_moe,
        quantization=quantization,
        dtype=None,  # will be set at runtime based on hardware
        context_length=context_length,
        supported_backends=supported_backends,
        required_dependencies=required_dependencies,
        approximate_memory_mb=approx_memory_mb,
    )