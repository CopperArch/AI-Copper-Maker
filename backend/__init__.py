# AI Copper Maker - Backend Package
# Provides inference backend abstraction and integration

from .inference_backend import InferenceBackend, ModelCapabilities, BackendRegistry
from .airllm_backend import AirLLMBackend
from .model_capability import detect_model_capabilities
from .hardware_detector import detect_hardware, HardwareInfo
from .backend_router import BackendRouter
from .backend_config import load_backend_config, get_backend_mode, get_airllm_config, get_ktransformers_config, suggest_backend
from .ktransformers_backend import KTransformersBackend

__all__ = [
    "InferenceBackend",
    "ModelCapabilities",
    "BackendRegistry",
    "AirLLMBackend",
    "detect_model_capabilities",
    "HardwareInfo",
    "detect_hardware",
    "BackendRouter",
    "load_backend_config",
    "get_backend_mode",
    "get_airllm_config",
    "get_ktransformers_config",
    "suggest_backend",
    "KTransformersBackend",
]