from __future__ import annotations

import abc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelCapabilities:
    """Capabilities and metadata for a model, computed by ModelCapabilityDetector."""

    def __init__(
        self,
        architecture: str,
        parameter_count: int | None,
        is_moe: bool,
        quantization: str | None,
        dtype: str | None,
        context_length: int | None,
        supported_backends: list[str],
        required_dependencies: dict[str, str | list[str]] | None = None,
        approximate_memory_mb: int | None = None,
    ):
        self.architecture = architecture
        self.parameter_count = parameter_count
        self.is_moe = is_moe
        self.quantization = quantization
        self.dtype = dtype
        self.context_length = context_length
        self.supported_backends = supported_backends
        self.required_dependencies = required_dependencies or {}
        self.approximate_memory_mb = approximate_memory_mb

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "is_moe": self.is_moe,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "context_length": self.context_length,
            "supported_backends": self.supported_backends,
            "required_dependencies": self.required_dependencies,
            "approximate_memory_mb": self.approximate_memory_mb,
        }


class InferenceBackend(abc.ABC):
    """Abstract base class for all inference backends.

    Each backend must implement the core lifecycle operations and provide
    self-describing capabilities so the BackendRouter can make informed
    selection decisions without knowing internal backend details.
    """

    @abc.abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Return a health/status dict.

        Returns
        -------
        dict
            Must contain at least:
            - "installed": bool
            - "compatible": bool
        """

    @abc.abstractmethod
    def check_dependencies(self) -> dict[str, Any]:
        """Check that all required packages and versions are available.

        Returns
        -------
        dict
            - "met": bool
            - "missing": list[str]
            - "version_mismatch": list[tuple[str, str, str]]  # (pkg, required, actual)
        """

    @abc.abstractmethod
    def check_hardware(self) -> dict[str, Any]:
        """Check that the hardware requirements are satisfied.

        Returns
        -------
        dict
            - "met": bool
            - "missing": list[str]  # e.g. ["GPU with 24GB VRAM", "AMX support"]
        """

    @abc.abstractmethod
    def check_model(self, capabilities: ModelCapabilities) -> dict[str, Any]:
        """Verify that this backend can load and run the given model.

        Parameters
        ----------
        capabilities : ModelCapabilities
            The model's capabilities as detected by ModelCapabilityDetector.

        Returns
        -------
        dict
            - "compatible": bool
            - "reason": str | None  # human-readable explanation if not compatible
        """

    @abc.abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """Load a model for inference.

        Parameters
        ----------
        model_path : str
            Path or identifier for the model (HuggingFace repo ID, local path, etc.).
        **kwargs
            Backend-specific loading options (dtype, device, quantization, etc.).

        Returns
        -------
        dict
            - "success": bool
            - "model_id": str  # identifier the backend uses internally
            - "error": str | None
        """

    @abc.abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text completion given a prompt.

        This is the primary inference method. Subclasses may yield tokens
        (streaming) or return a completed string (non-streaming), according
        to their design. The caller in main.py should be agnostic to this
        detail — it should work with both async generators and plain results.
        """

    @abc.abstractmethod
    def unload_model(self) -> dict[str, Any]:
        """Unload the currently loaded model and free resources.

        Returns
        -------
        dict
            - "success": bool
            - "error": str | None
        """

    @abc.abstractmethod
    def get_model_info(self) -> dict[str, Any] | None:
        """Return information about the currently loaded model.

        Returns
        -------
        dict | None
            May contain: model_name, architecture, parameter_count,
            context_length, loaded_on, memory_used_mb, etc.
            Returns None if no model is currently loaded.
        """

    @abc.abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Return the capabilities of this backend (what models it can run).

        Returns
        -------
        ModelCapabilities
        """


class BackendRegistry:
    """Central registry for discovered backends.

    Keeps track of all available backend instances, their status, and
    provides lookup by name. The BackendRouter uses this registry to
    query available backends and their current state.
    """

    def __init__(self) -> None:
        self._backends: dict[str, InferenceBackend] = {}
        self._names: dict[str, str] = {}  # human-readable name -> class name

    def register(self, name: str, backend: InferenceBackend) -> None:
        """Register a backend instance under a given name.

        Parameters
        ----------
        name : str
            Unique identifier for this backend (e.g. "airllm", "ktransformers").
        backend : InferenceBackend
            The backend instance.
        """
        self._backends[name] = backend
        self._names[name] = backend.get_capabilities().architecture if hasattr(backend, "get_capabilities") else "unknown"
        logger.info("Registered backend: %s (%s)", name, self._names[name])

    def unregister(self, name: str) -> None:
        """Unregister a backend.

        Parameters
        ----------
        name : str
        """
        if name in self._backends:
            del self._backends[name]
            if name in self._names:
                del self._names[name]

    def get(self, name: str) -> InferenceBackend | None:
        """Get a backend instance by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        InferenceBackend | None
        """
        return self._backends.get(name)

    def list(self) -> list[str]:
        """List registered backend names.

        Returns
        -------
        list[str]
        """
        return list(self._backends.keys())

    def has(self, name: str) -> bool:
        """Check if a backend is registered.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        return name in self._backends


# Global registry instance — backends register themselves on import
_global_registry = BackendRegistry()


def get_registry() -> BackendRegistry:
    """Return the global backend registry.

    Modules that need to query or register backends should use this
    function rather than accessing the registry directly.
    """
    return _global_registry