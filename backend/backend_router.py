from __future__ import annotations

import logging
import os
from typing import Any

from hardware_detector import (
    detect_hardware,
    check_gpu_memory_fitness,
    HardwareInfo,
)
from inference_backend import (
    InferenceBackend,
    ModelCapabilities,
    get_registry,
)
from model_capability import detect_model_capabilities

logger = logging.getLogger(__name__)


class BackendSelectionError(Exception):
    """Raised when no compatible backend can be selected for the given model/hardware."""

    def __init__(self, message: str, suggested_action: str | None = None) -> None:
        super().__init__(message)
        self.suggested_action = suggested_action


class BackendRouter:
    """Router that determines which inference backend to use for a given model and hardware.

    The router supports three modes:
    - "auto" (default): automatically select the best compatible backend based on
      model architecture, hardware capabilities, and installed backends.
    - "airllm": always use the AirLLM backend (requires AirLLM server reachable).
    - "ktransformers": always use the KTransformers backend (requires KTransformers
      installed and compatible).

    Key design principles:
    - Never use simplistic rules such as "MoE = KTransformers, Dense = AirLLM"
    - Always verify actual compatibility via backend.check_model()
    - User explicit config always overrides auto mode
    - If requested backend is unavailable, provide useful error explaining why
    - Main application must remain usable if KTransformers is not installed
    - Never automatically upgrade/downgrade core dependencies
    """

    #: Default backend selection mode
    DEFAULT_MODE = "auto"

    def __init__(
        self,
        mode: str | None = None,
        airllm_base_url: str | None = None,
        ktransformers_force_install: bool = False,
        unattended_install: bool = False,
    ) -> None:
        """Initialize the BackendRouter.

        Parameters
        ----------
        mode : str | None
            Backend selection mode. One of "auto", "airllm", "ktransformers".
            Defaults to "auto".
        airllm_base_url : str | None
            Base URL for the AirLLM server. Defaults to AIRLLM_HOST env var
            or "http://localhost:8082".
        ktransformers_force_install : bool
            If True, attempt to install KTransformers in auto mode if it's not
            installed but the model appears compatible. Defaults to False.
        unattended_install : bool
            If True, install dependencies without prompting for confirmation.
            Defaults to False (prompts for confirmation).
        """
        self.mode = mode or self.DEFAULT_MODE
        self._airllm_base_url = airllm_base_url or os.environ.get(
            "AIRLLM_HOST", "http://localhost:8082"
        )
        self._ktransformers_force_install = ktransformers_force_install
        self._unattended_install = unattended_install
        self._registry = get_registry()

        # Ensure AirLLM backend is registered
        if not self._registry.has("airllm"):
            from airllm_backend import AirLLMBackend

            self._registry.register("airllm", AirLLMBackend(base_url=self._airllm_base_url))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_backend(
        self,
        model_path: str,
        hardware: HardwareInfo | None = None,
        requested_backend: str | None = None,
    ) -> tuple[InferenceBackend, dict[str, Any]]:
        """Select the best backend for the given model and hardware.

        Parameters
        ----------
        model_path : str
            Model identifier or path (e.g. "airllm/Qwen/Qwen3-30B-A3B",
            "meta-llama/Llama-3.1-8B-Instruct").
        hardware : HardwareInfo | None
            Detected hardware. If None, will call detect_hardware().
        requested_backend : str | None
            Explicit backend request. One of None (auto), "airllm", "ktransformers".
            If provided, the mode is forced to that backend regardless of
            auto-detection results.

        Returns
        -------
        tuple[InferenceBackend, dict[str, Any]]
            (selected_backend, selection_info) where selection_info contains:
            - "mode": the mode used ("auto", "airllm", "ktransformers")
            - "reason": human-readable reason for the selection
            - "alternatives": list of viable alternatives
            - "install_required": whether KTransformers needed installation (if ktransformers selected)

        Raises
        ------
        BackendSelectionError
            If no compatible backend can be found.
        """
        # Resolve the mode: explicit request overrides all
        if requested_backend is not None:
            requested = requested_backend.lower()
            if requested not in ("auto", "airllm", "ktransformers"):
                raise ValueError(f"Invalid backend mode: {requested}. Must be one of: auto, airllm, ktransformers")
            mode = requested
        else:
            mode = self.mode

        logger.info("Backend selection mode: %s", mode)

        # ------------------------------------------------------------------
        # Step 1: Resolve model capabilities
        # ------------------------------------------------------------------
        model_caps = detect_model_capabilities(model_path)
        logger.info("Model capabilities: %s", model_caps.to_dict())

        # ------------------------------------------------------------------
        # Step 2: Detect hardware if not provided
        # ------------------------------------------------------------------
        if hardware is None:
            hardware = detect_hardware()
        logger.info("Hardware: %s", hardware.to_dict())

        # ------------------------------------------------------------------
        # Step 3: Mode-specific selection
        # ------------------------------------------------------------------
        # Dispatch on the RESOLVED mode (an explicit requested_backend
        # overrides the router's configured mode), not self.mode.
        if mode == "airllm":
            return self._select_airllm(model_caps, hardware)
        if mode == "ktransformers":
            return self._select_ktransformers(model_caps, hardware)
        # mode == "auto" (default)
        return self._select_auto(model_caps, hardware)

    def _select_airllm(
        self, model_caps: ModelCapabilities, hardware: HardwareInfo
    ) -> tuple[InferenceBackend, dict[str, Any]]:
        """Select AirLLM backend, verifying compatibility.

        AirLLM is always a viable option as it uses a remote server;
        the check ensures the server is reachable.
        """
        backend = self._registry.get("airllm")
        if backend is None:
            raise BackendSelectionError(
                "AirLLM backend not available. Ensure the AirLLM server is "
                "configured and reachable.",
                suggested_action="Check AIRLLM_HOST env var or run setup/airllm-linux.sh",
            )

        # Health check
        health = backend.health_check()
        if not health.get("installed", False):
            raise BackendSelectionError(
                "AirLLM server is not reachable. "
                "Cannot select AirLLM backend.",
                suggested_action=health.get("error", "Check AirLLM server status"),
            )

        # Check model compatibility
        model_check = backend.check_model(model_caps)
        if not model_check.get("compatible", False):
            raise BackendSelectionError(
                f"AirLLM reports model is not compatible: {model_check.get('reason', 'unknown')}",
                suggested_action="Select a different backend or a model compatible with AirLLM",
            )

        selection_info = {
            "mode": "airllm",
            "reason": "AirLLM server selected — broadly compatible with detected model",
            "alternatives": [],
            "install_required": False,
        }
        return backend, selection_info

    def _select_ktransformers(
        self, model_caps: ModelCapabilities, hardware: HardwareInfo
    ) -> tuple[InferenceBackend, dict[str, Any]]:
        """Select KTransformers backend, verifying installation and compatibility.

        If KTransformers is not installed, this will either:
        - In auto mode: attempt to install it (with confirmation) if the model
          appears compatible
        - In explicit ktransformers mode: raise an error explaining what's needed
        """
        # Check if KTransformers is importable/available
        try:
            import ktransformers  # noqa: F401

            kt_available = True
        except ImportError:
            kt_available = False

        if kt_available:
            # KTransformers is installed — verify it can handle this model
            try:
                backend = self._registry.get("ktransformers")
                if backend is None:
                    from ktransformers_backend import KTransformersBackend as _KTBackend

                    backend = _KTBackend()
                    self._registry.register("ktransformers", backend)

                # Health check
                health = backend.health_check()
                if not health.get("installed", False):
                    raise BackendSelectionError(
                        "KTransformers is installed but not properly functional.",
                        suggested_action="Reinstall KTransformers: pip install ktransformers",
                    )

                # Check model compatibility
                model_check = backend.check_model(model_caps)
                if not model_check.get("compatible", False):
                    raise BackendSelectionError(
                        f"KTransformers reports model not compatible: {model_check.get('reason', 'unknown')}",
                        suggested_action="Check model architecture/quantization compatibility",
                    )

                selection_info = {
                    "mode": "ktransformers",
                    "reason": "KTransformers selected — model compatible with detected hardware",
                    "alternatives": [],
                    "install_required": False,
                }
                return backend, selection_info

            except ImportError:
                # KTransformers import failed even though import appeared to work
                kt_available = False

        # KTransformers not available — handle based on mode
        if self.mode == "ktransformers":
            raise BackendSelectionError(
                "KTransformers backend explicitly requested but KTransformers is not installed. "
                "Install with: pip install ktransformers\n"
                "Note: KTransformers requires kt-kernel and sglang-kt packages for full functionality.",
                suggested_action="pip install ktransformers [and kt-kernel sglang-kt]",
            )

        # mode == "auto" — try to install if configured to do so
        if self._ktransformers_force_install or self._should_attempt_install(
            model_caps, hardware
        ):
            return self._install_and_select_ktransformers(model_caps, hardware)

        # Auto mode: suggest KTransformers as an alternative but don't force install
        selection_info = {
            "mode": "auto",
            "reason": "KTransformers not installed — AirLLM selected as fallback. "
            "Install KTransformers for MoE model support.",
            "alternatives": [
                {
                    "backend": "ktransformers",
                    "reason": "Best for MoE models with CPU/GPU heterogeneous computing",
                    "install_required": True,
                }
            ],
            "install_required": True,
        }
        # In auto mode without forced install, fall through to AirLLM
        return self._select_airllm(model_caps, hardware)

    def _should_attempt_install(
        self, model_caps: ModelCapabilities, hardware: HardwareInfo
    ) -> bool:
        """Determine whether we should attempt KTransformers installation.

        We attempt installation when:
        - The model is MoE (KTransformers excels at MoE)
        - OR the model has large parameter count and KTransformers could help
        - AND hardware has some GPU support (even limited) or good CPU
        - User has not explicitly disabled unattended install
        """
        # MoE models are the primary use case for KTransformers
        if model_caps.is_moe:
            return True

        # Large models (>30B) could benefit from KTransformers heterogeneous computing
        if model_caps.parameter_count and model_caps.parameter_count > 30_000_000_000:
            return True

        # If we have GPU with some VRAM, KTransformers CPU/GPU hybrid could help
        if hardware.has_gpu and hardware.gpu_vram_mb and hardware.gpu_vram_mb < 24000:
            # Small GPU (e.g. 8GB-12GB) — KTransformers hybrid could offload experts to CPU
            return True

        return False

    def _install_and_select_ktransformers(
        self, model_caps: ModelCapabilities, hardware: HardwareInfo
    ) -> tuple[InferenceBackend, dict[str, Any]]:
        """Attempt to install KTransformers and then select it.

        Note: This is a best-effort installation. The router does NOT blindly
        install every available backend. It only installs KTransformers when:
        - The user/configured setting permits it
        - The model appears compatible (MoE or very large)
        - We can verify the installation afterwards
        """
        from ktransformers_backend import KTransformersBackend

        # Check if already registered
        backend = self._registry.get("ktransformers")
        if backend is None:
            backend = KTransformersBackend()
            self._registry.register("ktransformers", backend)

        # Health check before install
        health = backend.health_check()
        if health.get("installed", False):
            # Already installed, just verify compatibility
            model_check = backend.check_model(model_caps)
            if model_check.get("compatible", False):
                selection_info = {
                    "mode": "ktransformers",
                    "reason": "KTransformers already installed and model compatible",
                    "alternatives": [],
                    "install_required": False,
                }
                return backend, selection_info
            raise BackendSelectionError(
                f"KTransformers installed but model not compatible: {model_check.get('reason', 'unknown')}",
                suggested_action="Select a different model or check quantization compatibility",
            )

        # Attempt installation
        selection_info = {
            "mode": "ktransformers",
            "reason": "Attempting to install KTransformers for model compatibility",
            "alternatives": [],
            "install_required": True,
            "install_in_progress": True,
        }

        # In unattended mode, proceed with installation
        if self._unattended_install:
            install_result = self._perform_ktransformers_install()
        else:
            # Prompt user — in a real UI, this would be a dialog
            # For now, we'll attempt it and log the decision
            logger.info(
                "KTransformers not installed. Attempting installation for MoE model compatibility. "
                "Use --unattended-install to skip this prompt, or --backend airllm to force AirLLM."
            )
            install_result = self._perform_ktransformers_install()

        # Verify installation
        if install_result.get("success", False):
            # Re-check health and compatibility
            health = backend.health_check()
            if health.get("installed", False):
                model_check = backend.check_model(model_caps)
                if model_check.get("compatible", False):
                    selection_info["install_in_progress"] = False
                    selection_info["install_required"] = False
                    selection_info["reason"] = (
                        "KTransformers installed successfully and model is compatible"
                    )
                    return backend, selection_info
                selection_info["reason"] = (
                    f"KTransformers installed but model not fully compatible: "
                    f"{model_check.get('reason', 'unknown')}"
                )
            else:
                selection_info["reason"] = (
                    "KTransformers installation reported success but health check failed"
                )
        else:
            selection_info["reason"] = (
                f"KTransformers installation failed: {install_result.get('error', 'unknown error')}"
            )

        # Fall back to AirLLM if installation failed or failed verification
        logger.warning(
            "KTransformers installation/verification failed, falling back to AirLLM: %s",
            selection_info["reason"],
        )
        return self._select_airllm(model_caps, hardware)

    def _perform_ktransformers_install(self) -> dict[str, Any]:
        """Perform the KTransformers installation.

        In a real implementation, this would use the project's package manager
        and environment management system. For now, we return a guidance dict.

        Returns
        -------
        dict
            - "success": bool
            - "error": str | None  # if failed
        """
        # This is a placeholder — actual installation would use:
        # - pip install ktransformers[optional deps]
        # - pip install kt-kernel sglang-kt
        # - System package checks (ROCm, CUDA versions)
        # - Validation after install

        # For now, just return guidance
        return {
            "success": False,
            "error": "KTransformers installation not automated in this version. "
            "Please install manually: pip install ktransformers kt-kernel sglang-kt",
        }

    def _select_auto(
        self, model_caps: ModelCapabilities, hardware: HardwareInfo
    ) -> tuple[InferenceBackend, dict[str, Any]]:
        """Auto-select the best backend based on model, hardware, and availability.

        The decision process follows the specification:
        Model → Architecture detection → Model requirements → Hardware detection
        → Installed backend detection → Compatibility checks → Backend selection
        → Install if required → Verify → Load model

        Key: Never use simplistic rules such as "MoE = KTransformers, Dense = AirLLM".
        Instead, verify actual compatibility via backend.check_model().
        """
        # ------------------------------------------------------------------
        # Step 1: Check installed backends' health and compatibility
        # ------------------------------------------------------------------
        airllm_backend = self._registry.get("airllm")
        if airllm_backend is None:
            from airllm_backend import AirLLMBackend as _Aib

            airllm_backend = _Aib(base_url=self._airllm_base_url)
            self._registry.register("airllm", airllm_backend)

        # Check AirLLM health
        airllm_health = airllm_backend.health_check()
        airllm_available = airllm_health.get("installed", False)

        # Check KTransformers availability
        kt_backend = self._registry.get("ktransformers")

        # ------------------------------------------------------------------
        # Step 2: Evaluate compatibility for each backend
        # ------------------------------------------------------------------
        airllm_compatible = False
        ktransformers_compatible = False

        if airllm_available:
            airllm_model_check = airllm_backend.check_model(model_caps)
            airllm_compatible = airllm_model_check.get("compatible", False)

        if kt_backend is not None:
            kt_health = kt_backend.health_check()
            if kt_health.get("installed", False):
                kt_model_check = kt_backend.check_model(model_caps)
                ktransformers_compatible = kt_model_check.get("compatible", False)

        # ------------------------------------------------------------------
        # Step 3: Decision logic — verify, don't assume
        # ------------------------------------------------------------------
        # Priority 1: Explicitly compatible backend with available hardware
        if airllm_available and airllm_compatible:
            # Always verify VRAM/context fit conservatively
            vram_check = check_gpu_memory_fitness(hardware, model_caps)
            if vram_check["fit"]:
                selection_info = {
                    "mode": "airllm",
                    "reason": "AirLLM selected — model compatible and fits within server capabilities",
                    "alternatives": (
                        [
                            {
                                "backend": "ktransformers",
                                "reason": "Alternative for MoE models",
                                "install_required": model_caps.is_moe,
                            }
                        ]
                        if model_caps.is_moe
                        else []
                    ),
                    "install_required": False,
                }
                return airllm_backend, selection_info

        # Priority 2: KTransformers if model is MoE and compatible
        if model_caps.is_moe and kt_backend is not None and ktransformers_compatible:
            # Verify GPU/VRAM fitness conservatively
            vram_check = check_gpu_memory_fitness(hardware, model_caps)
            if vram_check["fit"]:
                selection_info = {
                    "mode": "ktransformers",
                    "reason": "KTransformers selected — MoE model compatible with hardware",
                    "alternatives": [],
                    "install_required": False,
                }
                return kt_backend, selection_info

        # Priority 3: KTransformers for large models with suitable hardware
        if model_caps.parameter_count and model_caps.parameter_count > 30_000_000_000 and kt_backend is not None and ktransformers_compatible:
            vram_check = check_gpu_memory_fitness(hardware, model_caps)
            if vram_check["fit"]:
                selection_info = {
                    "mode": "ktransformers",
                    "reason": "KTransformers selected — large model compatible with hardware",
                    "alternatives": [],
                    "install_required": False,
                }
                return kt_backend, selection_info

        # Priority 4: Fall back to AirLLM if available and compatible (conservative)
        if airllm_available and airllm_compatible:
            selection_info = {
                "mode": "airllm",
                "reason": "AirLLM selected as fallback — broadly compatible with detected model",
                "alternatives": (
                    [
                        {
                            "backend": "ktransformers",
                            "reason": "For MoE models — install KTransformers",
                            "install_required": True,
                        }
                    ]
                    if model_caps.is_moe
                    else []
                ),
                "install_required": False,
            }
            return airllm_backend, selection_info

        # Priority 5: No backend compatible — raise error
        raise BackendSelectionError(
            "No compatible backend found for the given model and hardware. "
            "This may be because:\n"
            "• The model architecture is not supported by any installed backend\n"
            "• Hardware requirements (VRAM, RAM, GPU) are insufficient\n"
            "• Required dependencies are missing\n"
            "• The model's quantization format is not supported\n"
            "Try: --backend airllm for AirLLM server, or install KTransformers "
            "for MoE model support.",
            suggested_action=None,
        )

    # ------------------------------------------------------------------
    # Diagnostic/doctor mode
    # ------------------------------------------------------------------
    def diagnose(self, model_path: str | None = None) -> dict[str, Any]:
        """Run diagnostics for model backend compatibility.

        Useful for the "ai-maker doctor" or "ai-maker backend --check MODEL"
        commands described in the specification.

        Returns
        -------
        dict
            Structured diagnostics report.
        """
        report: dict[str, Any] = {
            "hardware": None,
            "model": None,
            "backends": {},
            "decision": None,
        }

        # Hardware
        hw = detect_hardware()
        report["hardware"] = hw.to_dict()

        # Model
        if model_path:
            model_caps = detect_model_capabilities(model_path)
            report["model"] = model_caps.to_dict()
        # (no model_path -> nothing to detect; skip)

        # Backends status
        registry = get_registry()
        for name in registry.list():
            backend = registry.get(name)
            if backend is not None:
                report["backends"][name] = {
                    "health": backend.health_check(),
                    "dependencies": backend.check_dependencies(),
                    "capabilities": backend.get_capabilities().to_dict(),
                }

        # Decision
        if model_path and model_caps:
            try:
                selected, info = self.select_backend(model_path)
                report["decision"] = {
                    "selected_backend": selected.__class__.__name__,
                    "mode": info["mode"],
                    "reason": info["reason"],
                }
            except BackendSelectionError as e:
                report["decision"] = {
                    "error": str(e),
                    "suggested_action": e.suggested_action,
                }

        return report