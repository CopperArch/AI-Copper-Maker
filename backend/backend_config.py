from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


#: Default backend configuration
DEFAULT_BACKEND_CONFIG: dict[str, Any] = {
    "backend": "auto",
    "airllm": {
        "base_url": "http://localhost:8082",
    },
    "ktransformers": {
        "base_url": "http://localhost:8083",
        "force_install": False,
        "unattended_install": False,
    },
}

#: Path to the backend configuration file (relative to project root)
BACKEND_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config.json"
)


def load_backend_config() -> dict[str, Any]:
    """Load the backend configuration from the project config.json.

    The config.json is expected to have a "backend" key at the top level,
    with optional subsections for each backend (e.g. "airllm", "ktransformers").

    Returns
    -------
    dict
        The merged backend configuration, falling back to defaults where
        not specified.
    """
    # config.json holds the WHOLE app config (save_dir, lsp, cloud_budget,
    # ...) — read the top-level keys only, never assume the file is ours.
    try:
        with open(BACKEND_CONFIG_PATH, "r") as f:
            user_config = json.load(f)
        if not isinstance(user_config, dict):
            raise ValueError("top-level JSON must be an object")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "Backend config file %s not found or invalid (%s); using defaults",
            BACKEND_CONFIG_PATH, e,
        )
        user_config = {}

    # Start with defaults
    config = dict(DEFAULT_BACKEND_CONFIG)

    # Merge user config on top
    user_backend = user_config.get("backend")
    if user_backend:
        config["backend"] = user_backend

    # Merge backend-specific subsections
    for backend_name in ("airllm", "ktransformers"):
        user_be = user_config.get(backend_name, {})
        if isinstance(user_be, dict):
            config_sect = config.get(backend_name, {})
            config_sect.update(user_be)
            config[backend_name] = config_sect

    return config


def save_backend_config(config: dict[str, Any]) -> None:
    """Save the backend configuration into the project config.json.

    config.json is shared with the rest of the app (save_dir, lsp,
    cloud_budget, ...), so this MERGES: only the "backend" / "airllm" /
    "ktransformers" top-level keys are updated, every other key is
    preserved verbatim. Writing the backend dict alone would clobber the
    whole app config.

    Parameters
    ----------
    config : dict
        Backend configuration (any subset of backend/airllm/ktransformers).
    """
    try:
        with open(BACKEND_CONFIG_PATH, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            existing = {}
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        existing = {}
    for key in ("backend", "airllm", "ktransformers"):
        if key in config:
            existing[key] = config[key]
    # Same protection as main.py's write_config: keep a last-known-good
    # backup, and replace atomically so a crash can't corrupt the file.
    backup_path = BACKEND_CONFIG_PATH + ".bak"
    tmp_path = BACKEND_CONFIG_PATH + ".tmp"
    try:
        payload = json.dumps(existing, indent=2)
        if os.path.exists(BACKEND_CONFIG_PATH):
            try:
                with open(BACKEND_CONFIG_PATH, "r") as f:
                    current = f.read()
                json.loads(current)
                with open(backup_path, "w") as f:
                    f.write(current)
            except (json.JSONDecodeError, ValueError, OSError):
                pass
        with open(tmp_path, "w") as f:
            f.write(payload)
        os.replace(tmp_path, BACKEND_CONFIG_PATH)
        logger.info("Backend configuration saved to %s", BACKEND_CONFIG_PATH)
    except Exception as e:
        logger.error("Failed to save backend config: %s", e)


def get_backend_mode(config: dict[str, Any] | None = None) -> str:
    """Get the backend selection mode from the configuration.

    Parameters
    ----------
    config : dict | None
        Backend config dict. If None, loads from file.

    Returns
    -------
    str
        One of "auto", "airllm", "ktransformers".
    """
    if config is None:
        config = load_backend_config()
    return config.get("backend", "auto")


def get_airllm_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get the AirLLM backend configuration.

    Parameters
    ----------
    config : dict | None
        Backend config dict. If None, loads from file.

    Returns
    -------
    dict
        AirLLM-specific configuration (base_url, etc.).
    """
    if config is None:
        config = load_backend_config()
    return config.get("airllm", {"base_url": "http://localhost:8082"})


def get_ktransformers_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get the KTransformers backend configuration.

    Parameters
    ----------
    config : dict | None
        Backend config dict. If None, loads from file.

    Returns
    -------
    dict
        KTransformers-specific configuration (base_url, force_install, etc.).
    """
    if config is None:
        config = load_backend_config()
    return config.get("ktransformers", {"base_url": "http://localhost:8083", "force_install": False, "unattended_install": False})


def suggest_backend(model_path: str, hardware_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Suggest the best backend for a given model, using the router and hardware info.

    This is a convenience function that combines model capability detection
    with hardware detection and router selection to produce a recommendation.

    Parameters
    ----------
    model_path : str
        Model identifier/path.
    hardware_config : dict | None
        Hardware specification. If None, hardware will be detected automatically.

    Returns
    -------
    dict
        Suggestion containing:
        - "selected_backend": str ("airllm", "ktransformers", "auto")
        - "reason": str
        - "install_ktransformers": bool
        - "alternatives": list
    """
    from backend_router import BackendRouter, HardwareInfo

    # Detect hardware if not provided
    if hardware_config is None:
        from hardware_detector import detect_hardware
        hw = detect_hardware()
    else:
        hw = HardwareInfo(**hardware_config)

    # Use the router to select
    router = BackendRouter(mode="auto", ktransformers_force_install=False, unattended_install=False)
    try:
        selected_backend, selection_info = router.select_backend(model_path, hardware=hw)
        return {
            "selected_backend": selection_info["mode"],
            "reason": selection_info["reason"],
            "install_ktransformers": selection_info.get("install_required", False),
            "alternatives": selection_info.get("alternatives", []),
        }
    except Exception as e:
        return {
            "selected_backend": "airllm",
            "reason": f"Router error: {e}; falling back to AirLLM",
            "install_ktransformers": False,
            "alternatives": [],
        }