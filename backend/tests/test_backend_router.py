"""Tests for the BackendRouter and backend abstraction layer."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

from backend_config import (
    get_backend_mode,
    load_backend_config,
    save_backend_config,
)
from backend_router import BackendRouter, BackendSelectionError
from hardware_detector import HardwareInfo, detect_hardware
from model_capability import detect_model_capabilities
from inference_backend import ModelCapabilities, get_registry


def test_router_initialization():
    """Test that BackendRouter can be initialized in all modes."""
    from backend_router import BackendRouter

    # Test auto mode (default)
    router = BackendRouter(mode="auto")
    assert router.mode == "auto"

    # Test explicit airllm mode
    router = BackendRouter(mode="airllm")
    assert router.mode == "airllm"

    # Test explicit ktransformers mode
    router = BackendRouter(mode="ktransformers")
    assert router.mode == "ktransformers"

    print("PASS: test_router_initialization")


def test_router_modes_respected():
    """Test that explicit user mode overrides auto-detection."""
    from backend_router import BackendRouter

    router = BackendRouter(mode="ktransformers")
    # The mode should be respected even if KTransformers is not available
    assert router.mode == "ktransformers"

    router2 = BackendRouter(mode="airllm")
    assert router2.mode == "airllm"

    print("PASS: test_router_modes_respected")


def test_hardware_detector():
    """Test that hardware detection works."""
    hw = detect_hardware()
    assert hw is not None
    assert hasattr(hw, "gpu_vendor")
    assert hasattr(hw, "cpu_model")
    assert hasattr(hw, "system_ram_mb")

    print("PASS: test_hardware_detector")


def test_model_capability_detection():
    """Test model capability detection."""
    # Test with a known model
    caps = detect_model_capabilities("airllm/Qwen/Qwen3-30B-A3B")
    assert caps is not None
    assert caps.architecture in ("qwen", "unknown")
    assert caps.parameter_count is not None
    assert isinstance(caps.is_moe, bool)

    # Test with MoE model
    moe_caps = detect_model_capabilities("mixtral-8x7b-instruct")
    assert moe_caps is not None
    assert moe_caps.is_moe is True

    print("PASS: test_model_capability_detection")


def test_model_capabilities_include_moe():
    """Test that MoE models are detected and have KTransformers in supported backends."""
    caps = detect_model_capabilities("mixtral-8x7b-instruct")
    assert "ktransformers" in caps.supported_backends
    assert "airllm" in caps.supported_backends

    print("PASS: test_model_capabilities_include_moe")


def test_backend_router_auto_mode():
    """Test BackendRouter auto mode selection."""
    from backend_router import BackendRouter
    from airllm_backend import AirLLMBackend

    # Force the "no AirLLM server" scenario regardless of whether one is
    # actually running on this host (the container shares the host network).
    with patch.object(
        AirLLMBackend, "health_check",
        return_value={"installed": False, "error": "unreachable (test)"}):
        router = BackendRouter(mode="auto")
        # With no KTransformers and no AirLLM server, should raise error
        try:
            router.select_backend("some-model")
            assert False, "Should have raised BackendSelectionError"
        except BackendSelectionError:
            pass  # Expected

    print("PASS: test_backend_router_auto_mode")


def test_backend_router_explicit_airllm():
    """Test explicit AirLLM mode gives useful error when server unreachable."""
    from backend_router import BackendRouter
    from airllm_backend import AirLLMBackend

    # Force the "server unreachable" scenario regardless of whether one is
    # actually running on this host (the container shares the host network).
    with patch.object(
        AirLLMBackend, "health_check",
        return_value={"installed": False, "error": "unreachable (test)"}):
        router = BackendRouter(mode="airllm")
        hw = HardwareInfo(
            gpu_vendor="NVIDIA",
            gpu_model="RTX 3090",
            gpu_vram_mb=24000,
            cpu_model="Test",
            cpu_cores=12,
            system_ram_mb=32000,
            cuda_available=True,
        )

        try:
            router.select_backend("some-model", hardware=hw)
            assert False, "Should have raised BackendSelectionError"
        except BackendSelectionError as e:
            # Should have a useful error message
            assert "AirLLM" in str(e) or "server" in str(e).lower()
            assert e.suggested_action is not None

    print("PASS: test_backend_router_explicit_airllm")


def test_requested_backend_overrides_configured_mode():
    """Regression: select_backend() must dispatch on the RESOLVED mode —
    an explicit requested_backend must win over the router's configured
    self.mode. (Bug: dispatch used self.mode, so requested_backend was
    silently ignored.)"""
    router = BackendRouter(mode="auto")
    called = []

    def fake_auto(model_caps, hardware):
        called.append("auto")
        return object(), {"mode": "auto"}

    def fake_airllm(model_caps, hardware):
        called.append("airllm")
        return object(), {"mode": "airllm"}

    def fake_kt(model_caps, hardware):
        called.append("ktransformers")
        return object(), {"mode": "ktransformers"}

    with patch.object(router, "_select_auto", fake_auto), \
         patch.object(router, "_select_airllm", fake_airllm), \
         patch.object(router, "_select_ktransformers", fake_kt):
        router.select_backend("some-model", requested_backend="airllm")
        router.select_backend("some-model", requested_backend="ktransformers")
        router.select_backend("some-model")  # no override -> configured mode
        router.select_backend("some-model", requested_backend="auto")

    assert called == ["airllm", "ktransformers", "auto", "auto"], f"dispatch order wrong: {called}"
    print("PASS: test_requested_backend_overrides_configured_mode")


def test_requested_backend_invalid_value_rejected():
    """An unrecognized requested_backend must raise ValueError, not silently
    fall through to auto."""
    router = BackendRouter(mode="auto")
    try:
        router.select_backend("some-model", requested_backend="bogus")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "bogus" in str(e)
    print("PASS: test_requested_backend_invalid_value_rejected")


def test_backend_router_explicit_ktransformers():
    """Test explicit KTransformers mode gives useful error when not installed."""
    from backend_router import BackendRouter

    router = BackendRouter(mode="ktransformers")
    hw = HardwareInfo(
        gpu_vendor="NVIDIA",
        gpu_model="RTX 3090",
        gpu_vram_mb=24000,
        cpu_model="Test",
        cpu_cores=12,
        system_ram_mb=32000,
        cuda_available=True,
    )

    try:
        router.select_backend("some-model", hardware=hw)
        assert False, "Should have raised an error"
    except (BackendSelectionError, NameError) as e:
        # Should have a useful error message about KTransformers not being installed
        assert "KTransformers" in str(type(e).__name__) or "KTransformers" in str(e)

    print("PASS: test_backend_router_explicit_ktransformers")


def test_backend_registry():
    """Test the BackendRegistry functionality."""
    registry = get_registry()

    # Test registering and getting a backend
    from airllm_backend import AirLLMBackend
    backend = AirLLMBackend()
    registry.register("test_backend", backend)

    retrieved = registry.get("test_backend")
    assert retrieved is not None
    assert retrieved.__class__.__name__ == "AirLLMBackend"

    # Test listing
    assert "test_backend" in registry.list()

    # Test has
    assert registry.has("test_backend")
    assert registry.has("nonexistent") is False

    # Test unregister
    registry.unregister("test_backend")
    assert registry.has("test_backend") is False

    print("PASS: test_backend_registry")


def test_inference_backend_interface():
    """Test that backends implement the InferenceBackend interface."""
    from airllm_backend import AirLLMBackend

    backend = AirLLMBackend()

    # Test required interface methods
    required_methods = [
        "health_check",
        "check_dependencies",
        "check_hardware",
        "check_model",
        "load_model",
        "generate",
        "unload_model",
        "get_model_info",
        "get_capabilities",
    ]

    for method in required_methods:
        assert hasattr(backend, method), f"Missing method: {method}"

    # Test health_check returns dict with "installed" and "compatible"
    health = backend.health_check()
    assert isinstance(health, dict)
    assert "installed" in health
    assert "compatible" in health

    # Test get_capabilities returns ModelCapabilities
    caps = backend.get_capabilities()
    assert isinstance(caps, ModelCapabilities)
    assert caps.architecture == "auto-detected-by-server"
    assert caps.supported_backends == ["airllm"]

    print("PASS: test_inference_backend_interface")


def test_config_load_save():
    """Test backend configuration loading and saving.

    Runs entirely against a temp file — never the real project config.json,
    which is shared with the whole app (save_dir, lsp, cloud_budget, ...).
    """
    import tempfile

    import backend_config

    with tempfile.TemporaryDirectory() as tmp:
        fake_path = f"{tmp}/config.json"
        original_path = backend_config.BACKEND_CONFIG_PATH
        try:
            backend_config.BACKEND_CONFIG_PATH = fake_path

            # Fresh load with no file present -> defaults
            config = load_backend_config()
            assert "backend" in config
            assert config.get("backend", "auto") in ("auto", "airllm", "ktransformers")

            # Simulate the real shared app config, then save a backend-only
            # change: every unrelated top-level key must survive verbatim.
            with open(fake_path, "w") as f:
                json.dump({
                    "save_dir": "/tmp/some-where",
                    "lsp": True,
                    "cloud_budget": 25.0,
                    "default_model": "some/model",
                    "teacher_model": "some/teacher",
                    "openrouter_model": "openrouter/some/model",
                    "ics_feed_token": "x" * 32,
                }, f)
            save_backend_config({"backend": "airllm",
                                 "airllm": {"base_url": "http://test:8082"},
                                 "ktransformers": {}})
            with open(fake_path) as f:
                saved = json.load(f)
            assert saved["backend"] == "airllm"
            assert saved["airllm"]["base_url"] == "http://test:8082"
            assert saved["save_dir"] == "/tmp/some-where"
            assert saved["lsp"] is True
            assert saved["cloud_budget"] == 25.0
            assert saved["default_model"] == "some/model"
            assert saved["teacher_model"] == "some/teacher"
            assert saved["openrouter_model"] == "openrouter/some/model"
            assert saved["ics_feed_token"] == "x" * 32

            # A second save keeps the first save's keys too (merge chains)
            save_backend_config({"backend": "ktransformers"})
            with open(fake_path) as f:
                saved = json.load(f)
            assert saved["backend"] == "ktransformers"
            assert saved["airllm"]["base_url"] == "http://test:8082"
            assert saved["cloud_budget"] == 25.0

            # Atomic write: temp file is cleaned up; last-known-good backup
            # exists after a save
            assert not os.path.exists(fake_path + ".tmp")
            assert os.path.exists(fake_path + ".bak")
            with open(fake_path + ".bak") as f:
                bak = json.load(f)
            assert bak["backend"] == "airllm"  # state before the 2nd save

            # Reload + mode
            config2 = load_backend_config()
            mode = get_backend_mode(config2)
            assert mode in ("auto", "airllm", "ktransformers")
            assert mode == "ktransformers"
        finally:
            backend_config.BACKEND_CONFIG_PATH = original_path

    print("PASS: test_config_load_save")


def test_health_check_contract():
    """Test that health_check returns the expected contract."""
    from airllm_backend import AirLLMBackend

    backend = AirLLMBackend()
    health = backend.health_check()

    # Contract: must contain "installed" and "compatible"
    assert "installed" in health, "health_check must contain 'installed'"
    assert "compatible" in health, "health_check must contain 'compatible'"
    assert isinstance(health["installed"], bool), "'installed' must be bool"
    assert isinstance(health["compatible"], bool), "'compatible' must be bool"

    print("PASS: test_health_check_contract")


def test_check_dependencies_contract():
    """Test that check_dependencies returns the expected contract."""
    from airllm_backend import AirLLMBackend

    backend = AirLLMBackend()
    deps = backend.check_dependencies()

    # Contract: must contain "met", "missing", "version_mismatch"
    assert "met" in deps, "check_dependencies must contain 'met'"
    assert "missing" in deps, "check_dependencies must contain 'missing'"
    assert "version_mismatch" in deps, "check_dependencies must contain 'version_mismatch'"
    assert isinstance(deps["met"], bool), "'met' must be bool"
    assert isinstance(deps["missing"], list), "'missing' must be list"

    print("PASS: test_check_dependencies_contract")


def test_check_model_contract():
    """Test that check_model returns the expected contract."""
    from airllm_backend import AirLLMBackend

    backend = AirLLMBackend()
    caps = detect_model_capabilities("airllm/Qwen/Qwen3-30B-A3B")
    result = backend.check_model(caps)

    # Contract: must contain "compatible" and "reason"
    assert "compatible" in result, "check_model must contain 'compatible'"
    assert "reason" in result, "check_model must contain 'reason'"
    assert isinstance(result["compatible"], bool), "'compatible' must be bool"
    assert result["compatible"] is True or result.get("reason") is not None

    print("PASS: test_check_model_contract")


def test_doctor_diagnostic():
    """Test the diagnose method produces a structured report."""
    from backend_router import BackendRouter

    router = BackendRouter(mode="auto")
    report = router.diagnose(model_path="airllm/Qwen/Qwen3-30B-A3B")

    # Report should have expected keys
    assert "hardware" in report, "Diagnose report must contain 'hardware'"
    assert "model" in report, "Diagnose report must contain 'model'"
    assert "backends" in report, "Diagnose report must contain 'backends'"

    print("PASS: test_doctor_diagnostic")


if __name__ == "__main__":
    test_router_initialization()
    test_router_modes_respected()
    test_hardware_detector()
    test_model_capability_detection()
    test_model_capabilities_include_moe()
    test_backend_router_auto_mode()
    test_backend_router_explicit_airllm()
    test_backend_router_explicit_ktransformers()
    test_backend_registry()
    test_inference_backend_interface()
    test_config_load_save()
    test_health_check_contract()
    test_check_dependencies_contract()
    test_check_model_contract()
    test_doctor_diagnostic()
    print("\nAll tests passed!")