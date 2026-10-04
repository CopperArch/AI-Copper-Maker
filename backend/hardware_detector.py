from __future__ import annotations

import glob
import json
import logging
import os
import shutil
from typing import Any

from inference_backend import ModelCapabilities

logger = logging.getLogger(__name__)


class HardwareInfo:
    """Detected hardware specifications."""

    def __init__(
        self,
        gpu_vendor: str | None = None,
        gpu_model: str | None = None,
        gpu_vram_mb: int | None = None,
        cpu_model: str | None = None,
        cpu_cores: int | None = None,
        system_ram_mb: int | None = None,
        cuda_available: bool = False,
        cuda_version: str | None = None,
        rocm_available: bool = False,
        rocm_version: str | None = None,
    ) -> None:
        self.gpu_vendor = gpu_vendor
        self.gpu_model = gpu_model
        self.gpu_vram_mb = gpu_vram_mb
        self.cpu_model = cpu_model
        self.cpu_cores = cpu_cores
        self.system_ram_mb = system_ram_mb
        self.cuda_available = cuda_available
        self.cuda_version = cuda_version
        self.rocm_available = rocm_available
        self.rocm_version = rocm_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_vendor": self.gpu_vendor,
            "gpu_model": self.gpu_model,
            "gpu_vram_mb": self.gpu_vram_mb,
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "system_ram_mb": self.system_ram_mb,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "rocm_available": self.rocm_available,
            "rocm_version": self.rocm_version,
        }

    @property
    def has_gpu(self) -> bool:
        return self.gpu_vendor is not None and self.gpu_vram_mb is not None

    @property
    def has_cuda(self) -> bool:
        return self.cuda_available

    @property
    def has_rocm(self) -> bool:
        return self.rocm_available


def _sysfs_display_gpus() -> list[dict[str, Any]]:
    """Display adapters from /sys/class/drm: vendor, PCI id, VRAM in MiB.

    Works without lspci (which the app's container does not ship) and skips
    iGPUs by reporting every card so callers can pick the largest.
    """
    gpus: list[dict[str, Any]] = []
    for path in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            with open(os.path.join(path, "vendor")) as f:
                vendor = int(f.read().strip(), 16)
            vram_mb = 0
            try:
                with open(os.path.join(path, "mem_info_vram_total")) as f:
                    vram_mb = int(f.read().strip()) // (1024 * 1024)
            except (OSError, ValueError):
                pass
            pci_id = ""
            try:
                with open(os.path.join(path, "uevent")) as f:
                    for line in f:
                        if line.startswith("PCI_ID="):
                            pci_id = line.split("=", 1)[1].strip()
            except OSError:
                pass
            gpus.append({"vendor": vendor, "vram_mb": vram_mb, "pci_id": pci_id})
        except (OSError, ValueError):
            continue
    return gpus


def _rocm_toolkit_present() -> bool:
    """True when a ROCm/HIP toolchain is installed, with or without rocm-smi."""
    if shutil.which("rocminfo") or shutil.which("rocm-smi"):
        return True
    if shutil.which("hipcc") or os.path.exists("/opt/rocm"):
        return True
    return any(glob.glob("/usr/lib*/libamdhip64.so*"))


def detect_hardware() -> HardwareInfo:
    """Detect available hardware: GPU, CPU, system RAM, CUDA/ROCm.

    Returns
    -------
    HardwareInfo
        A HardwareInfo instance populated with detected specifications.
        GPU/VRAM detection uses PyTorch if available, falls back to
        lspci/sysfs reading. CPU and RAM from /proc and os.sysconf.
    """
    # Default: unknown/conservative
    gpu_vendor = None
    gpu_model = None
    gpu_vram_mb = None
    cpu_model = None
    cpu_cores = None
    system_ram_mb = None
    cuda_available = False
    cuda_version = None
    rocm_available = False
    rocm_version = None

    # --- System RAM ---
    try:
        total_kb = os.sysconf("SC_AVAIL_MEMORY") * os.sysconf("SC_PAGE_SIZE") // 1024
        system_ram_mb = total_kb
    except (ValueError, AttributeError):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        system_ram_mb = total_kb // 1024
                        break
        except FileNotFoundError:
            system_ram_mb = None

    # --- CPU ---
    try:
        with open("/proc/cpuinfo", "r") as f:
            model_name = None
            core_count = 0
            for line in f:
                if line.startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    core_count += 1
            cpu_model = model_name or "Unknown CPU"
            cpu_cores = core_count
    except FileNotFoundError:
        cpu_model = "Unknown CPU"
        cpu_cores = os.cpu_count() or None

    # --- GPU detection ---
    # Try PyTorch first (CUDA or ROCm)
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            cuda_version = torch.cuda.version()
            gpu_model = torch.cuda.get_device_name(0)
            # A ROCm torch build also exposes devices through torch.cuda, so
            # derive the vendor from the device name instead of assuming NVIDIA.
            gpu_vendor = (
                "AMD"
                if any(t in gpu_model for t in ("AMD", "Radeon", "Instinct"))
                else "NVIDIA"
            )
            gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        elif hasattr(torch, "rocm") and torch.cuda.is_available() is False:
            # ROCm path - check if ROCm is available
            rocm_available = True
            try:
                rocm_version = torch.version.rocm
            except AttributeError:
                pass
    except ImportError:
        pass

    # --- Fallback: display adapters from sysfs (works without lspci, e.g. in the container) ---
    if gpu_vendor is None or gpu_model is None:
        gpus = _sysfs_display_gpus()
        if gpus:
            biggest = max(gpus, key=lambda g: g["vram_mb"])
            if biggest["vendor"] == 0x1002:
                gpu_vendor = "AMD"
            elif biggest["vendor"] == 0x10de:
                gpu_vendor = "NVIDIA"
            if gpu_model is None:
                if biggest["pci_id"]:
                    gpu_model = f"{gpu_vendor} GPU (PCI {biggest['pci_id']})"
                else:
                    gpu_model = f"{gpu_vendor} GPU"
            if gpu_vram_mb is None and biggest["vram_mb"] > 0:
                gpu_vram_mb = biggest["vram_mb"]

    # The ROCm toolkit can be installed without rocm-smi/rocminfo (hipcc-only
    # or bare HIP runtime); report it so routing/fitness checks can rely on it.
    if not rocm_available and _rocm_toolkit_present():
        rocm_available = True

    # If we still have no GPU info but torch found something, use that
    if gpu_vendor is None and cuda_available:
        gpu_vendor = "NVIDIA"
        try:
            import torch

            gpu_model = torch.cuda.get_device_name(0)
            gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        except Exception:
            pass

    # If we still have no GPU info and rocm was detected via torch
    if gpu_vendor is None and rocm_available:
        gpu_vendor = "AMD"
        try:
            import torch

            gpu_model = "AMD GPU (ROCm)"
        except Exception:
            pass

    # Build the HardwareInfo
    info = HardwareInfo(
        gpu_vendor=gpu_vendor,
        gpu_model=gpu_model,
        gpu_vram_mb=gpu_vram_mb,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        system_ram_mb=system_ram_mb,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        rocm_available=rocm_available,
        rocm_version=rocm_version,
    )

    logger.info(
        "Detected hardware: %s",
        json.dumps(info.to_dict(), default=str),
    )
    return info


def check_gpu_memory_fitness(
    hardware: HardwareInfo, capabilities: ModelCapabilities
) -> dict[str, Any]:
    """Check if the available GPU VRAM is sufficient for the model.

    Uses a conservative estimation:
    - FP16/BF16: 2 bytes per parameter
    - INT8: 1 byte per parameter
    - GPTQ/AWQ: ~0.5-1x parameter count depending on sparsity

    Returns
    -------
    dict
        - "fit": bool  # whether VRAM is sufficient
        - "required_mb": int  # estimated required VRAM in MB
        - "available_mb": int | None  # available GPU VRAM in MB, or None
        - "reason": str | None  # human-readable reason if not fit
    """
    available_mb = hardware.gpu_vram_mb
    if available_mb is None:
        return {
            "fit": False,
            "required_mb": capabilities.approximate_memory_mb or 0,
            "available_mb": None,
            "reason": "GPU VRAM not detected",
        }

    required_mb = capabilities.approximate_memory_mb or 0

    # Apply quantization multiplier for rough estimation
    if capabilities.quantization == "INT8":
        # INT8 typically needs ~1 byte per parameter
        estimated_mb = max(required_mb, 1)  # already in MB
    elif capabilities.quantization in ("GPTQ", "AWQ"):
        # GPTQ/AWQ needs ~0.5-1x, we'll use 1x conservatively
        estimated_mb = max(required_mb, 1)
    else:  # FP16, BF16, FP32 default
        # FP16/BF16 needs ~2 bytes per parameter
        estimated_mb = max(required_mb * 2, 1)

    fit = estimated_mb <= available_mb

    reason = None
    if not fit:
        reason = (
            f"Model requires ~{estimated_mb}MB VRAM but only "
            f"{available_mb}MB is available"
        )

    return {
        "fit": fit,
        "required_mb": estimated_mb,
        "available_mb": available_mb,
        "reason": reason,
    }