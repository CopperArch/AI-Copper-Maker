from __future__ import annotations

import json
import logging
import os
import re
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
            gpu_vendor = "NVIDIA"
            gpu_model = torch.cuda.get_device_name(0)
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

    # --- Fallback: lspci for GPU info (if PyTorch not available or no GPU found) ---
    if gpu_vendor is None or gpu_model is None:
        try:
            lspci_output = subprocess_check(["lspci", "-nn"]).strip()
            # Look for NVIDIA GPUs
            nvidia_match = re.search(
                r"(\d+:\d+\.\d+).+?NVIDIA [^\n]+",
                lspci_output,
                re.IGNORECASE,
            )
            if nvidia_match:
                gpu_vendor = "NVIDIA"
                gpu_model = nvidia_match.group(0).split("NVIDIA")[1].strip()
            # Look for AMD/ATI GPUs
            amd_match = re.search(
                r"(\d+:\d+\.\d+).+?Radeon [^\n]+",
                lspci_output,
                re.IGNORECASE,
            )
            if amd_match:
                gpu_vendor = "AMD"
                gpu_model = amd_match.group(0).split("Radeon")[1].strip()
            # Look for VRAM in lspci output (requires smbios or other methods)
            # For now, leave gpu_vram_mb as None if detected only via lspci
        except Exception:
            pass  # lspci may not be available; we fall back to conservative defaults

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


def subprocess_check(cmd: list[str]) -> str:
    """Run a subprocess command and return stdout.

    A small helper to avoid a top-level import of subprocess in this module
    when not needed, but still support lspci fallback.
    """
    import subprocess

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


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