"""Runtime capability checks for benchmark execution."""

from __future__ import annotations

import importlib.util

import torch

from llm_phase_bench.console import console


def detect_device() -> str:
    """Detect the best available compute device.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        console.print("[bold green]Device:[/] CUDA detected")
        return "cuda"
    if torch.backends.mps.is_available():
        console.print("[bold green]Device:[/] MPS detected (Apple Silicon)")
        return "mps"
    console.print("[bold yellow]Device:[/] No GPU — using CPU")
    return "cpu"


def require_cuda() -> None:
    """Raise if CUDA is not available."""
    if not torch.cuda.is_available():
        msg = "CUDA is required but torch.cuda.is_available() returned False"
        raise RuntimeError(msg)


def require_flash_attention() -> None:
    """Raise if flash-attn package is not installed."""
    if importlib.util.find_spec("flash_attn") is None:
        msg = (
            "flash_attention_2 requested but flash-attn is not installed. "
            "Install with: uv pip install flash-attn"
        )
        raise RuntimeError(msg)


def require_bitsandbytes() -> None:
    """Raise if bitsandbytes package is not installed."""
    if importlib.util.find_spec("bitsandbytes") is None:
        msg = (
            "INT4 quantization requested but bitsandbytes is not installed. "
            "Install with: uv pip install bitsandbytes"
        )
        raise RuntimeError(msg)


def device_info() -> dict[str, object]:
    """Collect device diagnostics for logging / debugging."""
    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "flash_attn_available": importlib.util.find_spec("flash_attn") is not None,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info
