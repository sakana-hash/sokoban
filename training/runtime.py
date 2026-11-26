"""Device selection for the optional neural-network entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def checkpoint_version(checkpoint: dict) -> int:
    if isinstance(checkpoint, dict) and checkpoint.get("version") in (1, 2, 3):
        return checkpoint["version"]
    raise ValueError("无法识别检查点版本")


def select_device(name: str) -> torch.device:
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请检查服务器的 NVIDIA 驱动和 PyTorch CUDA 版本。")
    return torch.device(name)
