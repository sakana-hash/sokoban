"""Device selection for the optional neural-network entry points."""

import torch


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请检查服务器的 NVIDIA 驱动和 PyTorch CUDA 版本。")
    return torch.device(name)
